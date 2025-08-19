import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# ----------------- U-Net Generator -----------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = self._down_block(in_channels, 64, batchnorm=False)
        self.enc2 = self._down_block(64, 128)
        self.enc3 = self._down_block(128, 256)
        self.enc4 = self._down_block(256, 512)
        self.enc5 = self._down_block(512, 512, dropout=0.5)

        self.dec1 = self._up_block(512 + 512, 512, dropout=0.5)
        self.dec2 = self._up_block(512 + 256, 256)
        self.dec3 = self._up_block(256 + 128, 128)
        self.dec4 = self._up_block(128 + 64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, 3, 1, 1),
            nn.Tanh()
        )

    def _down_block(self, in_c, out_c, batchnorm=True, dropout=0.0):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _up_block(self, in_c, out_c, dropout=0.0):
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d1 = self.dec1(torch.cat([F.interpolate(e5, size=e4.shape[2:]), e4], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d1, size=e3.shape[2:]), e3], dim=1))
        d3 = self.dec3(torch.cat([F.interpolate(d2, size=e2.shape[2:]), e2], dim=1))
        d4 = self.dec4(torch.cat([F.interpolate(d3, size=e1.shape[2:]), e1], dim=1))

        return self.final(d4)

class GANProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {
            'FullMask': self.load_model('models/fullmask_gan/generator.pth'),
            'EdgeOnly': self.load_model('models/contour_gan/generator.pth')
        }
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Must match training
        ])
    
    def load_model(self, model_path):
        """Load pre-trained GAN model"""
        model = UNetGenerator().to(self.device)
        
        # Load state dictionary
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Handle potential issues with state dict keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # Remove 'module.' prefix if present (from DataParallel)
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    
    def apply_gan_style(self, input_dir, style, job_id):
        output_dir = f'workspace/styled/{job_id}/{style}'
        os.makedirs(output_dir, exist_ok=True)
        
        model = self.models.get(style)
        if not model:
            raise ValueError(f"Invalid style: {style}")
        
        # Process all masks in the input directory
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, input_dir)
                    output_class_dir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_class_dir, exist_ok=True)
                    
                    # Process image
                    output_path = os.path.join(output_class_dir, file)
                    self.process_image(img_path, output_path, model)
        
        return output_dir
    
    def process_image(self, input_path, output_path, model):
        # Load and transform image
        img = Image.open(input_path).convert("L")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Apply GAN
        with torch.no_grad():
            output = model(img_tensor)
            output = (output + 1) / 2  # Denormalize to [0,1]
        
        # Convert to image and save
        output = output.squeeze(0).cpu()
        output_img = transforms.ToPILImage()(output)
        output_img.save(output_path)

# Global instance
gan_processor = GANProcessor()

# Module-level function
def apply_gan_style(input_dir, style, job_id):
    return gan_processor.apply_gan_style(input_dir, style, job_id)