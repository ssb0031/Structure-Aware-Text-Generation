import os
import re
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import img2pdf
from flask import send_file

# ------------------------
# CONFIGURATION
# ------------------------
# These will be set dynamically in the Flask app
MASK_ROOT_FULL = None
MASK_ROOT_EDGE = None
INK_COLORS = {'black': (0, 0, 0), 'red': (255, 0, 0), 'blue': (0, 0, 255)}
PUNCTUATION = set('.,;:?!')

# ------------------------
# CONFUSION MAP
# ------------------------
confusion_map = {
    "0": ["O", "o"], "O": ["0"], "o": ["0"],
    "1": ["l", "I"], "I": ["1", "l"], "l": ["1", "I"],
    "5": ["S", "s"], "S": ["5"], "s": ["5"],
    "2": ["Z", "z"], "Z": ["2"], "z": ["2"],
    "9": ["g", "q"], "g": ["9", "q"], "q": ["9", "g"],
    "4": ["y"], "y": ["4"],
    "u": ["v"], "v": ["u"],
    "m": ["n"], "n": ["m"],
    "e": ["E"], "E": ["e"],
    "k": ["K"], "K": ["k"],
    "x": ["X"], "X": ["x"],
    "a": ["A"], "A": ["a"],
    "c": ["C"], "C": ["c"],
    "b": ["B"], "B": ["b"]
}

# ------------------------
# UTILS
# ------------------------
def build_char_to_paths(folder):
    mapping = {}
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".png"):
            continue
        match = re.match(r"Class-([A-Za-z0-9])(?:_|\.png)", fname)
        if not match:
            continue
        char = match.group(1)
        mapping.setdefault(char, []).append(os.path.join(folder, fname))
    return mapping

def get_mask_for_char(ch, style):
    if style == "Edge Only":
        char_to_paths = build_char_to_paths(MASK_ROOT_EDGE)
    else:
        char_to_paths = build_char_to_paths(MASK_ROOT_FULL)
        
    if ch in char_to_paths:
        return random.choice(char_to_paths[ch])
    if ch.lower() in char_to_paths:
        return random.choice(char_to_paths[ch.lower()])
    if ch.upper() in char_to_paths:
        return random.choice(char_to_paths[ch.upper()])
    for alt in confusion_map.get(ch, []):
        if alt in char_to_paths:
            return random.choice(char_to_paths[alt])
    return None

# ------------------------
# COLORIZATION FUNCTION
# ------------------------
def colorize_mask(mask_img, ink_rgb, style="Full Character"):
    mask_np = np.array(mask_img.convert("L"))
    color_img = np.ones((*mask_np.shape, 3), dtype=np.uint8) * 255  # White background

    if style == "Edge Only":
        # 1. Gentle background removal - preserve faint character details
        _, binary_mask = cv2.threshold(mask_np, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Noise reduction while preserving character shape
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)     # Connect broken lines
        
        # 3. Edge detection with optimized parameters
        edges = cv2.Canny(cleaned, threshold1=30, threshold2=70)  # Lower thresholds for more sensitivity
        
        # 4. Gentle dilation to enhance character structure
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 5. Selective contour filtering - preserve all meaningful contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_edges = np.zeros_like(edges)
        for contour in contours:
            # Keep all contours - we're only filtering during display
            cv2.drawContours(final_edges, [contour], -1, 255, 1)

        # Apply ink color to edges
        edge_mask = final_edges > 0
        for i in range(3):
            color_img[:, :, i] = np.where(edge_mask, ink_rgb[i], 255)

    else:  # Full Character style remains unchanged
        character_mask = mask_np < 200
        for i in range(3):
            color_img[:, :, i] = np.where(character_mask, ink_rgb[i], 255)

    return Image.fromarray(color_img)

# ------------------------
# HANDWRITING GENERATOR CLASS
# ------------------------
class HandwritingGenerator:
    def __init__(self, mask_root_full, mask_root_edge):
        global MASK_ROOT_FULL, MASK_ROOT_EDGE
        MASK_ROOT_FULL = mask_root_full
        MASK_ROOT_EDGE = mask_root_edge
        
        self.config = {
            'page_width': 1000,
            'page_height': 1400,
            'margins': {'left': 50, 'right': 50, 'top': 50, 'bottom': 50},
            'line_spacing': 1.5,
            'paragraph_spacing': 0.5,
            'rule_color': (220, 220, 220),
            'font_size': 'medium',  # small/medium/large
            'size_jitter': 0.05,
            'tilt_range': (-10, 10),  # degrees
            'char_spacing_range': (-2, 5),
            'word_spacing_factor': 1.2,
            'stroke_variation': True,
            'ink_color': 'black',
            'style': 'Full Character',
            'page_layout': 'unruled'
        }
    
    def set_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
            elif key == 'margin_left':
                self.config['margins']['left'] = value
            elif key == 'margin_right':
                self.config['margins']['right'] = value
            elif key == 'margin_top':
                self.config['margins']['top'] = value
            elif key == 'margin_bottom':
                self.config['margins']['bottom'] = value
    
    def get_char_size(self):
        """Get base character size with random jitter"""
        base_size = {
            'small': 32,
            'medium': 64,
            'large': 96
        }[self.config['font_size']]
        
        if self.config['size_jitter'] > 0:
            jitter = random.uniform(-self.config['size_jitter'], self.config['size_jitter'])
            return int(base_size * (1 + jitter))
        return base_size
    
    def render_char(self, char, ink_rgb):
        """Render a single character with style variations and transparency"""
        # Get character mask path
        mask_path = get_mask_for_char(char, self.config['style'])
        if not mask_path:
            return None
        
        # Determine character size
        char_size = self.get_char_size()
        
        # Load and resize mask
        mask_img = Image.open(mask_path).convert('L')
        mask_img = mask_img.resize((char_size, char_size))
        
        # Apply stroke variation
        if self.config['stroke_variation'] and self.config['style'] == "Edge Only":
            # Vary Canny thresholds for edge detection
            t1 = random.randint(20, 40)
            t2 = random.randint(50, 90)
            mask_np = np.array(mask_img)
            _, binary_mask = cv2.threshold(mask_np, 240, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            edges = cv2.Canny(cleaned, threshold1=t1, threshold2=t2)
            edges = cv2.dilate(edges, kernel, iterations=1)
            mask_img = Image.fromarray(edges)
        
        # Colorize mask
        char_img = colorize_mask(mask_img, ink_rgb, self.config['style'])
        
        # Apply tilt
        if self.config['tilt_range']:
            angle = random.uniform(*self.config['tilt_range'])
            char_img = char_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        
        # Convert to RGBA with transparency
        np_img = np.array(char_img)
        # Create alpha channel (initially all opaque)
        alpha = np.ones((np_img.shape[0], np_img.shape[1]), dtype=np.uint8) * 255
        
        # Find white background (where all RGB channels > 240)
        white_mask = (np_img[:, :, 0] > 240) & (np_img[:, :, 1] > 240) & (np_img[:, :, 2] > 240)
        
        # Set alpha to 0 (transparent) for white background
        alpha[white_mask] = 0
        
        # Combine RGB with alpha channel
        rgba = np.dstack((np_img, alpha))
        return Image.fromarray(rgba)
    
    def calculate_line_spacing(self, char_height):
        """Calculate line spacing based on configuration"""
        return char_height * self.config['line_spacing']
    
    def draw_ruled_lines(self, img):
        """Draw ruled lines on the page"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        margins = self.config['margins']
        
        # Get character size for baseline calculation
        char_size = self.get_char_size()
        line_height = self.calculate_line_spacing(char_size)
        
        # Calculate baseline positions
        y = margins['top'] + char_size
        while y < height - margins['bottom']:
            draw.line([(margins['left'], y), (width - margins['right'], y)], 
                     fill=self.config['rule_color'], width=1)
            y += line_height
    
    def render_text(self, text):
        """Render full text into one or more pages"""
        # Preprocess text
        paragraphs = text.split('\n')
        ink_rgb = INK_COLORS[self.config['ink_color']]
        pages = []
        current_page = []
        current_line = []
        
        # Get character size metrics
        base_char_size = self.get_char_size()
        avg_char_width = base_char_size * 0.6  # Estimate average character width
        
        for para in paragraphs:
            # Split paragraph into words
            words = para.split()
            if not words:
                # Add empty line for empty paragraph (use None as placeholder)
                current_page.append(None)
                continue
            
            for word in words:
                # Add character spacing variations
                char_spacing = random.randint(*self.config['char_spacing_range'])
                
                # Estimate word width
                word_width = len(word) * avg_char_width + max(0, len(word)-1) * char_spacing
                
                # Calculate word spacing if not first word in line
                word_spacing = 0
                if current_line:
                    word_spacing = base_char_size * self.config['word_spacing_factor']
                
                # Calculate current line width
                current_line_width = self._line_width(current_line, avg_char_width)
                
                # Check if we need to wrap to next line
                if current_line and current_line_width + word_spacing + word_width > self._available_width():
                    current_page.append(current_line)
                    current_line = []
                    word_spacing = 0  # Reset for new line
                
                # Add word to current line
                current_line.append((word, char_spacing))
            
            # Finish paragraph
            if current_line:
                current_page.append(current_line)
                current_line = []
            
            # Add paragraph spacing (use None as placeholder)
            current_page.append(None)
        
        # Add final line if exists
        if current_line:
            current_page.append(current_line)
        
        # Render pages
        all_pages = []
        while current_page:
            page, remaining = self.render_page(current_page, ink_rgb)
            all_pages.append(page)
            current_page = remaining
        
        return all_pages
    
    def _line_width(self, line, avg_char_width):
        """Calculate approximate width of a line"""
        if not line:
            return 0
    
        width = 0
        base_char_size = self.get_char_size()
        
        for i, (word, char_spacing) in enumerate(line):
            # Word characters + spacing between characters
            word_width = len(word) * avg_char_width + max(0, len(word)-1) * char_spacing
    
            # Add word spacing between words
            if i > 0:
                word_spacing = base_char_size * self.config['word_spacing_factor']
                width += word_spacing
    
            width += word_width

        return width
    
    def _available_width(self):
        """Calculate available width for text"""
        margins = self.config['margins']
        return self.config['page_width'] - margins['left'] - margins['right']
    
    def _available_height(self):
        """Calculate available height for text"""
        margins = self.config['margins']
        return self.config['page_height'] - margins['top'] - margins['bottom']
    
    def render_page(self, lines, ink_rgb):
        """Render a single page of text"""
        # Create blank page
        page_img = Image.new('RGB', 
                            (self.config['page_width'], self.config['page_height']), 
                            (255, 255, 255))
        
        # Draw ruled lines if needed
        if self.config['page_layout'] == 'ruled':
            self.draw_ruled_lines(page_img)
        
        # Get character metrics
        base_char_size = self.get_char_size()
        line_spacing = self.calculate_line_spacing(base_char_size)
        
        # Set initial position
        x = self.config['margins']['left']
        y = self.config['margins']['top']
        rendered_lines = 0
        max_lines = int(self._available_height() / line_spacing)
        
        # Process lines until page is full
        page_lines = []
        remaining_lines = []
        
        for i, line in enumerate(lines):
            # Handle empty lines (paragraph breaks)
            if line is None:
                y += line_spacing * self.config['paragraph_spacing']
                rendered_lines += self.config['paragraph_spacing']
                page_lines.append(None)
                continue
            
            # Check if we can fit another line
            if rendered_lines >= max_lines:
                remaining_lines = lines[i:]
                break
            
            # Render line
            current_x = x
            line_words = []
            
            for word, char_spacing in line:
                for j, char in enumerate(word):
                    # Render character
                    char_img = self.render_char(char, ink_rgb)
                    if not char_img:
                        continue
                    
                    # Calculate position (center vertically)
                    char_width, char_height = char_img.size
                    y_pos = y + (line_spacing - char_height) // 2
                    
                    # Paste character
                    page_img.paste(char_img, (int(current_x), int(y_pos)), char_img)
                    
                    # Update position
                    current_x += char_width
                    
                    # Add character spacing
                    if j < len(word) - 1:
                        current_x += char_spacing
                
                # Add word spacing between words
                if word != line[-1][0]:
                    current_x += base_char_size * self.config['word_spacing_factor']
                
                line_words.append(word)
            
            # Add extra space after punctuation
            if line_words and line_words[-1][-1] in PUNCTUATION:
                current_x += base_char_size * 0.5
            
            # Prepare for next line
            y += line_spacing
            rendered_lines += 1
            page_lines.append(line_words)
        
        return page_img, remaining_lines

def render_handwriting(text, masks_dir, style, ink_color, paper, lines_per_page, job_id):
    # Create the generator with mask directories
    generator = HandwritingGenerator(
        mask_root_full=os.path.join(masks_dir, "FullMask"),
        mask_root_edge=os.path.join(masks_dir, "EdgeOnly")
    )
    
    # Set configuration
    generator.set_config(
        style="Edge Only" if style == "EdgeOnly" else "Full Character",
        ink_color=ink_color,
        font_size='medium',
        stroke_variation=True,
        page_layout='ruled' if paper == "ruled" else "unruled",
        line_spacing=1.5,
        margin_left=50,
        margin_right=50,
        margin_top=50,
        margin_bottom=50,
        tilt_range=(-10, 10),
        char_spacing_range=(-2, 5),
        size_jitter=0.05,
        word_spacing_factor=1.2
    )
    
    # Render the text
    pages = generator.render_text(text)
    
    # Save to output directory
    output_dir = f'workspace/output/{job_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    if not pages:
        return None
    
    # For simplicity, return the first page
    output_path = os.path.join(output_dir, 'output.png')
    pages[0].save(output_path)
    
    return output_path