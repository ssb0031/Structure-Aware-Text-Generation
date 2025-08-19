import os
import uuid
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename
from extraction import extract_characters
from detection import classify_characters
from augmentation import augment_characters
from masking import create_masks
from gans import apply_gan_style
from render import render_handwriting

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'workspace/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Home page with upload form
@app.route('/')
def index():
    return render_template('index.html')

# Upload image endpoint
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
    file.save(filepath)
    
    # Process pipeline steps
    try:
        # Step 1: Character extraction
        extracted_dir = extract_characters(filepath, job_id)
        
        # Step 2: Character classification
        classified_dir = classify_characters(extracted_dir, job_id)
        
        # Step 3: Augmentation
        augmented_dir = augment_characters(classified_dir, job_id)
        
        # Step 4: Mask creation
        masks_dir = create_masks(augmented_dir, job_id)
        
        return jsonify({
            'job_id': job_id,
            'status': 'success',
            'message': 'Image processed successfully',
            'masks_dir': masks_dir
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Generate handwriting endpoint
@app.route('/generate_text', methods=['POST'])
def generate_text():
    data = request.json
    job_id = data.get('job_id')
    text = data.get('text')
    style = data.get('style', 'FullMask')
    ink_color = data.get('ink_color', 'black')
    paper = data.get('paper', 'plain')
    lines_per_page = data.get('lines_per_page', 10)
    
    if not job_id or not text:
        return jsonify({'error': 'Missing parameters'}), 400
    
    try:
        # Step 5: GAN Stylization
        masks_dir = os.path.join('workspace/masks', job_id)
        styled_dir = apply_gan_style(masks_dir, style, job_id)
        
        # Step 6: Rendering
        output_path = render_handwriting(
            text=text,
            masks_dir=styled_dir,
            style=style,
            ink_color=ink_color,
            paper=paper,
            lines_per_page=lines_per_page,
            job_id=job_id
        )
        
        if not output_path:
            return jsonify({'error': 'Failed to generate handwriting'}), 500
        
        return send_file(output_path, as_attachment=True, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)