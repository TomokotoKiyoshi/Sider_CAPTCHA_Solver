#!/usr/bin/env python3
"""
Web-based CAPTCHA Annotation Tool
Uses Flask to create a web interface for annotation
"""
import os
import json
import shutil
from pathlib import Path
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import base64

app = Flask(__name__)

# Configuration
INPUT_DIR = Path("../../data/real_captchas/merged/site1")
OUTPUT_DIR = Path("../../data/real_captchas/annotated")
MAX_IMAGES = 100

# Global state
current_index = 0
images_list = []
annotations = []
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_images():
    """Load list of images to annotate"""
    global images_list
    images_list = list(INPUT_DIR.glob("*.png"))[:MAX_IMAGES]
    return len(images_list)

def get_image_base64(image_path):
    """Convert image to base64 for web display"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

@app.route('/')
def index():
    """Main annotation page"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>CAPTCHA Annotator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f0f0f0;
        }
        #container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        #canvas-container {
            position: relative;
            display: inline-block;
            border: 2px solid #333;
            cursor: crosshair;
        }
        canvas {
            display: block;
        }
        .controls {
            margin: 20px 0;
        }
        button {
            font-size: 16px;
            padding: 10px 20px;
            margin: 0 5px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            background-color: #4CAF50;
            color: white;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .info {
            margin: 10px 0;
            font-size: 18px;
        }
        .instructions {
            background-color: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        #progress {
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
        .marker {
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }
        .slider-marker {
            background-color: red;
            border: 2px solid darkred;
        }
        .gap-marker {
            background-color: blue;
            border: 2px solid darkblue;
        }
        .marker-label {
            position: absolute;
            color: white;
            font-weight: bold;
            font-size: 12px;
            transform: translate(-50%, -50%);
        }
    </style>
</head>
<body>
    <div id="container">
        <h1>CAPTCHA Annotation Tool</h1>
        
        <div class="instructions">
            <strong>Instructions:</strong><br>
            1. Click on the <span style="color: red;">SLIDER</span> center (Red marker)<br>
            2. Click on the <span style="color: blue;">GAP</span> center (Blue marker)<br>
            3. Click "Save & Next" or press SPACE when done<br>
            4. Press R to reset, S to skip
        </div>
        
        <div id="progress">Loading...</div>
        <div class="info" id="filename"></div>
        
        <div id="canvas-container">
            <canvas id="canvas"></canvas>
        </div>
        
        <div class="controls">
            <button onclick="resetAnnotation()">Reset (R)</button>
            <button onclick="skipImage()">Skip (S)</button>
            <button id="saveBtn" onclick="saveAndNext()" disabled>Save & Next (Space)</button>
        </div>
        
        <div class="info" id="status"></div>
    </div>
    
    <script>
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let container = document.getElementById('canvas-container');
        
        let currentImage = null;
        let sliderPos = null;
        let gapPos = null;
        let annotatedCount = 0;
        
        // Load current image
        function loadCurrentImage() {
            fetch('/get_current_image')
                .then(response => response.json())
                .then(data => {
                    if (data.finished) {
                        document.getElementById('progress').textContent = 'All images annotated!';
                        document.getElementById('status').textContent = 'Annotation complete. You can close this window.';
                        document.querySelector('.controls').style.display = 'none';
                        return;
                    }
                    
                    currentImage = new Image();
                    currentImage.onload = function() {
                        canvas.width = this.width;
                        canvas.height = this.height;
                        ctx.drawImage(this, 0, 0);
                        
                        // Reset markers
                        document.querySelectorAll('.marker').forEach(m => m.remove());
                        sliderPos = null;
                        gapPos = null;
                        updateStatus();
                    };
                    currentImage.src = 'data:image/png;base64,' + data.image;
                    
                    document.getElementById('filename').textContent = 'File: ' + data.filename;
                    document.getElementById('progress').textContent = 
                        `Progress: ${data.current_index + 1}/${data.total_images} (Annotated: ${annotatedCount})`;
                });
        }
        
        // Handle canvas click
        canvas.addEventListener('click', function(e) {
            let rect = canvas.getBoundingClientRect();
            let x = Math.round(e.clientX - rect.left);
            let y = Math.round(e.clientY - rect.top);
            
            if (!sliderPos) {
                sliderPos = {x: x, y: y};
                addMarker(x, y, 'slider');
                updateStatus();
            } else if (!gapPos) {
                gapPos = {x: x, y: y};
                addMarker(x, y, 'gap');
                updateStatus();
            }
        });
        
        // Add visual marker
        function addMarker(x, y, type) {
            let marker = document.createElement('div');
            marker.className = 'marker ' + (type === 'slider' ? 'slider-marker' : 'gap-marker');
            marker.style.left = x + 'px';
            marker.style.top = y + 'px';
            
            let label = document.createElement('div');
            label.className = 'marker-label';
            label.textContent = type === 'slider' ? 'S' : 'G';
            label.style.left = x + 'px';
            label.style.top = y + 'px';
            
            container.appendChild(marker);
            container.appendChild(label);
        }
        
        // Update status
        function updateStatus() {
            if (!sliderPos) {
                document.getElementById('status').textContent = 'Click to mark SLIDER position (Red)';
                document.getElementById('saveBtn').disabled = true;
            } else if (!gapPos) {
                document.getElementById('status').textContent = 'Click to mark GAP position (Blue)';
                document.getElementById('saveBtn').disabled = true;
            } else {
                document.getElementById('status').textContent = 
                    `Slider: (${sliderPos.x}, ${sliderPos.y}), Gap: (${gapPos.x}, ${gapPos.y})`;
                document.getElementById('saveBtn').disabled = false;
            }
        }
        
        // Reset annotation
        function resetAnnotation() {
            ctx.drawImage(currentImage, 0, 0);
            document.querySelectorAll('.marker').forEach(m => m.remove());
            document.querySelectorAll('.marker-label').forEach(m => m.remove());
            sliderPos = null;
            gapPos = null;
            updateStatus();
        }
        
        // Skip image
        function skipImage() {
            fetch('/skip_image', {method: 'POST'})
                .then(() => loadCurrentImage());
        }
        
        // Save and next
        function saveAndNext() {
            if (!sliderPos || !gapPos) return;
            
            fetch('/save_annotation', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    slider_x: sliderPos.x,
                    slider_y: sliderPos.y,
                    gap_x: gapPos.x,
                    gap_y: gapPos.y
                })
            })
            .then(response => response.json())
            .then(data => {
                annotatedCount = data.annotated_count;
                loadCurrentImage();
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.key === 'r' || e.key === 'R') resetAnnotation();
            else if (e.key === 's' || e.key === 'S') skipImage();
            else if (e.key === ' ' && !document.getElementById('saveBtn').disabled) {
                e.preventDefault();
                saveAndNext();
            }
        });
        
        // Load first image
        loadCurrentImage();
    </script>
</body>
</html>
    '''

@app.route('/get_current_image')
def get_current_image():
    """Get current image data"""
    global current_index
    
    if current_index >= len(images_list):
        return jsonify({'finished': True})
    
    image_path = images_list[current_index]
    image_base64 = get_image_base64(image_path)
    
    return jsonify({
        'image': image_base64,
        'filename': image_path.name,
        'current_index': current_index,
        'total_images': len(images_list),
        'finished': False
    })

@app.route('/skip_image', methods=['POST'])
def skip_image():
    """Skip current image"""
    global current_index
    current_index += 1
    return jsonify({'success': True})

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    """Save annotation for current image"""
    global current_index, annotations
    
    data = request.json
    image_path = images_list[current_index]
    
    # Generate new filename following README.md format
    # Format: Pic{XXXX}_Bgx{gap_x}Bgy{gap_y}_Sdx{slider_x}Sdy{slider_y}_{hash}.png
    annotated_count = len(annotations) + 1
    hash_str = hashlib.md5(str(image_path).encode()).hexdigest()[:8]
    filename = f"Pic{annotated_count:04d}_Bgx{data['gap_x']}Bgy{data['gap_y']}_Sdx{data['slider_x']}Sdy{data['slider_y']}_{hash_str}.png"
    
    # Copy image with new name
    output_path = OUTPUT_DIR / filename
    shutil.copy2(image_path, output_path)
    
    # Save annotation - 使用与test数据集一致的格式
    annotation = {
        'filename': filename,
        'bg_center': [data['gap_x'], data['gap_y']],      # gap对应bg_center
        'sd_center': [data['slider_x'], data['slider_y']]  # slider对应sd_center
    }
    annotations.append(annotation)
    
    # Save JSON file
    with open(OUTPUT_DIR / 'annotations.json', 'w') as f:
        json.dump(annotations, f, indent=2)
    
    # Move to next image
    current_index += 1
    
    return jsonify({
        'success': True,
        'annotated_count': len(annotations),
        'saved_as': filename
    })

@app.route('/download_annotations')
def download_annotations():
    """Download annotations JSON file"""
    return send_file(OUTPUT_DIR / 'annotations.json', as_attachment=True)

if __name__ == '__main__':
    print(f"Loading images from: {INPUT_DIR}")
    num_images = load_images()
    print(f"Found {num_images} images to annotate")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nStarting web server...")
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=False, port=5000)