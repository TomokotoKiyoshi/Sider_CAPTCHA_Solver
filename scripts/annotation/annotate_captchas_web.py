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
import sys

app = Flask(__name__)

# Configuration - will be set after user selection
INPUT_DIR = None
OUTPUT_DIR = None
MAX_IMAGES = 200

# Global state
current_index = 0
images_list = []
annotations = []

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
                // 缺口的Y坐标必须与滑块相同
                gapPos = {x: x, y: sliderPos.y};  // 使用滑块的Y坐标
                addMarker(x, sliderPos.y, 'gap');  // 在滑块的Y位置显示标记
                
                // 画一条水平线显示Y坐标对齐
                ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
                ctx.setLineDash([5, 5]);
                ctx.beginPath();
                ctx.moveTo(0, sliderPos.y);
                ctx.lineTo(canvas.width, sliderPos.y);
                ctx.stroke();
                ctx.setLineDash([]);
                
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
                document.getElementById('status').textContent = '点击标记滑块中心位置 (红色)';
                document.getElementById('saveBtn').disabled = true;
            } else if (!gapPos) {
                document.getElementById('status').textContent = `点击标记缺口X位置 (Y自动对齐到${sliderPos.y})`;
                document.getElementById('saveBtn').disabled = true;
            } else {
                document.getElementById('status').textContent = 
                    `滑块: (${sliderPos.x}, ${sliderPos.y}), 缺口: (${gapPos.x}, ${gapPos.y})`;
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

def select_folder():
    """Let user select input and output folders"""
    global INPUT_DIR, OUTPUT_DIR
    
    print("\n" + "="*60)
    print("CAPTCHA Annotation Tool - Folder Selection")
    print("="*60)
    
    # Select input folder
    print("\n选择输入文件夹 (merged目录下):")
    merged_dir = Path("data/real_captchas/merged")
    
    if not merged_dir.exists():
        print(f"错误: {merged_dir} 不存在")
        return False
    
    # List available folders
    folders = [d for d in merged_dir.iterdir() if d.is_dir()]
    if not folders:
        print("错误: merged目录下没有文件夹")
        return False
    
    print("可用的文件夹:")
    for i, folder in enumerate(sorted(folders), 1):
        img_count = len(list(folder.glob("*.png")))
        print(f"  {i}. {folder.name} ({img_count} 张图片)")
    
    # Get user choice
    while True:
        try:
            choice = input(f"\n请选择输入文件夹 (1-{len(folders)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(folders):
                INPUT_DIR = sorted(folders)[idx]
                break
            else:
                print("无效的选择")
        except ValueError:
            print("请输入数字")
    
    # Set output folder
    folder_name = INPUT_DIR.name
    OUTPUT_DIR = Path("data/real_captchas/annotated") / folder_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for existing annotations
    existing_json = OUTPUT_DIR / "annotations.json"
    if existing_json.exists():
        print(f"\n⚠️  发现已存在的标注文件: {existing_json}")
        overwrite = input("是否覆盖? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("选择另一个文件夹或退出")
            return False
    
    print(f"\n✅ 输入文件夹: {INPUT_DIR}")
    print(f"✅ 输出文件夹: {OUTPUT_DIR}")
    
    # Load existing annotations if any
    global annotations
    if existing_json.exists():
        try:
            with open(existing_json, 'r') as f:
                annotations = json.load(f)
                print(f"   已加载 {len(annotations)} 个现有标注")
        except:
            annotations = []
    
    return True

if __name__ == '__main__':
    # Select folders first
    if not select_folder():
        print("文件夹选择失败，退出程序")
        sys.exit(1)
    
    print(f"\nLoading images from: {INPUT_DIR}")
    num_images = load_images()
    print(f"Found {num_images} images to annotate")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Ask for max images
    try:
        max_input = input(f"\n要标注多少张图片? (默认: {MAX_IMAGES}, 最大: {num_images}): ").strip()
        if max_input:
            MAX_IMAGES = min(int(max_input), num_images)
            images_list = images_list[:MAX_IMAGES]
            print(f"将标注 {len(images_list)} 张图片")
    except ValueError:
        print(f"使用默认值: {MAX_IMAGES}")
    
    print("\nStarting web server...")
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=False, port=5000)