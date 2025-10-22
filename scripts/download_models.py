#!/usr/bin/env python3
"""
Model Download Script
Downloads and prepares required models for the PDF Translation Pipeline
"""

import os
import sys
import subprocess
import hashlib
import tarfile
import zipfile
from pathlib import Path
import urllib.request
import json
from tqdm import tqdm

class ModelDownloader:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def download_file(self, url, destination, description="Downloading"):
        """Download file with progress bar"""
        try:
            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def extract_archive(self, archive_path, extract_to):
        """Extract tar.gz or zip archives"""
        print(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.gz':
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            print(f"Unknown archive format: {archive_path.suffix}")
            return False
        
        return True
    
    def download_surya_models(self):
        """Download Surya layout detection models"""
        print("\n📦 Downloading Surya Layout Detection Models...")
        
        # Surya uses HuggingFace models, they will be auto-downloaded on first use
        # We'll create a script to pre-download them
        
        surya_script = """
import os
os.environ['HF_HOME'] = 'models/huggingface'

try:
    from surya.model.detection.segformer import load_model as load_detection_model
    from surya.model.recognition.model import load_model as load_recognition_model
    
    print("Downloading detection model...")
    det_model = load_detection_model()
    
    print("Downloading recognition model...")
    rec_model = load_recognition_model()
    
    print("✓ Surya models downloaded successfully")
except ImportError:
    print("⚠ Surya not installed. Install with: pip install surya-ocr")
except Exception as e:
    print(f"⚠ Error downloading Surya models: {e}")
"""
        
        try:
            result = subprocess.run([sys.executable, "-c", surya_script], 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
        except Exception as e:
            print(f"⚠ Could not download Surya models: {e}")
    
    def download_paddleocr_models(self):
        """Download PaddleOCR models"""
        print("\n📦 Downloading PaddleOCR Models...")
        
        paddle_dir = self.models_dir / "paddleocr"
        paddle_dir.mkdir(exist_ok=True)
        
        models = {
            "det": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar",
                "name": "ch_PP-OCRv4_det_infer"
            },
            "rec": {
                "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
                "name": "ch_PP-OCRv4_rec_infer"
            },
            "cls": {
                "url": "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar",
                "name": "ch_ppocr_mobile_v2.0_cls_infer"
            }
        }
        
        for model_type, info in models.items():
            tar_path = paddle_dir / f"{info['name']}.tar"
            
            if not (paddle_dir / info['name']).exists():
                print(f"Downloading {model_type} model...")
                if self.download_file(info['url'], tar_path, f"Downloading {model_type}"):
                    # Extract
                    with tarfile.open(tar_path, 'r') as tar:
                        tar.extractall(paddle_dir)
                    tar_path.unlink()  # Remove tar file
                    print(f"✓ {model_type} model ready")
            else:
                print(f"✓ {model_type} model already exists")
    
    def download_latex_ocr_models(self):
        """Download LaTeX-OCR models"""
        print("\n📦 Setting up LaTeX-OCR Models...")
        
        latex_script = """
try:
    from pix2tex import cli
    print("✓ LaTeX-OCR (pix2tex) is installed")
    print("  Models will be downloaded on first use")
except ImportError:
    print("⚠ LaTeX-OCR not installed. Install with: pip install pix2tex[gui]")
"""
        
        subprocess.run([sys.executable, "-c", latex_script])
    
    def download_table_transformer(self):
        """Download Table Transformer models"""
        print("\n📦 Setting up Table Transformer Models...")
        
        # Table transformer models from HuggingFace
        table_script = """
import os
os.environ['HF_HOME'] = 'models/huggingface'

try:
    from transformers import AutoModelForObjectDetection
    
    print("Downloading Table Structure Recognition model...")
    model = AutoModelForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition"
    )
    print("✓ Table Transformer model downloaded")
except ImportError:
    print("⚠ Transformers not installed. Install with: pip install transformers")
except Exception as e:
    print(f"⚠ Error downloading Table Transformer: {e}")
"""
        
        subprocess.run([sys.executable, "-c", table_script])
    
    def setup_layoutparser_models(self):
        """Setup LayoutParser models"""
        print("\n📦 Setting up LayoutParser Models...")
        
        layout_dir = self.models_dir / "layoutparser"
        layout_dir.mkdir(exist_ok=True)
        
        # Create config for LayoutParser
        config = {
            "PubLayNet": {
                "config_url": "https://github.com/Layout-Parser/layout-model-training/raw/master/configs/prima/fast_rcnn_R_50_FPN_3x.yaml",
                "model_url": "https://www.dropbox.com/s/57zjbwv6gh3srry/model_final.pth?dl=1"
            }
        }
        
        config_path = layout_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✓ LayoutParser configuration created")
        print("  Models will be downloaded on first use")
    
    def create_model_info(self):
        """Create model information file"""
        info = {
            "models": {
                "surya": {
                    "description": "Layout and text detection",
                    "size": "~500MB",
                    "location": "models/huggingface"
                },
                "paddleocr": {
                    "description": "OCR for multiple languages",
                    "size": "~300MB",
                    "location": "models/paddleocr"
                },
                "latex_ocr": {
                    "description": "Mathematical formula recognition",
                    "size": "~200MB",
                    "location": "auto-download"
                },
                "table_transformer": {
                    "description": "Table structure recognition",
                    "size": "~150MB",
                    "location": "models/huggingface"
                },
                "layoutparser": {
                    "description": "Document layout analysis",
                    "size": "~250MB",
                    "location": "models/layoutparser"
                }
            }
        }
        
        info_path = self.models_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✓ Model information saved to {info_path}")

def main():
    print("="*60)
    print("PDF Translation Pipeline - Model Download")
    print("="*60)
    
    downloader = ModelDownloader()
    
    # Check for required packages
    required_packages = []
    
    try:
        import paddleocr
    except ImportError:
        required_packages.append("paddleocr")
    
    try:
        import transformers
    except ImportError:
        required_packages.append("transformers")
    
    if required_packages:
        print(f"\n⚠ Missing packages: {', '.join(required_packages)}")
        print("Install with: pip install " + " ".join(required_packages))
        response = input("\nInstall missing packages now? (y/N): ")
        if response.lower() == 'y':
            subprocess.run([sys.executable, "-m", "pip", "install"] + required_packages)
    
    # Download models
    print("\nStarting model downloads...")
    print("Note: Some models will be downloaded on first use\n")
    
    # Download each model type
    downloader.download_paddleocr_models()
    downloader.download_surya_models()
    downloader.download_latex_ocr_models()
    downloader.download_table_transformer()
    downloader.setup_layoutparser_models()
    
    # Create model info file
    downloader.create_model_info()
    
    print("\n" + "="*60)
    print("✅ Model setup complete!")
    print("="*60)
    print("\nModels are stored in: ./models/")
    print("Some models will be downloaded automatically on first use")

if __name__ == "__main__":
    main()