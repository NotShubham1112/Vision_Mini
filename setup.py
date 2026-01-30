#!/usr/bin/env python3
"""
Installation and Setup Script
Automated setup for the Multimodal Vision System
"""

import os
import sys
import subprocess
import platform


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required")
        sys.exit(1)
    
    print("✓ Python version is compatible")


def check_gpu():
    """Check for NVIDIA GPU availability"""
    print_header("Checking GPU Availability")
    
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            print(result.stdout.split('\n')[0:10])  # Print first 10 lines
            return True
        else:
            print("⚠ No NVIDIA GPU detected - will use CPU")
            return False
    
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("⚠ nvidia-smi not found - will use CPU")
        return False


def install_dependencies(use_gpu=True):
    """Install Python dependencies"""
    print_header("Installing Python Dependencies")
    
    print("This may take 5-10 minutes...")
    
    # Upgrade pip
    print("\n1. Upgrading pip...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # Install PyTorch (with or without CUDA)
    print("\n2. Installing PyTorch...")
    if use_gpu:
        torch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cu118'
        ]
    else:
        torch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch', 'torchvision', 'torchaudio'
        ]
    
    subprocess.run(torch_cmd)
    
    # Install other requirements
    print("\n3. Installing other dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    
    # Install spaCy model
    print("\n4. Downloading spaCy language model...")
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    
    print("\n✓ All dependencies installed successfully")


def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    dirs = ['models', 'data']
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✓ Created directory: {dir_name}/")


def download_models():
    """Download required models"""
    print_header("Downloading Models")
    
    print("Models will be downloaded automatically on first run.")
    print("This includes:")
    print("  - YOLOv8 nano model (~6MB)")
    print("  - InsightFace buffalo_l model (~400MB)")
    print("  - Faster-Whisper base model (~150MB)")
    print("\nTotal download size: ~550MB")


def test_imports():
    """Test if all required packages can be imported"""
    print_header("Testing Imports")
    
    packages = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('spacy', 'spacy'),
    ]
    
    failed = []
    
    for module, package in packages:
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    return True


def check_camera():
    """Check if camera is available"""
    print_header("Checking Camera")
    
    try:
        import cv2
        
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            
            if ret:
                print("✓ Camera is working")
                return True
            else:
                print("⚠ Camera opened but couldn't read frame")
                return False
        else:
            print("⚠ Couldn't open camera")
            print("Make sure your webcam is connected and not in use")
            return False
    
    except Exception as e:
        print(f"❌ Error checking camera: {e}")
        return False


def check_microphone():
    """Check if microphone is available"""
    print_header("Checking Microphone")
    
    try:
        import sounddevice as sd
        
        devices = sd.query_devices()
        
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if input_devices:
            print("✓ Found microphone(s):")
            for i, device in enumerate(input_devices):
                print(f"  {i}: {device['name']}")
            return True
        else:
            print("⚠ No microphone detected")
            return False
    
    except Exception as e:
        print(f"⚠ Could not check microphone: {e}")
        print("You may need to install portaudio or pyaudio")
        return False


def print_next_steps():
    """Print instructions for running the system"""
    print_header("Setup Complete!")
    
    print("To run the system:\n")
    print("  python main.py\n")
    print("Voice commands:")
    print("  - 'My name is [name]'        → Identify yourself")
    print("  - 'This person is [name]'    → Identify someone else")
    print("  - 'That is a [object]'       → Label an object")
    print("\nPress Q or ESC to exit the video window")
    print("\nFor more information, see README.md")


def main():
    """Main setup flow"""
    print("\n" + "🤖 " * 20)
    print(" " * 15 + "MULTIMODAL VISION SYSTEM")
    print(" " * 20 + "Setup Script")
    print("🤖 " * 20)
    
    # Check Python version
    check_python_version()
    
    # Check for GPU
    has_gpu = check_gpu()
    
    # Ask user if they want to proceed
    print("\n" + "-" * 60)
    response = input("Continue with installation? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Setup cancelled.")
        sys.exit(0)
    
    try:
        # Install dependencies
        install_dependencies(use_gpu=has_gpu)
        
        # Create directories
        create_directories()
        
        # Test imports
        if not test_imports():
            print("\n❌ Setup incomplete. Please fix import errors.")
            sys.exit(1)
        
        # Check hardware
        check_camera()
        check_microphone()
        
        # Show model download info
        download_models()
        
        # Print next steps
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()