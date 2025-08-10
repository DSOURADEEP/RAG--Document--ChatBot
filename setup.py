"""
Setup script for RAG Chatbot
Helps with installation and environment setup
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.8 or higher")
        return False

def create_virtual_environment():
    """Create a virtual environment"""
    venv_path = Path("venv")
    if venv_path.exists():
        print("📁 Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def install_requirements():
    """Install Python requirements"""
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        pip_command = "pip install -r requirements.txt"
    else:
        # Use venv pip
        if os.name == 'nt':  # Windows
            pip_command = "venv\\Scripts\\pip install -r requirements.txt"
        else:  # Unix/Linux/macOS
            pip_command = "venv/bin/pip install -r requirements.txt"
    
    return run_command(pip_command, "Installing Python packages")

def check_external_dependencies():
    """Check if external dependencies are installed"""
    print(f"\n{'='*50}")
    print("🔍 Checking External Dependencies")
    print(f"{'='*50}")
    
    # Check Poppler
    try:
        result = subprocess.run("pdftoppm -h", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Poppler is installed and accessible")
        else:
            print("⚠️ Poppler not found - PDF processing may not work")
            print("Please install Poppler for Windows from:")
            print("https://github.com/oschwartz10612/poppler-windows/releases/")
    except Exception:
        print("⚠️ Poppler not found - PDF processing may not work")
        print("Please install Poppler for Windows from:")
        print("https://github.com/oschwartz10612/poppler-windows/releases/")
    
    # Check Tesseract (optional)
    try:
        result = subprocess.run("tesseract --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed and accessible")
        else:
            print("⚠️ Tesseract OCR not found - Image processing will be limited")
            print("You can install it from:")
            print("https://github.com/UB-Mannheim/tesseract/wiki")
    except Exception:
        print("⚠️ Tesseract OCR not found - Image processing will be limited")
        print("You can install it from:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")

def main():
    """Main setup function"""
    print("🤖 RAG Chatbot Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print("Failed to create virtual environment")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements")
        sys.exit(1)
    
    # Check external dependencies
    check_external_dependencies()
    
    print(f"\n{'='*50}")
    print("🎉 Setup Complete!")
    print(f"{'='*50}")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source venv/bin/activate")
    print("2. Run the application:")
    print("   streamlit run app.py")
    print("3. Open your browser to http://localhost:8501")
    print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main()
