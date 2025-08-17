#!/usr/bin/env python3
"""
Quick launcher for the Image Captioning Pipeline

This script provides a simple interactive interface for running the pipeline.
"""

import os
import sys
from pathlib import Path


def check_setup():
    """Check if the pipeline is properly set up"""
    issues = []
    
    # Check if virtual environment exists
    if not os.path.exists('venv'):
        issues.append("Virtual environment not found. Run setup.sh or setup.bat first.")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        issues.append(".env file not found. Copy .env.example to .env and add your API keys.")
    
    # Check if requirements are installed (basic check)
    try:
        import click
        import openai
    except ImportError:
        issues.append("Dependencies not installed. Run: pip install -r requirements.txt")
    
    return issues


def interactive_mode():
    """Interactive mode for easier usage"""
    print("Image Captioning Pipeline - Interactive Mode")
    print("=" * 50)
    
    # Get input path
    while True:
        input_path = input("\nEnter path to image or folder: ").strip()
        if not input_path:
            print("Please enter a valid path.")
            continue
        
        input_path = Path(input_path).expanduser().resolve()
        if not input_path.exists():
            print(f"Path does not exist: {input_path}")
            continue
        
        break
    
    # Get output path
    output_path = input("Enter output directory (press Enter for current directory): ").strip()
    if not output_path:
        output_path = "output"
    else:
        output_path = Path(output_path).expanduser().resolve()
    
    # Get additional context
    user_caption = input("Enter additional context (optional): ").strip()
    
    # Get caption source
    print("\nCaption source options:")
    print("1. folder - Use folder name as context")
    print("2. filename - Use filename as context") 
    print("3. manual - Use only the context you provided above")
    
    while True:
        choice = input("Choose caption source (1-3, default: 1): ").strip()
        if not choice:
            caption_source = "folder"
            break
        elif choice == "1":
            caption_source = "folder"
            break
        elif choice == "2":
            caption_source = "filename"
            break
        elif choice == "3":
            caption_source = "manual"
            break
        else:
            print("Please enter 1, 2, or 3.")
    
    # Build command
    cmd = f"python pipeline.py -i \"{input_path}\" -o \"{output_path}\" --caption-source {caption_source}"
    if user_caption:
        cmd += f" -c \"{user_caption}\""
    
    print(f"\nRunning command: {cmd}")
    print("-" * 50)
    
    # Execute command
    os.system(cmd)


def show_help():
    """Show help information"""
    print("Image Captioning Pipeline - Launcher")
    print("=" * 40)
    print("Usage:")
    print("  python launcher.py                    # Interactive mode")
    print("  python launcher.py --help            # Show this help")
    print("  python launcher.py --check           # Check setup")
    print("")
    print("Direct usage:")
    print("  python pipeline.py --help            # Show pipeline options")
    print("")
    print("Examples:")
    print("  python pipeline.py -i image.jpg -o output")
    print("  python pipeline.py -i /path/to/folder -o output")
    print("  python pipeline.py -i dataset/ -o output --caption-source folder")


def main():
    """Main launcher function"""
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            show_help()
            return
        elif sys.argv[1] == "--check":
            issues = check_setup()
            if issues:
                print("Setup Issues Found:")
                for issue in issues:
                    print(f"  - {issue}")
                sys.exit(1)
            else:
                print("✓ Setup looks good!")
                return
    
    # Check setup
    issues = check_setup()
    if issues:
        print("Setup Issues Found:")
        for issue in issues:
            print(f"  ✗ {issue}")
        print("\nPlease resolve these issues before running the pipeline.")
        return
    
    # Run interactive mode
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == '__main__':
    main()
