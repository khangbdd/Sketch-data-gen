#!/usr/bin/env python3
"""
Test script for the Image Captioning Pipeline

This script tests the basic functionality without making actual API calls.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import click
        print("  ✓ click")
    except ImportError:
        print("  ✗ click - run: pip install click")
        return False
    
    try:
        import openai
        print("  ✓ openai")
    except ImportError:
        print("  ✗ openai - run: pip install openai")
        return False
    
    try:
        import anthropic
        print("  ✓ anthropic")
    except ImportError:
        print("  ✗ anthropic - run: pip install anthropic")
        return False
    
    try:
        import google.generativeai
        print("  ✓ google-generativeai")
    except ImportError:
        print("  ✗ google-generativeai - run: pip install google-generativeai")
        return False
    
    try:
        from PIL import Image
        print("  ✓ PIL (Pillow)")
    except ImportError:
        print("  ✗ PIL - run: pip install pillow")
        return False
    
    try:
        from tqdm import tqdm
        print("  ✓ tqdm")
    except ImportError:
        print("  ✗ tqdm - run: pip install tqdm")
        return False
    
    try:
        from dotenv import load_dotenv
        print("  ✓ python-dotenv")
    except ImportError:
        print("  ✗ python-dotenv - run: pip install python-dotenv")
        return False
    
    return True


def test_env_file():
    """Test if .env file exists and has API keys"""
    print("\nTesting environment configuration...")
    
    if not os.path.exists('.env'):
        print("  ✗ .env file not found")
        print("    Copy .env.example to .env and add your API keys")
        return False
    
    load_dotenv()
    
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
    }
    
    valid_keys = 0
    for key, value in api_keys.items():
        if value and value != f'your_{key.lower()}_here':
            print(f"  ✓ {key} configured")
            valid_keys += 1
        else:
            print(f"  ⚠ {key} not configured")
    
    if valid_keys == 0:
        print("  ✗ No API keys configured")
        print("    Please add at least one API key to your .env file")
        return False
    
    print(f"  ✓ {valid_keys}/3 API keys configured")
    return True


def test_pipeline_import():
    """Test if the pipeline modules can be imported"""
    print("\nTesting pipeline imports...")
    
    try:
        from captioners import OpenAICaptioner, AnthropicCaptioner, GoogleCaptioner, CaptionMerger
        print("  ✓ captioners module")
    except ImportError as e:
        print(f"  ✗ captioners module - {e}")
        return False
    
    try:
        import pipeline
        print("  ✓ pipeline module")
    except ImportError as e:
        print(f"  ✗ pipeline module - {e}")
        return False
    
    return True


def create_test_dataset():
    """Create a simple test dataset structure"""
    print("\nCreating test dataset structure...")
    
    test_dir = Path("test_dataset")
    images_dir = test_dir / "images"
    captions_dir = test_dir / "captions"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple test image info file
    test_info = test_dir / "test_info.txt"
    with open(test_info, 'w') as f:
        f.write("Test dataset structure created.\n")
        f.write("Add your test images to the 'images' folder.\n")
        f.write("Captions will be generated in the 'captions' folder.\n")
    
    print(f"  ✓ Test dataset structure created at: {test_dir.absolute()}")
    print(f"    - Images folder: {images_dir}")
    print(f"    - Captions folder: {captions_dir}")
    
    return True


def main():
    """Run all tests"""
    print("Image Captioning Pipeline - Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Package imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Environment configuration
    if test_env_file():
        tests_passed += 1
    
    # Test 3: Pipeline imports
    if test_pipeline_import():
        tests_passed += 1
    
    # Test 4: Create test dataset
    if create_test_dataset():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Your pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Add test images to test_dataset/images/")
        print("2. Run: python pipeline.py -i test_dataset -o test_output")
    else:
        print("✗ Some tests failed. Please address the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
