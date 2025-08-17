#!/usr/bin/env python3
"""
Example usage of the Image Captioning Pipeline

This script demonstrates how to use the pipeline programmatically.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from captioners import OpenAICaptioner, AnthropicCaptioner, GoogleCaptioner, CaptionMerger


def example_single_image():
    """Example of processing a single image programmatically"""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize captioners (add your API keys to .env)
    captioners = []
    
    if os.getenv('OPENAI_API_KEY'):
        captioners.append(OpenAICaptioner(os.getenv('OPENAI_API_KEY')))
    
    if os.getenv('ANTHROPIC_API_KEY'):
        captioners.append(AnthropicCaptioner(os.getenv('ANTHROPIC_API_KEY')))
    
    if os.getenv('GOOGLE_API_KEY'):
        captioners.append(GoogleCaptioner(os.getenv('GOOGLE_API_KEY')))
    
    # Initialize merger
    merger = CaptionMerger(os.getenv('OPENAI_API_KEY'))
    
    # Path to your test image
    image_path = "path/to/your/test/image.jpg"
    
    if not os.path.exists(image_path):
        print("Please update the image_path variable with a valid image file.")
        return
    
    print(f"Processing image: {image_path}")
    
    # Generate captions
    captions = []
    for i, captioner in enumerate(captioners):
        try:
            caption = captioner.caption_image(image_path, "Test image")
            captions.append(caption)
            print(f"Caption {i+1}: {caption}")
        except Exception as e:
            print(f"Error generating caption {i+1}: {e}")
    
    # Merge captions
    if captions:
        try:
            merged_caption = merger.merge_captions(captions, "Test image", "example")
            print(f"\nMerged Caption: {merged_caption}")
        except Exception as e:
            print(f"Error merging captions: {e}")
    else:
        print("No captions were generated successfully.")


def example_batch_processing():
    """Example of batch processing images"""
    from pipeline import ImageCaptioningPipeline, load_config
    
    # Load configuration
    config = load_config()
    
    # Initialize pipeline
    pipeline = ImageCaptioningPipeline(config)
    
    # Process a folder of images
    folder_path = "test_dataset"
    output_path = "test_output"
    
    if not os.path.exists(folder_path):
        print("Please create a test_dataset folder with images or update the folder_path variable.")
        return
    
    print(f"Processing folder: {folder_path}")
    
    try:
        results = pipeline.process_image_folder(folder_path, output_path, "folder")
        
        successful = len([r for r in results if r.get('success', False)])
        total = len(results)
        
        print(f"Processing completed: {successful}/{total} images successful")
        
    except Exception as e:
        print(f"Error processing folder: {e}")


if __name__ == '__main__':
    print("Image Captioning Pipeline - Examples")
    print("=" * 40)
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("Please create a .env file with your API keys before running examples.")
        print("Copy .env.example to .env and add your API keys.")
        exit(1)
    
    print("\n1. Single Image Example:")
    try:
        example_single_image()
    except Exception as e:
        print(f"Single image example failed: {e}")
    
    print("\n" + "-" * 40)
    print("\n2. Batch Processing Example:")
    try:
        example_batch_processing()
    except Exception as e:
        print(f"Batch processing example failed: {e}")
    
    print("\nExamples completed!")
