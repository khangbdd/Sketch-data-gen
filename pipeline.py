#!/usr/bin/env python3
"""
Image Captioning Pipeline

A comprehensive pipeline that uses multiple LLM models to generate and merge image captions.
Supports single images or batch processing of image folders.
"""

import os
import sys
import json
import click
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import time

from captioners import OpenAICaptioner, AnthropicCaptioner, GoogleCaptioner, CaptionMerger


class ImageCaptioningPipeline:
    """Main pipeline class for image captioning"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.captioners = []
        self.merger = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all the captioning models and merger"""
        try:
            # Initialize captioners
            if self.config.get('openai_api_key'):
                self.captioners.append(
                    OpenAICaptioner(
                        self.config['openai_api_key'],
                        self.config.get('caption_model_1', 'gpt-4-vision-preview')
                    )
                )
            
            if self.config.get('anthropic_api_key'):
                self.captioners.append(
                    AnthropicCaptioner(
                        self.config['anthropic_api_key'],
                        self.config.get('caption_model_2', 'claude-3-sonnet-20240229')
                    )
                )
            
            if self.config.get('google_api_key'):
                self.captioners.append(
                    GoogleCaptioner(
                        self.config['google_api_key'],
                        self.config.get('caption_model_3', 'gemini-pro-vision')
                    )
                )
            
            # Initialize merger (using OpenAI by default)
            if self.config.get('openai_api_key'):
                self.merger = CaptionMerger(
                    self.config['openai_api_key'],
                    self.config.get('merge_model', 'gpt-4-turbo-preview')
                )
            
            if not self.captioners:
                raise ValueError("No valid API keys provided. Please check your configuration.")
            
            if not self.merger:
                raise ValueError("No valid API key for caption merging. OpenAI API key is required for merging.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
    
    def process_single_image(self, image_path: str, user_caption: str = "", output_dir: str = None) -> Dict:
        """Process a single image and generate merged caption"""
        try:
            image_path = Path(image_path)
            image_name = image_path.stem
            
            # Generate captions from multiple models
            captions = []
            caption_sources = []
            
            print(f"Generating captions for {image_path.name}...")
            
            for i, captioner in enumerate(self.captioners):
                try:
                    caption = captioner.caption_image(str(image_path), user_caption)
                    captions.append(caption)
                    caption_sources.append(f"Model_{i+1}")
                    print(f"  ✓ Caption {i+1} generated")
                    time.sleep(1)  # Rate limiting
                except Exception as e:
                    print(f"  ✗ Caption {i+1} failed: {str(e)}")
                    captions.append(f"Failed to generate caption: {str(e)}")
                    caption_sources.append(f"Model_{i+1}_Error")
            
            # Merge captions
            print("  Merging captions...")
            try:
                merged_caption = self.merger.merge_captions(captions, user_caption, image_name)
                print("  ✓ Captions merged successfully")
            except Exception as e:
                print(f"  ✗ Caption merging failed: {str(e)}")
                merged_caption = f"Failed to merge captions: {str(e)}"
            
            result = {
                'image_path': str(image_path),
                'image_name': image_name,
                'individual_captions': dict(zip(caption_sources, captions)),
                'user_caption': user_caption,
                'merged_caption': merged_caption,
                'success': True
            }
            
            # Save caption if output directory is specified
            if output_dir:
                self._save_caption(result, output_dir)
            
            return result
            
        except Exception as e:
            return {
                'image_path': str(image_path),
                'image_name': image_path.stem if hasattr(image_path, 'stem') else 'unknown',
                'error': str(e),
                'success': False
            }
    
    def process_image_folder(self, folder_path: str, output_dir: str = None, user_caption_source: str = "folder") -> List[Dict]:
        """Process all images in a folder"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
        image_files = []
        
        # Check if it's a dataset structure
        images_folder = folder_path / 'images'
        if images_folder.exists():
            print(f"Found dataset structure, processing images from: {images_folder}")
            for ext in image_extensions:
                image_files.extend(images_folder.glob(f'*{ext}'))
                image_files.extend(images_folder.glob(f'*{ext.upper()}'))
        else:
            print(f"Processing images directly from: {folder_path}")
            for ext in image_extensions:
                image_files.extend(folder_path.glob(f'*{ext}'))
                image_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No image files found in {folder_path}")
        
        print(f"Found {len(image_files)} images to process")
        
        # Prepare output directory
        if output_dir:
            output_path = Path(output_dir)
            captions_dir = output_path / 'captions'
            captions_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        results = []
        
        for image_file in tqdm(image_files, desc="Processing images"):
            # Determine user caption
            user_caption = ""
            if user_caption_source == "folder":
                user_caption = folder_path.name
            elif user_caption_source == "filename":
                user_caption = image_file.stem
            
            result = self.process_single_image(str(image_file), user_caption, output_dir)
            results.append(result)
            
            # Small delay to avoid overwhelming APIs
            time.sleep(0.5)
        
        # Save summary
        if output_dir:
            self._save_summary(results, output_dir)
        
        return results
    
    def _save_caption(self, result: Dict, output_dir: str):
        """Save individual caption to file"""
        output_path = Path(output_dir)
        captions_dir = output_path / 'captions'
        captions_dir.mkdir(parents=True, exist_ok=True)
        
        caption_file = captions_dir / f"{result['image_name']}.txt"
        
        with open(caption_file, 'w', encoding='utf-8') as f:
            f.write(result['merged_caption'])
    
    def _save_summary(self, results: List[Dict], output_dir: str):
        """Save processing summary to JSON file"""
        output_path = Path(output_dir)
        summary_file = output_path / 'captioning_summary.json'
        
        summary = {
            'total_images': len(results),
            'successful': len([r for r in results if r.get('success', False)]),
            'failed': len([r for r in results if not r.get('success', False)]),
            'results': results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Summary saved to: {summary_file}")


def load_config() -> Dict[str, str]:
    """Load configuration from environment variables"""
    load_dotenv()
    
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
        'google_api_key': os.getenv('GOOGLE_API_KEY'),
        'caption_model_1': os.getenv('CAPTION_MODEL_1', 'gpt-4-vision-preview'),
        'caption_model_2': os.getenv('CAPTION_MODEL_2', 'claude-3-sonnet-20240229'),
        'caption_model_3': os.getenv('CAPTION_MODEL_3', 'gemini-pro-vision'),
        'merge_model': os.getenv('MERGE_MODEL', 'gpt-4-turbo-preview')
    }
    
    return config


@click.command()
@click.option('--input', '-i', required=True, help='Path to image file or folder containing images')
@click.option('--output', '-o', help='Output directory for captions (optional)')
@click.option('--user-caption', '-c', default='', help='Additional caption context from user')
@click.option('--caption-source', default='folder', 
              type=click.Choice(['folder', 'filename', 'manual']),
              help='Source for additional context: folder name, filename, or manual input')
@click.option('--config-file', help='Path to custom config file (optional)')
def main(input, output, user_caption, caption_source, config_file):
    """
    Image Captioning Pipeline
    
    Generate captions for images using multiple LLM models and merge them into
    comprehensive descriptions.
    
    Examples:
    
    Single image:
    python pipeline.py -i /path/to/image.jpg -o /path/to/output
    
    Folder of images:
    python pipeline.py -i /path/to/images/ -o /path/to/output
    
    Dataset structure:
    python pipeline.py -i /path/to/dataset/ -o /path/to/output
    """
    
    try:
        # Load configuration
        config = load_config()
        
        # Validate that we have at least one API key
        if not any([config['openai_api_key'], config['anthropic_api_key'], config['google_api_key']]):
            click.echo("Error: No API keys found. Please set up your .env file with at least one API key.")
            click.echo("Copy .env.example to .env and fill in your API keys.")
            sys.exit(1)
        
        # Initialize pipeline
        pipeline = ImageCaptioningPipeline(config)
        
        input_path = Path(input)
        
        if input_path.is_file():
            # Single image processing
            click.echo(f"Processing single image: {input_path}")
            
            if caption_source == 'manual':
                user_context = user_caption
            elif caption_source == 'filename':
                user_context = input_path.stem
            else:
                user_context = input_path.parent.name
            
            result = pipeline.process_single_image(str(input_path), user_context, output)
            
            if result['success']:
                click.echo("\n✓ Processing completed successfully!")
                click.echo(f"Merged caption: {result['merged_caption']}")
            else:
                click.echo(f"\n✗ Processing failed: {result.get('error', 'Unknown error')}")
                
        elif input_path.is_dir():
            # Folder processing
            click.echo(f"Processing folder: {input_path}")
            
            if caption_source == 'manual':
                caption_src = 'folder'  # Default for batch processing
            else:
                caption_src = caption_source
            
            results = pipeline.process_image_folder(str(input_path), output, caption_src)
            
            successful = len([r for r in results if r.get('success', False)])
            total = len(results)
            
            click.echo(f"\n✓ Processing completed!")
            click.echo(f"Successfully processed: {successful}/{total} images")
            
            if output:
                click.echo(f"Captions saved to: {output}/captions/")
                click.echo(f"Summary saved to: {output}/captioning_summary.json")
        
        else:
            click.echo(f"Error: Input path does not exist: {input_path}")
            sys.exit(1)
    
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
