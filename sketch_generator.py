#!/usr/bin/env python3
"""
Sketch Generator Module

Integrates the informative-drawings model for translating images to sketches.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
import shutil


class SketchGenerator:
    """Handles sketch generation using the informative-drawings model"""
    
    def __init__(self, model_name: str = "anime_style", checkpoints_dir: str = None):
        self.model_name = model_name
        self.informative_drawings_dir = Path(__file__).parent / "informative-drawings"
        self.checkpoints_dir = checkpoints_dir or str(self.informative_drawings_dir / "checkpoints")
        
        # Verify that the informative-drawings directory exists
        if not self.informative_drawings_dir.exists():
            raise FileNotFoundError(f"Informative-drawings directory not found: {self.informative_drawings_dir}")
        
        # Verify that test.py exists
        self.test_script = self.informative_drawings_dir / "test.py"
        if not self.test_script.exists():
            raise FileNotFoundError(f"Test script not found: {self.test_script}")
    
    def generate_sketches(self, input_dir: str, output_dir: str, **kwargs) -> Dict:
        """
        Generate sketches for all images in the input directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save generated sketches
            **kwargs: Additional arguments for the test script
        
        Returns:
            Dict with generation results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure we use absolute paths to avoid permission issues
        abs_input_path = input_path.resolve()
        abs_output_path = output_path.resolve()
        
        # Prepare command arguments
        cmd_args = [
            sys.executable, str(self.test_script),
            "--name", self.model_name,
            "--dataroot", str(abs_input_path),
            "--results_dir", str(abs_output_path),
            "--checkpoints_dir", self.checkpoints_dir,
            "--how_many", "99999",  # Process all images (large number)
            "--no_flip",  # Don't flip images for data augmentation
        ]
        
        # Add additional arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd_args.extend([f"--{key}", str(value)])
        
        try:
            print(f"Generating sketches for images in: {abs_input_path}")
            print(f"Output directory: {abs_output_path}")
            print(f"Working directory: {self.informative_drawings_dir}")
            print(f"Command: {' '.join(cmd_args)}")
            
            # Verify input directory exists and has images
            if not abs_input_path.exists():
                return {
                    "success": False,
                    "error": f"Input directory does not exist: {abs_input_path}"
                }
            
            # Check for image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(abs_input_path.glob(f'*{ext}'))
                image_files.extend(abs_input_path.glob(f'*{ext.upper()}'))
            
            if not image_files:
                return {
                    "success": False,
                    "error": f"No image files found in: {abs_input_path}"
                }
            
            print(f"Found {len(image_files)} image files to process")
            
            # Verify model checkpoint exists
            model_checkpoint = Path(self.checkpoints_dir) / self.model_name / "netG_A_latest.pth"
            if not model_checkpoint.exists():
                return {
                    "success": False,
                    "error": f"Model checkpoint not found: {model_checkpoint}. Available models: {self.list_available_models()}"
                }
            
            print(f"Using model checkpoint: {model_checkpoint}")
            
            # Change to informative-drawings directory
            original_cwd = os.getcwd()
            os.chdir(self.informative_drawings_dir)
            
            # Run the sketch generation
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                cwd=self.informative_drawings_dir
            )
            
            # Change back to original directory
            os.chdir(original_cwd)
            
            if result.returncode == 0:
                # Count generated files
                sketch_dir = abs_output_path / self.model_name
                if sketch_dir.exists():
                    sketch_files = list(sketch_dir.glob("*_out.png"))
                    return {
                        "success": True,
                        "output_dir": str(sketch_dir),
                        "sketches_generated": len(sketch_files),
                        "sketch_files": [str(f) for f in sketch_files],
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Output directory not created: {sketch_dir}",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
            else:
                error_msg = f"Sketch generation failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nStderr: {result.stderr}"
                if result.stdout:
                    error_msg += f"\nStdout: {result.stdout}"
                
                return {
                    "success": False,
                    "error": error_msg,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
                
        except Exception as e:
            # Ensure we return to original directory
            os.chdir(original_cwd)
            return {
                "success": False,
                "error": f"Exception during sketch generation: {str(e)}"
            }
    
    def process_single_image(self, image_path: str, output_dir: str, **kwargs) -> Dict:
        """
        Process a single image by creating a temporary directory structure
        
        Args:
            image_path: Path to the input image
            output_dir: Directory to save the generated sketch
            **kwargs: Additional arguments for the test script
        
        Returns:
            Dict with generation results
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Create temporary input directory
        temp_input_dir = Path(output_dir) / "temp_input"
        temp_input_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy image to temporary directory
            temp_image_path = temp_input_dir / image_path.name
            shutil.copy2(image_path, temp_image_path)
            
            # Generate sketch
            result = self.generate_sketches(str(temp_input_dir), output_dir, **kwargs)
            
            # If successful, move the output to a more convenient location
            if result["success"]:
                sketch_dir = Path(result["output_dir"])
                if sketch_dir.exists():
                    # Find the generated sketch
                    sketch_files = list(sketch_dir.glob(f"{image_path.stem}_out.png"))
                    if sketch_files:
                        # Move sketch to main output directory
                        final_sketch_path = Path(output_dir) / f"{image_path.stem}_sketch.png"
                        shutil.move(sketch_files[0], final_sketch_path)
                        result["final_sketch_path"] = str(final_sketch_path)
            
            return result
            
        finally:
            # Clean up temporary directory
            if temp_input_dir.exists():
                shutil.rmtree(temp_input_dir)
    
    def check_model_availability(self) -> bool:
        """Check if the required model checkpoints are available"""
        model_checkpoint = Path(self.checkpoints_dir) / self.model_name / "netG_A_latest.pth"
        return model_checkpoint.exists()
    
    def list_available_models(self) -> List[str]:
        """List all available model checkpoints"""
        checkpoints_path = Path(self.checkpoints_dir)
        if not checkpoints_path.exists():
            return []
        
        models = []
        for model_dir in checkpoints_path.iterdir():
            if model_dir.is_dir():
                checkpoint_file = model_dir / "netG_A_latest.pth"
                if checkpoint_file.exists():
                    models.append(model_dir.name)
        
        return models
