# Image Captioning and Sketch Generation Pipeline

A comprehensive Python pipeline that uses multiple Large Language Model (LLM) APIs to generate and merge image captions, and optionally converts images to sketches using the informative-drawings model. The pipeline generates captions from multiple LLM models and merges them into a single, comprehensive caption. It can also independently generate artistic sketches from input images.

## Features

### Captioning Features
- **Multi-Model Captioning**: Uses OpenAI GPT-4 Vision, Google Gemini Pro Vision, and Groq models
- **Intelligent Merging**: Combines multiple captions into one comprehensive description
- **Batch Processing**: Process single images or entire folders
- **Dataset Structure Support**: Works with organized dataset structures
- **Command Line Interface**: Easy-to-use CLI with various options
- **Flexible Context**: Uses folder names or filenames as additional context
- **Progress Tracking**: Real-time progress bars for batch processing

### Sketch Generation Features
- **Image-to-Sketch Translation**: Convert photos to artistic sketches using deep learning
- **Multiple Model Support**: Support for different artistic styles (anime_style, etc.)
- **Batch Processing**: Process entire folders of images
- **Independent Operation**: Can run sketch generation independently of captioning
- **Combined Pipeline**: Run both captioning and sketch generation together

## Dataset Structure

The pipeline supports the following dataset structure:

```
/dataset
    /images
        image001.jpg
        image002.jpg
        image003.jpg
        ...
    /captions
        image001.txt
        image002.txt
        image003.txt
        ...
```

## Installation

1. **Clone or download the project files**

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Configure API keys**:
   - Copy `.env.example` to `.env`
   - Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Activate the conda environment**:
   ```bash
   conda activate sketch-data
   ```

## Usage

### Command Line Interface

```bash
python pipeline.py [OPTIONS]
```

#### Options:
- `--input, -i`: Path to image file or folder containing images (required)
- `--output, -o`: Output directory for captions and sketches (optional)
- `--user-caption, -c`: Additional caption context from user (optional)
- `--caption-source`: Source for additional context - 'folder', 'filename', or 'manual' (default: 'folder')
- `--generate-sketches, -s`: Generate sketches using informative-drawings model (flag)
- `--sketch-model`: Name of the sketch generation model (default: anime_style)
- `--caption-only`: Only generate captions, skip sketch generation (flag)
- `--sketch-only`: Only generate sketches, skip caption generation (flag)
- `--config-file`: Path to custom config file (optional)

### Examples

#### Caption Generation Only
```bash
# Basic single image captioning
python pipeline.py -i /path/to/image.jpg

# With output directory
python pipeline.py -i /path/to/image.jpg -o /path/to/output

# With custom user caption
python pipeline.py -i /path/to/image.jpg -c "A photo from my vacation" -o /path/to/output

# Process all images in a folder
python pipeline.py -i /path/to/images/ -o /path/to/output
```

#### Sketch Generation Only
```bash
# Generate sketch for single image
python pipeline.py -i /path/to/image.jpg -o /path/to/output --sketch-only

# Generate sketches for all images in folder
python pipeline.py -i /path/to/images/ -o /path/to/output --sketch-only

# Use specific sketch model
python pipeline.py -i /path/to/images/ -o /path/to/output --sketch-only --sketch-model anime_style
```

#### Combined Caption and Sketch Generation
```bash
# Generate both captions and sketches for single image
python pipeline.py -i /path/to/image.jpg -o /path/to/output --generate-sketches

# Process folder with both captions and sketches
python pipeline.py -i /path/to/images/ -o /path/to/output --generate-sketches --sketch-model anime_style

# Use folder name as context
python pipeline.py -i /path/to/vacation_photos/ -o /path/to/output --generate-sketches --caption-source folder

# Use filenames as context
python pipeline.py -i /path/to/images/ -o /path/to/output --caption-source filename
```

#### Dataset Processing
```bash
# Process dataset with /images and /captions structure (captions only)
python pipeline.py -i /path/to/dataset/ -o /path/to/output

# Process dataset with both captions and sketches
python pipeline.py -i /path/to/dataset/ -o /path/to/output --generate-sketches
```

## Configuration

### API Configuration

The pipeline supports multiple LLM providers for caption generation:

#### Required API Keys (for captioning)
- **Google Gemini**: Required for caption generation and merging
- **Groq**: Optional for additional caption generation
- **OpenAI**: Optional for additional caption generation

#### Model Configuration
You can customize which models to use in your `.env` file:

```
# Caption generation models
CAPTION_MODEL_1=gemma-3-27b-it
CAPTION_MODEL_2=meta-llama/llama-4-scout-17b-16e-instruct  
CAPTION_MODEL_3=gemini-2.5-flash

# Model for merging captions
MERGE_MODEL=gemini-2.5-flash
```

### Sketch Generation Configuration

#### Model Requirements
- The sketch generation uses pre-trained PyTorch models from the informative-drawings project
- Model checkpoints should be placed in `informative-drawings/checkpoints/`
- Default model name is `anime_style` but can be changed with `--sketch-model`

#### Available Models
- Check available models with: `python test_sketch.py`
- Models should have the structure: `checkpoints/{model_name}/netG_A_latest.pth`

#### Dependencies
- PyTorch 2.2.0+ with torchvision and torchaudio
- CUDA support recommended for faster processing
- CLIP model for feature extraction

# Merging model (should be the strongest)
MERGE_MODEL=gpt-4-turbo-preview
```

## Output Structure

### Caption-Only Output
#### Single Image
- Text file with merged caption: `{image_name}.txt`
- Console output with individual and merged captions

#### Batch Processing
```
/output
    /captions
        image001.txt
        image002.txt
        image003.txt
        ...
    captioning_summary.json
```

### Sketch-Only Output
```
/output
    /sketches
        /anime_style (or specified model name)
            image001_out.png
            image002_out.png
            image003_out.png
            ...
```

### Combined Output (Captions + Sketches)
```
/output
    /captions
        image001.txt
        image002.txt
        image003.txt
        ...
    /sketches
        /anime_style
            image001_out.png
            image002_out.png
            image003_out.png
            ...
    captioning_summary.json
```

### Summary File
The `captioning_summary.json` contains:
- Processing statistics
- Individual captions from each model
- Merged captions
- Error information (if any)

## Pipeline Workflow

1. **Image Input**: Load single image or batch of images
2. **Multi-Model Captioning**: Generate captions using up to 3 different LLM models
3. **Context Addition**: Include user-provided context (folder name, filename, or custom text)
4. **Caption Merging**: Use the strongest LLM model to merge all captions into one comprehensive description
5. **Output Generation**: Save merged captions and processing summary

## Features in Detail

### Multi-Model Support
- **OpenAI GPT-4 Vision**: Excellent for detailed descriptions
- **Anthropic Claude 3**: Strong analytical capabilities
- **Google Gemini Pro Vision**: Good at identifying objects and scenes

### Intelligent Merging
- Removes duplicate information
- Combines unique details from all sources
- Maintains coherent narrative flow
- Prioritizes accuracy and completeness

### Error Handling
- Graceful handling of API failures
- Continues processing even if one model fails
- Detailed error reporting in summary files

### Rate Limiting
- Built-in delays to respect API rate limits
- Configurable timing between requests

## Troubleshooting

### Common Issues

1. **API Key Errors**: Make sure your `.env` file has valid API keys
2. **Rate Limiting**: The pipeline includes delays, but you may need to adjust for your API limits
3. **Model Availability**: Some models may not be available in all regions
4. **Image Format**: Supported formats: JPG, PNG, GIF, BMP, WebP, TIFF

### Requirements

- Python 3.8+
- Valid API keys for at least one LLM provider
- Internet connection for API calls

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the pipeline.

## License

This project is open source. Please check individual API provider terms for usage restrictions.
