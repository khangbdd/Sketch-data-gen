# Image Captioning Pipeline

A comprehensive Python pipeline that uses multiple Large Language Model (LLM) APIs to generate and merge image captions. The pipeline generates captions from 3 different LLM models and then uses the strongest model to merge them into a single, comprehensive caption without duplicated information.

## Features

- **Multi-Model Captioning**: Uses OpenAI GPT-4 Vision, Anthropic Claude 3, and Google Gemini Pro Vision
- **Intelligent Merging**: Combines multiple captions into one comprehensive description
- **Batch Processing**: Process single images or entire folders
- **Dataset Structure Support**: Works with organized dataset structures
- **Command Line Interface**: Easy-to-use CLI with various options
- **Flexible Context**: Uses folder names or filenames as additional context
- **Progress Tracking**: Real-time progress bars for batch processing

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

4. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

## Usage

### Command Line Interface

```bash
python pipeline.py [OPTIONS]
```

#### Options:
- `--input, -i`: Path to image file or folder containing images (required)
- `--output, -o`: Output directory for captions (optional)
- `--user-caption, -c`: Additional caption context from user (optional)
- `--caption-source`: Source for additional context - 'folder', 'filename', or 'manual' (default: 'folder')
- `--config-file`: Path to custom config file (optional)

### Examples

#### Single Image Processing
```bash
# Basic single image captioning
python pipeline.py -i /path/to/image.jpg

# With output directory
python pipeline.py -i /path/to/image.jpg -o /path/to/output

# With custom user caption
python pipeline.py -i /path/to/image.jpg -c "A photo from my vacation" -o /path/to/output
```

#### Folder Processing
```bash
# Process all images in a folder
python pipeline.py -i /path/to/images/ -o /path/to/output

# Use folder name as context
python pipeline.py -i /path/to/vacation_photos/ -o /path/to/output --caption-source folder

# Use filenames as context
python pipeline.py -i /path/to/images/ -o /path/to/output --caption-source filename
```

#### Dataset Processing
```bash
# Process dataset with /images and /captions structure
python pipeline.py -i /path/to/dataset/ -o /path/to/output
```

## API Configuration

The pipeline supports three LLM providers:

### Required API Keys
- **OpenAI**: Required for caption merging, optional for captioning
- **Anthropic**: Optional for captioning
- **Google**: Optional for captioning

### Model Configuration
You can customize which models to use in your `.env` file:

```
# Caption generation models
CAPTION_MODEL_1=gpt-4-vision-preview
CAPTION_MODEL_2=claude-3-sonnet-20240229
CAPTION_MODEL_3=gemini-pro-vision

# Merging model (should be the strongest)
MERGE_MODEL=gpt-4-turbo-preview
```

## Output Structure

### Single Image Output
- Text file with merged caption: `{image_name}.txt`
- Console output with individual and merged captions

### Batch Processing Output
```
/output
    /captions
        image001.txt
        image002.txt
        image003.txt
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
