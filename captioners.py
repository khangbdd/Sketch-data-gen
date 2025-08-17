import base64
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import openai
import anthropic
import google.generativeai as genai
from PIL import Image
import io


class LLMCaptioner(ABC):
    """Abstract base class for LLM image captioners"""
    
    @abstractmethod
    def caption_image(self, image_path: str, additional_context: str = "") -> str:
        """Generate a caption for the given image"""
        pass


class OpenAICaptioner(LLMCaptioner):
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def caption_image(self, image_path: str, additional_context: str = "") -> str:
        """Generate caption using OpenAI's vision model"""
        try:
            # Encode image to base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = f"Provide a detailed, accurate caption for this image. Focus on describing the visual elements, objects, people, actions, setting, colors, and composition."
            if additional_context:
                prompt += f" Additional context: {additional_context}"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating OpenAI caption: {str(e)}"


class AnthropicCaptioner(LLMCaptioner):
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def caption_image(self, image_path: str, additional_context: str = "") -> str:
        """Generate caption using Anthropic's Claude vision model"""
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine image type
            image_type = "image/jpeg"
            if image_path.lower().endswith('.png'):
                image_type = "image/png"
            elif image_path.lower().endswith('.webp'):
                image_type = "image/webp"
            
            prompt = f"Analyze this image and provide a comprehensive caption describing what you see. Include details about objects, people, actions, setting, colors, and overall composition."
            if additional_context:
                prompt += f" Additional context: {additional_context}"
            
            message = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_type,
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt}
                        ],
                    }
                ],
            )
            return message.content[0].text.strip()
        except Exception as e:
            return f"Error generating Anthropic caption: {str(e)}"


class GoogleCaptioner(LLMCaptioner):
    def __init__(self, api_key: str, model: str = "gemini-pro-vision"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def caption_image(self, image_path: str, additional_context: str = "") -> str:
        """Generate caption using Google's Gemini vision model"""
        try:
            # Load image
            img = Image.open(image_path)
            
            prompt = f"Describe this image in detail. Include information about objects, people, actions, setting, colors, and composition. Be specific and comprehensive."
            if additional_context:
                prompt += f" Additional context: {additional_context}"
            
            response = self.model.generate_content([prompt, img])
            return response.text.strip()
        except Exception as e:
            return f"Error generating Google caption: {str(e)}"


class CaptionMerger:
    """Merges multiple captions into a single comprehensive caption"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def merge_captions(self, captions: list, user_caption: str = "", image_name: str = "") -> str:
        """Merge multiple captions into one comprehensive caption"""
        try:
            captions_text = "\n".join([f"Caption {i+1}: {caption}" for i, caption in enumerate(captions)])
            
            prompt = f"""You are tasked with merging multiple image captions into a single, comprehensive, and well-written caption. 

Here are the captions to merge:
{captions_text}

Additional user context: {user_caption if user_caption else 'None provided'}
Image filename context: {image_name if image_name else 'None provided'}

Instructions:
1. Combine all unique information from the captions
2. Remove any duplicate or redundant information
3. Create a coherent, flowing description
4. Prioritize accuracy and completeness
5. Keep the merged caption concise but comprehensive
6. If there are conflicting details, use the most commonly mentioned or most specific one

Provide only the merged caption as your response."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error merging captions: {str(e)}"
