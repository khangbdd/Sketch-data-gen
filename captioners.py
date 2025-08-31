import base64
from http import client
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from google import genai
from PIL import Image
from groq import Groq
from openai import OpenAI

import io

from google.genai import types


class LLMCaptioner(ABC):
    """Abstract base class for LLM image captioners"""
    
    @abstractmethod
    def caption_image(self, image_path: str, additional_context: str = "") -> str:
        """Generate a caption for the given image"""
        pass


class OpenAICaptioner(LLMCaptioner):
    def __init__(self, api_key: str, model: str = "gemma-3-27b-it"):
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def caption_image(self, image_path: str, additional_context: str = "") -> str:
        """Generate caption using Google's Gemini vision model"""
        try:
            # 1. Load the image
            img = Image.open(image_path)

            # 2. Create an in-memory byte stream
            img_byte_arr = io.BytesIO()
            
            # 3. Save the image to the stream, preserving its original format
            #    img.format will be 'JPEG', 'PNG', etc.
            img.save(img_byte_arr, format=img.format)
            
            # 4. Get the full, encoded byte content from the stream
            img_bytes = img_byte_arr.getvalue()

            # 5. Set the correct MIME type based on the image's actual format
            mime_type = f"image/{img.format.lower()}"
                
            prompt = f"Describe this image in detail (Focus on the main garment). Include information about objects, setting, colors, pattern, and composition. Be specific and comprehensive."
            if additional_context:
                prompt += f" Additional context: {additional_context}"
            
            response = self.client.models.generate_content(
                model= self.model,
                contents= [
                    prompt,
                    types.Part.from_bytes(data = img_bytes, mime_type=mime_type)
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            print(response.text)
            return response.text
        except Exception as e:
            return f"Error generating Google caption: {str(e)}"


class FacebookCaptioner(LLMCaptioner):
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def caption_image(self, image_path: str, additional_context: str = "") -> str:
        """Generate caption using Facebook's LLaMA vision model"""
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
            
            prompt = f"Describe this image in detail (Focus on the main garment). Include information about objects, setting, colors, pattern, and composition. Be specific and comprehensive."
            if additional_context:
                prompt += f" Additional context: {additional_context}"
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type":"text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_type};base64,{image_data}",
                                },
                            },
                        ]
                    }
                ],
                model= self.model
            )
            content = chat_completion.choices[0].message.content
            print(chat_completion.choices[0].message.content)
            return content
        except Exception as e:
            return f"Error generating Llama caption: {str(e)}"


class GoogleCaptioner(LLMCaptioner):

    def __init__(self, api_key: str, model: str = "gemini-pro-vision"):
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def caption_image(self, image_path: str, additional_context: str = "") -> str:
        """Generate caption using Google's Gemini vision model"""
        try:
            # 1. Load the image
            img = Image.open(image_path)

            # 2. Create an in-memory byte stream
            img_byte_arr = io.BytesIO()
            
            # 3. Save the image to the stream, preserving its original format
            #    img.format will be 'JPEG', 'PNG', etc.
            img.save(img_byte_arr, format=img.format)
            
            # 4. Get the full, encoded byte content from the stream
            img_bytes = img_byte_arr.getvalue()

            # 5. Set the correct MIME type based on the image's actual format
            mime_type = f"image/{img.format.lower()}"
                
            prompt = f"Describe this image in detail (Focus on the main garment). Include information about objects, setting, colors, pattern, and composition. Be specific and comprehensive."
            if additional_context:
                prompt += f" Additional context: {additional_context}"
            
            response = self.client.models.generate_content(
                model= self.model,
                contents= [
                    prompt,
                    types.Part.from_bytes(data = img_bytes, mime_type=mime_type)
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            print(response.text)
            return response.text
        except Exception as e:
            return f"Error generating Google caption: {str(e)}"


class CaptionMerger:
    """Merges multiple captions into a single comprehensive caption"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
    
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

            response = self.client.models.generate_content(
                model= self.model,
                contents= prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            print(response.text)
            return response.text
        except Exception as e:
            return f"Error merging captions: {str(e)}"
