"""
Minimal example: Using Google's Gemini Vision-Language Model (VLM).

Steps:
1. Configure the client with your API key.
2. Create a GenerativeModel instance with the Gemini model name of your choice.
3. Provide text + image as input.
"""

import os
from google import generativeai as genai

# Configure Gemini client (requires GEMINI_API_KEY in your environment)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model (replace 'your-model-name' with the Gemini model you want to use)
model = genai.GenerativeModel("your-model-name")

# Example usage: send text + image
response = model.generate_content([
    "Describe this image",
    {"mime_type": "image/jpeg", "data": open("example.jpg", "rb").read()}
])

print(response.text)
