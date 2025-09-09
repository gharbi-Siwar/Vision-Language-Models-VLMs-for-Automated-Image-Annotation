import google.generativeai as genai
from PIL import Image
from io import BytesIO
import numpy as np
from typing import Any
from .image_annotator import BaseImageAnnotator


class GeminiVLM(BaseImageAnnotator):
    """Gemini VLM for image annotation using Google Generative AI."""

    def __init__(self, config_path: str) -> None:
        """
        Constructor for GeminiVLM.
        - Initializes BaseImageAnnotator with the config file.
        - Configures the Google Generative AI API key.
        - Loads the Gemini model.
        """
        super().__init__(config_path)
        genai.configure(api_key=self.config["google_api_key"])
        self.model = self.load_model()

    def load_model(self) -> Any:
        """
        Loads the Gemini generative model.

        Returns:
            generative_model: the Gemini generative model instance
        """
        return genai.GenerativeModel(self.config["vlm_model_path"])

    def generate_estimation(self, image_np: np.ndarray, prompt: str) -> str:
        """
        Generates an estimation/description from a NumPy image and a prompt.

        Args:
            image_np: Image as a NumPy array
            prompt: Text describing what the model should generate

        Returns:
            str: Generated text from Gemini model
        """
        # Use BaseImageAnnotator helper to convert to PIL
        image_pil: Image.Image = self.preprocess_image(image_np)

        response = self.model.generate_content([prompt, image_pil])
        return response.text




