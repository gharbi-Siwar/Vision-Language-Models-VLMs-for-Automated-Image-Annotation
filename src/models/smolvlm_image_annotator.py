import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from typing import Tuple, Dict, Any
from .image_annotator import BaseImageAnnotator


class SmollVLM(BaseImageAnnotator):
    """SmollVLM for image annotation using HuggingFace Transformers."""

    def __init__(self, config_path: str) -> None:
        """
        Constructor for SmollVLM.
        - Initializes BaseImageAnnotator with the config file.
        - Loads the SmollVLM model and processor.
        """
        super().__init__(config_path)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor, self.model = self.load_model()

    def load_model(self) -> Tuple[AutoProcessor, AutoModelForImageTextToText]:
        """
        Loads the SmollVLM model and its processor.

        Returns:
            Tuple[AutoProcessor, AutoModelForImageTextToText]: processor and model ready for inference
        """
        processor: AutoProcessor = AutoProcessor.from_pretrained(self.config["vlm_model_path"])
        model: AutoModelForImageTextToText = AutoModelForImageTextToText.from_pretrained(
            self.config["vlm_model_path"],
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        model.eval()
        return processor, model

    def generate_estimation(self, image_np: np.ndarray, prompt: str) -> str:
        """
        Generates an estimation/description from a NumPy image and a text prompt.

        Args:
            image_np: Image as a NumPy array
            prompt: Text describing what the model should generate

        Returns:
            str: Generated text from the model
        """
        # Use BaseImageAnnotator helper to convert to PIL
        pil_img: Image.Image = self.preprocess_image(image_np)

        messages: list[Dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs: Dict[str, torch.Tensor] = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids: torch.Tensor = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=self.config.get("temperature", 0.0),
                max_new_tokens=self.config.get("max_new_tokens", 125),
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]





