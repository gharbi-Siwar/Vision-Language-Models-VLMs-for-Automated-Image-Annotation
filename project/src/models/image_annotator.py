import abc
import numpy as np
from PIL import Image
from typing import Any, List, Optional
from copy import deepcopy
from .base import BaseVLMAnnotator
from utils import image_utils


class BaseImageAnnotator(BaseVLMAnnotator):
    """Base class for image-based VLM annotators."""

    # ---- 1. Image preprocessing ----
    def preprocess_image(self, image_np: np.ndarray) -> Image.Image:
        """Convert a NumPy image to a PIL image."""
        return image_utils.np_to_pil(image_np)

    def numpy_to_bytes(self, image_np: np.ndarray, format: str = "PNG") -> bytes:
        """Convert a NumPy image to bytes."""
        return image_utils.numpy_image_to_bytes(image_np, format=format)

    # ---- 2. Model loading ----
    @abc.abstractmethod
    def load_model(self) -> Any:
        """Load and return the VLM model. Must be implemented by subclass."""
        raise NotImplementedError

    # ---- 3. Model-specific estimation ----
    @abc.abstractmethod
    def generate_estimation(self, image_np: np.ndarray, prompt: str) -> str:
        """Generate prediction/description for a single image."""
        raise NotImplementedError

    # ---- 4. Batch inference ----
    def infer_batch_images(
        self,
        images_np: List[np.ndarray],
        prompts: List[str],
        max_retries: int = 3,
        additional_attributes: Optional[List[str]] = None,
    ) -> List[Any]:
        """Batch inference for multiple images, safely handling empty responses."""
        outputs: List[Any] = []

        for attempt in range(max_retries):
            try:
                response = super().infer_batch(
                    inputs=images_np,
                    prompts=prompts,
                    max_retries=1,
                    additional_attributes=additional_attributes,
                )

                outputs = []
                for res in response:
                    if isinstance(res, dict) and res:
                        outputs.append(res)
                    else:
                        outputs.append({"error": "No valid output"})

                if any("error" not in o for o in outputs):
                    break

            except Exception:
                outputs = [{"error": "Failed batch"} for _ in images_np]

        return outputs

    # ---- 5. Default field handling (reuse from base) ----
    def get_default_fields(self, additional_attributes: Optional[List[str]] = None) -> dict:
        config_defaults = self.config.get("default_fields", {})
        attributes_list: List[str] = additional_attributes or config_defaults.get("attributes", [])
        attributes_dict = {attr: None for attr in attributes_list}

        defaults = config_defaults.copy()
        defaults["attributes"] = attributes_dict
        return defaults

    def recursive_merge(self, defaults_dict: dict, data_dict: dict) -> dict:
        for key, value in defaults_dict.items():
            if isinstance(value, dict):
                if key not in data_dict or not isinstance(data_dict.get(key), dict):
                    data_dict[key] = deepcopy(value)
                else:
                    self.recursive_merge(value, data_dict[key])
            else:
                if key not in data_dict or data_dict[key] in [None, ""]:
                    data_dict[key] = value
        return data_dict

    def validate_and_complete_output(
        self,
        model_output: dict,
        additional_attributes: Optional[List[str]] = None
    ) -> dict:
        defaults: dict = self.get_default_fields(additional_attributes)
        validated_output: dict = deepcopy(defaults)

        if "reasoning" in model_output and isinstance(model_output["reasoning"], str):
            model_output["reasoning"] = model_output["reasoning"].replace('"', '').strip()

        return self.recursive_merge(validated_output, model_output)


