import abc
import json
import yaml
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional


class BaseVLMAnnotator(abc.ABC):
    """General base class for Vision-Language Models (VLMs), for both images and videos."""

    # ---- 1. Model initialization & configuration ----
    def __init__(self, config_path: str) -> None:
        config_path_obj: Path = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path_obj, encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

    @abc.abstractmethod
    def load_model(self) -> Any:
        """Load a specific VLM model. Implemented in subclasses."""
        raise NotImplementedError

    # ---- 2. Model-specific estimation ----
    @abc.abstractmethod
    def generate_estimation(self, input_data: Any, prompt: str) -> str:
        """Generate a prediction/description from input data and a prompt."""
        raise NotImplementedError

    # ---- 3. JSON extraction ----
    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract first valid JSON object from a string."""
        if not text or not isinstance(text, str):
            return None

        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        if start == -1:
            return None

        brace_count = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                brace_count += 1
            elif cleaned[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_str = cleaned[start:i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
        return None

    # ---- 4. Recursive merge for defaults ----
    @staticmethod
    def recursive_merge(defaults_dict: Dict[str, Any], data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge default values into a data dictionary."""
        for key, value in defaults_dict.items():
            if isinstance(value, dict):
                if key not in data_dict or not isinstance(data_dict.get(key), dict):
                    data_dict[key] = deepcopy(value)
                else:
                    BaseVLMAnnotator.recursive_merge(value, data_dict[key])
            else:
                if key not in data_dict or data_dict[key] in [None, ""]:
                    data_dict[key] = value
        return data_dict

    # ---- 5. Abstract defaults & validation ----
    @abc.abstractmethod
    def get_default_fields(self, additional_attributes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Return default template fields for the model output.
        Implemented by subclasses (image vs video).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate_and_complete_output(
        self, model_output: Dict[str, Any], additional_attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Fill missing fields with defaults.
        Implementation depends on image vs video annotation.
        """
        raise NotImplementedError

    # ---- 6. Batch inference ----
    def infer_batch(
        self,
        inputs: List[Any],
        prompts: List[str],
        max_retries: int = 1,
        additional_attributes: Optional[List[str]] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Run batch inference for multiple inputs and prompts."""
        results: List[Optional[Dict[str, Any]]] = []
        for i, (input_data, prompt) in enumerate(zip(inputs, prompts)):
            print(f"\nInput {i+1} — Prompt:\n{prompt}")
            output_json: Optional[Dict[str, Any]] = None
            for attempt in range(max_retries):
                output_text: str = self.generate_estimation(input_data, prompt)
                print(f"Generated text (attempt {attempt+1}):\n{output_text}")
                output_json = self.extract_json(output_text)
                if output_json is not None:
                    break
                print("Invalid JSON, retrying...")

            if output_json is None:
                print(f"Failed to generate valid JSON for input {i+1}")
                results.append(None)
                continue

            results.append(self.validate_and_complete_output(output_json, additional_attributes))
        return results



