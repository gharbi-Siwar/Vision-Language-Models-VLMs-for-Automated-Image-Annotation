import abc
import cv2
from typing import Any, List, Dict
import numpy as np
from .base import BaseVLMAnnotator
import json
import re


class BaseVideoAnnotator(BaseVLMAnnotator):
    """Base class for video-based VLM annotators (single-word action per person)."""

    # ---- 1. Model loading ----
    @abc.abstractmethod
    def load_model(self) -> Any:
        """Load and return the video VLM model."""
        raise NotImplementedError

    # ---- 2. Frame extraction ----
    def process_video_into_frames(
        self, video_path: str, frame_skip: int = 1, resize: tuple = None
    ) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_skip == 0:
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                frames.append(frame)
            frame_idx += 1

        cap.release()
        return frames

    # ---- 3. Model-specific estimation ----
    @abc.abstractmethod
    def generate_estimation(self, frame: np.ndarray, prompt: str) -> str:
        raise NotImplementedError

    # ---- 4. Robust JSON parsing ----
    def parse_json_output(self, generated_text: str) -> List[Dict[str, Any]]:
        """
        Extract JSON object or array from VLM output for video frames.
        Returns a list of dicts, even if output is a single JSON object.
        """
        if not generated_text or not generated_text.strip():
            return []

        # Remove code fences and whitespace
        cleaned = generated_text.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return [data]  # wrap single dict in list
            elif isinstance(data, list):
                return data
            else:
                return []
        except json.JSONDecodeError:
            # fallback: extract first {...} block
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    return [data] if isinstance(data, dict) else []
                except:
                    return []
            return []

    # ---- 5. Minimal JSON validation ----
    def get_default_fields(self) -> Dict[str, Any]:
        return {"person_id": None, "bbox": [0, 0, 0, 0], "action": "unknown"}

    def validate_and_complete_output(self, model_output: Any) -> List[Dict[str, Any]]:
        defaults = self.get_default_fields()
        validated: List[Dict[str, Any]] = []

        if isinstance(model_output, dict):
            model_output = [model_output]

        for entry in model_output:
            if not isinstance(entry, dict):
                entry = {}
            validated.append({**defaults, **entry})
        return validated

    # ---- 6. Batch inference for frames/crops ----
    def infer_batch_videos(
        self,
        video_frames: List[np.ndarray],
        prompt_template: str,
        max_retries: int = 1,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for idx, frame in enumerate(video_frames):
            output_json: List[Dict[str, Any]] = []
            for attempt in range(max_retries):
                try:
                    output_text = self.generate_estimation(frame, prompt_template)
                    output_json = self.parse_json_output(output_text)
                    if output_json:
                        break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for frame {idx}: {e}")

            if not output_json:
                results.append(self.get_default_fields())
            else:
                validated = self.validate_and_complete_output(output_json)
                results.append(validated[0] if validated else self.get_default_fields())

        return results


   



