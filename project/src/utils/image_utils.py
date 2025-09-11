import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Union


def decode_image(base64_string: str) -> Image.Image:
    """
    Decode a base64 string into a PIL Image (RGB).

    Args:
        base64_string: Base64-encoded image string

    Returns:
        PIL.Image in RGB mode
    """
    image_data: bytes = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")


def encode_image(pil_image: Image.Image) -> str:
    """
    Encode a PIL Image to a base64 string (PNG format).

    Args:
        pil_image: PIL.Image object

    Returns:
        Base64-encoded PNG string
    """
    buffer: BytesIO = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def pil_to_np(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to an OpenCV BGR image.

    Args:
        pil_image: PIL.Image object

    Returns:
        NumPy array in BGR format
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def np_to_pil(image_np: np.ndarray) -> Image.Image:
    """
    Convert a NumPy array (BGR) to a PIL Image (RGB).

    Args:
        image_np: NumPy array in BGR format

    Returns:
        PIL.Image object
    """
    return Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))


def numpy_image_to_bytes(image_np: np.ndarray, format: str = "PNG") -> bytes:
    """
    Convert a NumPy image to bytes (for GeminiVLM or similar models).

    Args:
        image_np: NumPy array in BGR format
        format: Image format (default PNG)

    Returns:
        Image bytes
    """
    pil_img: Image.Image = np_to_pil(image_np)
    buf: BytesIO = BytesIO()
    pil_img.save(buf, format=format)
    buf.seek(0)
    return buf.read()


def draw_boxes(
    image: np.ndarray,
    bboxes: List[List[float]],
    outputs: List[Union[Dict[str, Any], str]],
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw bounding boxes with labels on an image.

    Args:
        image: NumPy array, the image to draw on
        bboxes: List of [x1, y1, x2, y2] in **normalized coordinates** (0-1)
        outputs: List of dicts or JSON strings, containing 'age' and 'gender'
        color: Tuple, box color in BGR (default green)

    Returns:
        Annotated image as NumPy array
    """
    h, w = image.shape[:2]

    for bbox, output in zip(bboxes, outputs):
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = [
            int(coord * w) if i % 2 == 0 else int(coord * h)
            for i, coord in enumerate(bbox)
        ]

        # Prepare label
        label: str
        try:
            parsed: Dict[str, Any] = eval(output) if isinstance(output, str) else output
            age: Any = parsed.get("age", "?")
            gender: Any = parsed.get("gender", "?")
            label = f"{gender}, {age} yrs"
        except Exception:
            label = "Error"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return image



