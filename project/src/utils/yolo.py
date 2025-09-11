from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional,Dict
import numpy.typing as npt


def load_yolo_model(model_path="yolov8n.pt"):
    """
    Load YOLO model from given weights path.
    Default path is 'yolov8n.pt'.
    """
    return YOLO(model_path)



def detect_objects(
    model,
    image_bgr: npt.NDArray[np.uint8],
    dynamic_filter: bool = False,
    min_ratio_to_median: Optional[float] = 0.7,
    roi: Optional[Tuple[int, int, int, int]] = None,
    use_roi: bool = False
) -> List[Dict[str, float]]:
    """
    Detect all objects in an image using YOLO, with optional ROI and dynamic filtering.

    Returns a list of dictionaries with keys:
        'box': [x1, y1, x2, y2]
        'class_id': int
        'score': float

    Args:
        model: YOLO model instance
        image_bgr: numpy array in BGR format (H x W x 3)
        dynamic_filter: if True, ignore small/remote objects based on median height
        min_ratio_to_median: minimum proportion of median height to keep (used only if dynamic_filter=True)
        roi: optional region of interest (x1, y1, x2, y2)
        use_roi: if True, apply static ROI filtering
    """
    results = model(image_bgr)
    detections = results[0].boxes.data.cpu().numpy()
    
    # Extract boxes, class IDs, and scores
    objects: List[Dict[str, float]] = [
        {
            "box": list(map(int, box[:4])),
            "score": float(box[4]),
            "class_id": int(box[5])
        }
        for box in detections
    ]

    # --- Optional Static ROI filtering ---
    if use_roi and roi is not None:
        rx1, ry1, rx2, ry2 = roi
        objects = [
            obj for obj in objects
            if not (obj["box"][2] < rx1 or obj["box"][0] > rx2 or
                    obj["box"][3] < ry1 or obj["box"][1] > ry2)
        ]
    
    # --- Optional Dynamic median-height filtering ---
    if dynamic_filter and objects and min_ratio_to_median is not None:
        heights = [obj["box"][3] - obj["box"][1] for obj in objects]
        median_height = float(np.median(heights))
        threshold = median_height * min_ratio_to_median
        objects = [obj for obj in objects if (obj["box"][3] - obj["box"][1]) >= threshold]
    
    return objects



def get_crops(image_bgr: npt.NDArray[np.uint8], detected_objects: List[dict]) -> List[npt.NDArray[np.uint8]]:
    """
    Crop detected people from the image.

    Args:
        image_bgr: original image (numpy array)
        detected_objects: list of dicts from detect_objects() with keys 'box', 'class_id', 'score'

    Returns:
        List of cropped images (numpy arrays) for people only
    """
    crops = []
    for obj in detected_objects:
        if obj["class_id"] == 0:  # COCO class 0 = person
            x1, y1, x2, y2 = obj["box"]
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
    return crops

