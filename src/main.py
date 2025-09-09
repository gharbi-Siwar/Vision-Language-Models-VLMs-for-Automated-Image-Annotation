"""
Multi-Model Image Annotation Pipeline using YOLO and VLMs

Workflow:
1. Logging setup
2. Load YOLO model
3. Dynamically load VLMs (GeminiVLM & SmollVLM)
4. Process all images from inputs/
5. Resume progress from progress.json if available
6. Detect persons with YOLO, extract crops
7. Run VLM inference on crops
8. Annotate images + save per-image JSON + global JSON
"""

import os
import json
import importlib
import logging
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

# ==== Logging setup ====
LOG_FILE = "processing.log"
logger = logging.getLogger("VLM_Pipeline")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info("==== Pipeline Start ====")

# ==== Import YOLO utils ====
try:
    from utils.yolo import load_yolo_model, detect_objects, get_crops
    logger.info("YOLO utilities imported successfully")
except ImportError as e:
    logger.error(f"Failed to import YOLO utilities: {e}")
    raise

# ==== Config ====
MODELS_FOLDER = "models"
model_order = [
    ("gemini_image_annotator", "GeminiVLM"),
    ("smolvlm_image_annotator", "SmollVLM")
]

yolo_model_path = "cfg/yolov8n.pt"
images_folder = "inputs"
progress_file = "progress.json"

output_dir = "output"
output_gemini_dir = os.path.join(output_dir, "gemini")
output_smoll_dir = os.path.join(output_dir, "smoll")
os.makedirs(output_gemini_dir, exist_ok=True)
os.makedirs(output_smoll_dir, exist_ok=True)

# ==== Load Models Dynamically ====
loaded_models = {}
for module_file, class_name in model_order:
    module_path = f"{MODELS_FOLDER}.{module_file}"
    try:
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
        config_path = Path(f"cfg/{module_file}_config.yaml")
        loaded_models[class_name] = ModelClass(config_path=config_path)
        logger.info(f"Loaded model: {class_name}")
    except Exception as e:
        logger.error(f"Failed to load model {class_name}: {e}")
        # Dummy fallback model
        class DummyModel:
            def __init__(self, config_path): self.config = {"prompt_template": "describe person"}
            def infer_batch(self, images, prompts): return [{"gender": "unknown", "exact_age": "unknown"} for _ in images]
        loaded_models[class_name] = DummyModel(None)
        logger.info(f"Using dummy model for {class_name}")

# ==== Load all images from inputs/ ====
image_paths = [os.path.join(images_folder, f) for f in os.listdir(images_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
images = [cv2.imread(p) for p in image_paths]
logger.info(f"{len(images)} images ready for processing")

# ==== Progress Handling ====
def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f)
    return {"last_gemini_index": -1}

def save_progress(last_index: int):
    with open(progress_file, "w") as f:
        json.dump({"last_gemini_index": last_index}, f)

progress = load_progress()
last_gemini_index = progress.get("last_gemini_index", -1)
logger.info(f"Gemini last completed crop index: {last_gemini_index}")

# ==== Load YOLO ====
yolo_model = load_yolo_model(yolo_model_path)
logger.info(f"YOLO model loaded from {yolo_model_path}")

# ==== Optional ROI ====
roi_static: Optional[List[int]] = None
if images:
    h, w = images[0].shape[:2]
    roi_static = [0, h // 2, w, h]

# ==== Detection ====
logger.info("Running YOLO detection on all images")
all_detected_objects: List[List[dict]] = [
    detect_objects(yolo_model, img, dynamic_filter=True,
                   min_ratio_to_median=0.7, roi=roi_static, use_roi=True)
    for img in images
]

# ==== Crops ====
all_crops: List[np.ndarray] = []
image_crop_indices: List[List[int]] = []
for idx, img in enumerate(images):
    detected_objects = all_detected_objects[idx]
    crops = get_crops(img, detected_objects)
    start_idx = len(all_crops)
    all_crops.extend(crops)
    image_crop_indices.append(list(range(start_idx, start_idx + len(crops))))
logger.info(f"Total person crops extracted: {len(all_crops)}")

# ==== VLM Inference ====
vlm_outputs: Dict[str, List[Optional[Dict]]] = {m: [None]*len(all_crops) for m in loaded_models}

batch_size = 8
for i in range(0, len(all_crops), batch_size):
    batch = all_crops[i:i+batch_size]
    for model_name in loaded_models:
        model = loaded_models[model_name]
        if model_name == "GeminiVLM" and i <= last_gemini_index:
            continue
        try:
            prompts = [model.config.get("prompt_template")] * len(batch)
            outputs = model.infer_batch(batch, prompts)
            vlm_outputs[model_name][i:i+len(batch)] = outputs
            if model_name == "GeminiVLM":
                save_progress(i + len(batch) - 1)
            logger.info(f"{model_name} batch {i} processed")
        except Exception as e:
            logger.error(f"{model_name} failed at batch {i}: {e}")

# ==== Annotate & Save ====
all_results = []
for idx, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
    img_np = images[idx]
    detected_objects = all_detected_objects[idx]
    crop_indices = image_crop_indices[idx]

    persons_results = []
    h, w = img_np.shape[:2]
    person_objects = [obj for obj in detected_objects if obj["class_id"] == 0]

    for i, obj in enumerate(person_objects):
        box = obj["box"]
        norm_box = [box[0]/w, box[1]/h, box[2]/w, box[3]/h]
        person_data = {"person_index": i, "bbox": norm_box}

        for model_name, outputs in vlm_outputs.items():
            crop_idx = crop_indices[i] if i < len(crop_indices) else None
            if crop_idx is not None and crop_idx < len(outputs):
                person_data[model_name] = outputs[crop_idx]
            else:
                person_data[model_name] = None

        persons_results.append(person_data)

    for model_name, folder in zip(["GeminiVLM", "SmollVLM"], [output_gemini_dir, output_smoll_dir]):
        img_copy = img_np.copy()
        outputs = vlm_outputs.get(model_name, [])

        for i, obj in enumerate(person_objects):
            crop_idx = crop_indices[i] if i < len(crop_indices) else None
            if crop_idx is not None and crop_idx < len(outputs):
                output = outputs[crop_idx]
            else:
                output = None

            if output is None:
                continue

            gender = output.get("gender", "unknown")
            age_group = output.get("exact_age", "unknown")
            text = f"{gender}, {age_group}"

            x1, y1, x2, y2 = map(int, obj["box"])
            color = (0, 255, 0) if model_name == "GeminiVLM" else (255, 0, 0)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            ((tw, th), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_copy, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(img_copy, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        annotated_path = os.path.join(folder, os.path.basename(img_path))
        cv2.imwrite(annotated_path, img_copy)

    # Save per-image JSON
    individual_json_path = os.path.join(output_dir, f"{Path(img_path).stem}.json")
    with open(individual_json_path, "w", encoding="utf-8") as f:
        json.dump(persons_results, f, ensure_ascii=False, indent=2)

    all_results.append({
        "image_file": os.path.basename(img_path),
        "gemini_annotated": os.path.join("gemini", os.path.basename(img_path)),
        "smoll_annotated": os.path.join("smoll", os.path.basename(img_path)),
        "persons": persons_results
    })

# ==== Save global JSON ====
output_json_path = os.path.join(output_dir, "all_results.json")
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

logger.info("==== Multi-Model Image Pipeline Completed ====")










