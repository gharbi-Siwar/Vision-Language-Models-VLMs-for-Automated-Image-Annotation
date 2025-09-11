Multi-Model Image Annotation Pipeline (YOLO + VLMs)
🔹 Overview

This project combines YOLOv8 for person detection with Vision-Language Models (VLMs) (GeminiVLM and SmollVLM) for fine-grained annotation such as gender, age group, and more.

YOLOv8 → detects persons in images and extracts crops.

GeminiVLM → main annotator with quota tracking.

SmollVLM → runs in parallel to provide complementary annotations.

Outputs → annotated images, per-image JSON, and a global results file.

🔹 Workflow Recap

Load YOLO + VLM models.
Detect persons → extract crops.

Run both GeminiVLM and SmollVLM on each crop.

Save outputs:

Annotated images → /output/gemini/ and /output/smoll/

Per-image JSON → /output/*.json

Global results JSON → /output/all_results.json

Resume GeminiVLM from last processed image using progress.json.

🔹 Example JSON Output
```json
{
  "person_0": {
    "bounding_box": [
      0.41015625,
      0.02638888888888889,
      0.52109375,
      0.5986111111111111
    ],
    "GeminiVLM": {
      "gender": "male",
      "age_group": "58 to 68",
      "exact_age": 61,
      "attributes": {
        "glasses": false,
        "cap": false,
        "hood": false,
        "headscarf": false,
        "mask": false,
        "occlusion": true,
        "facing_camera": true,
        "season_clothing": "summer"
      },
      "reasoning": "Male features, graying hair, wrinkles, skin texture."
    },
    "SmollVLM": {
      "gender": "male",
      "age_group": "38 to 48",
      "exact_age": 38,
      "attributes": {
        "glasses": false,
        "cap": false,
        "hood": false,
        "headscarf": false,
        "mask": false,
        "occlusion": false,
        "facing_camera": true,
        "season_clothing": "summer"
      }
    }
  }
}

