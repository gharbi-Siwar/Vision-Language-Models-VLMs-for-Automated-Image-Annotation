"""
vlm_simple_explainer_with_doc.py

This script demonstrates a simple workflow for using a Vision-Language Model (VLM)
to generate text based on image and/or text input. 

Key Steps:
1. Load the processor and model.
2. Prepare input messages containing images and text.
3. Generate output from the model, optionally limiting output length using max_new_tokens.
4. Decode the generated output to human-readable text.
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# -----------------------------
# Step 1: Load the processor and model
# -----------------------------
# The processor handles preprocessing of images and text into model-readable format
processor = AutoProcessor.from_pretrained("your-model-path")

# Load the VLM model for image+text tasks
# The model is moved to GPU if available for faster inference
model = AutoModelForImageTextToText.from_pretrained("your-model-path").to("cuda")

# -----------------------------
# Step 2: Prepare input
# -----------------------------
# Messages is a structured list containing image(s) and optional text instructions
# Each message has a "role" (user/model) and "content" list with images and/or text
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "path_or_image_data_here"},
            {"type": "text", "text": "Your question or instruction here"}
        ]
    }
]

# Convert messages into model-readable tensors
# apply_chat_template handles tokenization and formatting for the VLM
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# -----------------------------
# Step 3: Generate output
# -----------------------------
# max_new_tokens limits the length of the generated text; here it is left general
generated_ids = model.generate(
    **inputs,
    do_sample=False,       # deterministic output
    max_new_tokens=None    # optional: can specify to limit response length
)

# -----------------------------
# Step 4: Decode and read output
# -----------------------------
# Convert the generated token IDs into human-readable text
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)

# Print the result
print("VLM Output:", generated_texts[0])
