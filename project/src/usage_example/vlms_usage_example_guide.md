# Vision-Language Models (VLM) Usage Guide

This guide explains how to use both **GeminiVLM** (Google’s Generative AI) and **general VLMs** (HuggingFace Transformers) to generate text from images and text prompts.

---

## 1️⃣ Workflow Diagram

      +------------------+
      |   Input Image     |
      +------------------+
               |
               v
      +------------------+
      |  Input Text/Prompt|
      +------------------+
               |
               v
      +------------------+
      |     VLM Model     |
      | (Gemini or HuggingFace) |
      +------------------+
               |
               v
      +------------------+
      |  JSON / Text Output|
      +------------------+
2️⃣ GeminiVLM Example

GeminiVLM generates text descriptions from images using Google’s Generative AI.

### Steps

1. Configure the client with your API key.
2. Create a `GenerativeModel` instance with the desired Gemini model.
3. Provide text + image as input.

3️⃣ General VLM Example

Demonstrates a simple workflow for any Vision-Language Model (VLM) using HuggingFace Transformers.

### Steps for General VLMs

1.Load the processor and model from the desired VLM checkpoint

2.Prepare input messages containing images and optional text prompts.

3.Convert messages into model-readable tensors using the processor.

4.Generate output text from the model, optionally limiting the output length.

5.Decode the generated token IDs to human-readable text.