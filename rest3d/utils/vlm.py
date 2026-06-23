import os
import json
import re
import numpy as np
from PIL import Image, ImageDraw
import base64
from io import BytesIO

# Global variable to store selected VLM backend
_VLM_BACKEND = "gemini"  # options: gemini, gpt4o


def set_vlm_backend(backend):
    """Set which VLM backend to use: 'gemini' or 'gpt4o'"""
    global _VLM_BACKEND
    _VLM_BACKEND = backend


def image_to_base64(image):
    """Convert PIL Image to base64 string for API calls"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def generate_vlm_response_gpt4o(messages, save_dir=None, save_name='test'):
    """Generate response using GPT-4o via OpenAI API (uses requests, no openai package needed)"""
    import requests
    import time

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Convert messages to GPT-4o format
    gpt_messages = []
    for msg in messages:
        content_list = []
        for item in msg["content"]:
            if item["type"] == "text":
                content_list.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                # Convert PIL Image to base64
                img = item["image"]
                base64_image = image_to_base64(img)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
        gpt_messages.append({"role": msg["role"], "content": content_list})

    # Prepare API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": gpt_messages,
        "max_tokens": 4096,
        "temperature": 0
    }

    # Make API call with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            output_text = result["choices"][0]["message"]["content"]
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"GPT-4o API call failed after {max_retries} attempts: {e}")
            print(f"   ⚠️  API call failed (attempt {attempt+1}/{max_retries}), retrying...")
            time.sleep(2)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f'{save_name}.txt'), 'w') as file:
            file.write(output_text)

    return output_text


def generate_vlm_response_gemini(messages, save_dir=None, save_name="test",
                                 model="gemini-3-flash-preview", max_retries=5):
    """Generate response using Gemini via Google API (uses requests, no google-generativeai package needed)"""
    from google import genai
    import time
    from google.genai import types

    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    project = os.getenv("GOOGLE_CLOUD_PROJECT", "phsyscene")
    os.environ["GOOGLE_CLOUD_PROJECT"] = project

    client = genai.Client(api_key=api_key)

    # Build a single user turn from your messages (you can extend to multi-turn later)
    parts = []
    for msg in messages:
        for item in msg["content"]:
            if item["type"] == "text":
                parts.append(types.Part.from_text(text=item["text"]))
            elif item["type"] == "image":
                img = item["image"]  # PIL Image
                b64 = image_to_base64(img)  # base64 str, no prefix
                img_bytes = base64.b64decode(b64)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            else:
                raise ValueError(f"Unknown content type: {item['type']}")

    contents = [types.Content(role="user", parts=parts)]

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8192,
                ),
            )
            output_text = resp.text or ""
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Gemini API call failed after {max_retries} attempts: {e}")
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            wait = (60 * (2 ** attempt)) if is_rate_limit else 10
            print(f"⚠️ API call failed (attempt {attempt+1}/{max_retries}), waiting {wait}s... {e}")
            time.sleep(wait)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{save_name}.txt"), "w") as f:
            f.write(output_text)

    return output_text


def generate_vlm_response(messages, save_dir=None, save_name='test', save=True):
    """
    Generate VLM response using the selected backend.

    Args:
        messages: List of message dicts with role and content
        save_dir: Directory to save response
        save_name: Name for saved file
        save: Whether to save (kept for backwards compatibility)

    Returns:
        str: VLM response text
    """
    if _VLM_BACKEND == "gpt4o":
        return generate_vlm_response_gpt4o(messages, save_dir, save_name)
    elif _VLM_BACKEND == "gemini":
        return generate_vlm_response_gemini(messages, save_dir, save_name)
    else:
        raise ValueError(f"Unknown VLM backend: {_VLM_BACKEND}. Choose from: gpt4o, gemini")


def analyze_scene_object_lists(image, save_dir=None, vlm_prompt_file=None):
    """
    Use VLM to identify all salient objects in an image.

    Args:
        image: PIL Image or file path
        save_dir: Directory to save vlm_objects.json
        vlm_prompt_file: Path to txt file containing VLM prompt. If None,
            defaults to ``rest3d/prompts/list_objects.txt`` shipped with
            this package. Relative paths are resolved against ``PROMPTS_DIR``.

    Returns:
        list[str]: Object description prompts, one per object
    """
    from rest3d import PROMPTS_DIR

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    # Resolve prompt file path
    if vlm_prompt_file is None:
        vlm_prompt_file = os.path.join(PROMPTS_DIR, "list_objects.txt")
    elif not os.path.isabs(vlm_prompt_file):
        vlm_prompt_file = os.path.join(PROMPTS_DIR, vlm_prompt_file)
    with open(vlm_prompt_file, "r", encoding="utf-8") as f:
        vlm_prompt = f.read().strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": vlm_prompt},
            ],
        }
    ]

    response = generate_vlm_response(messages, save_dir, 'scene_object_lists')
    print(f"VLM response:\n{response}")

    # Parse: one object per line, strip numbering prefixes
    objects = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\d]+[.\)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)
        line = line.strip()
        if line and len(line) < 100:
            objects.append(line)
    objects.append("the floor")

    return objects

