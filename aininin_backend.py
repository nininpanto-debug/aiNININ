import os
import io
import json
import base64
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
import fal_client

app = FastAPI(title="aiNININ Real AI Generation Backend")

# Allow the GitHub Pages app to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAL_KEY = os.getenv("FAL_KEY")
MODEL = "fal-ai/ip-adapter-face-id"


def add_watermark(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)

    text = "aiNININ"
    margin = max(18, image.width // 50)
    font_size = max(18, image.width // 55)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = image.width - tw - margin
    y = image.height - th - margin

    # Subtle watermark: visible but not distracting.
    draw.text((x + 2, y + 2), text, fill=(0, 0, 0, 85), font=font)
    draw.text((x, y), text, fill=(255, 255, 255, 120), font=font)

    return image.convert("RGB")


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=94)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "aiNININ",
        "message": "aiNININ Real AI Generation Backend is online."
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
async def generate(
    face_image: UploadFile = File(...),
    template_name: str = Form(""),
    identity_strength: float = Form(0.90),
    reference_match: float = Form(0.85),
    face_position: str = Form(""),
    custom_prompt: str = Form(""),
):
    if not FAL_KEY:
        raise HTTPException(
            status_code=500,
            detail="FAL_KEY is not configured on the server."
        )

    raw = await face_image.read()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="No face/reference image was uploaded."
        )

    # Validate the uploaded image before sending it to fal.
    try:
        Image.open(io.BytesIO(raw)).verify()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {exc}"
        )

    parsed_position: Optional[dict] = None
    if face_position:
        try:
            parsed_position = json.loads(face_position)
        except Exception:
            parsed_position = None

    prompt = custom_prompt.strip() or template_name.strip()
    if not prompt:
        prompt = (
            "A highly realistic photograph of the same person "
            "in a natural cinematic scene."
        )

    full_prompt = (
        f"{prompt}. "
        "Preserve the person's facial identity and recognizable facial structure. "
        "Photorealistic real-camera appearance, natural skin texture, "
        "realistic lighting, correct anatomy, believable perspective and depth. "
        "The same person must remain recognizable."
    )

    try:
        # fal's official Python client supports uploading raw bytes.
        reference_url = fal_client.upload(
            raw,
            "image/jpeg",
            face_image.filename or "reference.jpg"
        )

        # IMPORTANT: the IP Adapter Face ID model expects face_image_url.
        # It does not expose an identity_strength parameter in its current schema.
        arguments = {
            "model_type": "1_5-v1",
            "prompt": full_prompt,
            "face_image_url": reference_url,
            "negative_prompt": (
                "blurry, low resolution, bad anatomy, distorted face, "
                "different person, duplicate person, cartoon, illustration, "
                "plastic skin, deformed hands, extra fingers"
            ),
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "num_samples": 4,
            "width": 768,
            "height": 1024,
            "face_id_det_size": 640,
        }

        result = fal_client.run(MODEL, arguments=arguments)

        # Current fal schema returns one image object: {"image": {"url": ...}}.
        image_url: Optional[str] = None

        if isinstance(result, dict):
            image_obj = result.get("image")
            if isinstance(image_obj, dict):
                image_url = image_obj.get("url")
            elif isinstance(image_obj, str):
                image_url = image_obj

            # Tolerate older/list-shaped responses too.
            if not image_url:
                images = result.get("images")
                if isinstance(images, list) and images:
                    first = images[0]
                    if isinstance(first, dict):
                        image_url = first.get("url")
                    elif isinstance(first, str):
                        image_url = first

        if not image_url:
            keys = list(result.keys()) if isinstance(result, dict) else []
            raise RuntimeError(
                f"Fal returned no image URL. Response keys: {keys}"
            )

        # Fetch the generated image and add the aiNININ watermark.
        generated_response = requests.get(image_url, timeout=60)
        generated_response.raise_for_status()

        generated = Image.open(
            io.BytesIO(generated_response.content)
        ).convert("RGB")

        generated = add_watermark(generated)
        watermarked_data_url = image_to_data_url(generated)

        return JSONResponse({
            "ok": True,
            "image_url": watermarked_data_url,
            "source_image_url": image_url,
            "template_name": template_name,
            "identity_strength": float(identity_strength),
            "reference_match": float(reference_match),
            "face_position": parsed_position,
            "watermark": "aiNININ",
            "model": MODEL,
        })

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI generation failed: {exc}"
        )
