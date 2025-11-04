import os
import io
import warnings

import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import uvicorn

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Create output folder
os.makedirs("background", exist_ok=True)

# FastAPI app
app = FastAPI(title="Simple Background Remover")

# --- CORS (allow your frontend to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# --- Serve frontend static files from ./frontend at /ui
# Put your index.html in frontend/index.html
if os.path.isdir("frontend"):
    app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")

# Optional: redirect root to /ui for convenience
@app.get("/", include_in_schema=False)
def root():
    # if frontend exists, redirect to the UI; otherwise, simple message
    if os.path.isdir("frontend"):
        return RedirectResponse(url="/ui/")
    return {"message": "Simple Background Remover API. Visit /docs for API docs."}

# Load model once
torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet",
    trust_remote_code=True
).to(device)

# Image transform
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def remove_background(image: Image.Image):
    """Core logic to produce RGBA image with alpha mask."""
    img = image.convert("RGB")
    image_size = img.size
    input_image = transform_image(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_image)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)

    img.putalpha(mask)
    return img  # PIL image with alpha

@app.post("/remove-background")
async def remove_bg(file: UploadFile = File(...)):
    """
    Upload an image -> get the same image with transparent background (PNG).
    Also saves the processed image into the 'background/' folder.
    """
    try:
        input_image = Image.open(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        result_img = remove_background(input_image)

        # Get filename for saving
        filename = os.path.splitext(file.filename)[0]
        output_path = os.path.join("background", f"{filename}.png")

        # Save processed image
        result_img.save(output_path, format="PNG")

        # Return image as response also
        img_bytes = io.BytesIO()
        result_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

if __name__ == "__main__":
    # FIXED: Only one closing parenthesis
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=7862)
