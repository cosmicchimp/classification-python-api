from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models
import torch
import io
    

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://classification-website.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


weights = models.MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights)
model.eval()

preprocess = weights.transforms()
categories = weights.meta["categories"]


@app.get("/")
def root():
    return {
        "message": "FastAPI CNN image classification server is running."
    }


@app.post("/classify")
async def classify_image(image: UploadFile = File(...)):
    try:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be an image."
            )

        image_bytes = await image.read()

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Could not process uploaded image."
            )

        input_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_index = torch.max(probabilities, dim=0)

        predicted_label = categories[predicted_index.item()]
        confidence_score = confidence.item()

        return {
            "result": predicted_label,
            "confidence": confidence_score
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )