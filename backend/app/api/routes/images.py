import os
import shutil

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path

from app.core.config import settings
from editImage.object_store import get_images_path
from editImage.rac_main import retake_image

UPLOAD_DIR = Path(settings.IMAGE_DB(), "source")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/image", tags=["image"])


@router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Clean directory for new upload
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    shutil.rmtree(f"{settings.IMAGE_DB()}/output", ignore_errors=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    # Create file path
    file_path = UPLOAD_DIR / file.filename

    try:
        # Save file to disk
        with file_path.open("wb") as buffer:
            buffer.write(await file.read())

        image_paths = get_images_path(UPLOAD_DIR)
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            name, extnshn = filename.split('.')
            simple_file_path = os.path.join(f"{settings.IMAGE_DB()}/output", f"{name}_simple.{extnshn}")
            smile_file_path = os.path.join(f"{settings.IMAGE_DB()}/output", f"{name}_smile.{extnshn}")
            close_mouth_file_path = os.path.join(f"{settings.IMAGE_DB()}/output", f"{name}_close_mouth.{extnshn}")
            retake_image(image_path, simple_file_path, -0.3)
            retake_image(image_path, close_mouth_file_path, -0.1, lip_false=True)
            retake_image(image_path, smile_file_path, 0.9)
            # Get the filename and extension from the image path
        print(f"{settings.IMAGE_DB()}/output")

        return JSONResponse({"message": "Image uploaded successfully", "filename": file.filename,
                             "satatus": F"Result is in {settings.IMAGE_DB()}/output"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")
