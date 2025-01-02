from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path

from app.core.config import settings

UPLOAD_DIR = Path(settings.IMAGE_DB())
UPLOAD_DIR.mkdir(exist_ok=True)

router = APIRouter(prefix="/image", tags=["image"])

@router.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    # Create file path
    file_path = UPLOAD_DIR / file.filename

    try:
        # Save file to disk
        with file_path.open("wb") as buffer:
            buffer.write(await file.read())

        return JSONResponse({"message": "Image uploaded successfully", "filename": file.filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload image: {str(e)}")
