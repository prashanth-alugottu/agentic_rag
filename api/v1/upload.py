from fastapi import APIRouter, UploadFile, File
from services.uploading_service import upload_file

router = APIRouter(prefix="/api/v1")

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_bytes = await file.read()
    result = upload_file(file_bytes, file.filename)
    return {
        "message": "File uploaded successfully",
        "details": result
    }
