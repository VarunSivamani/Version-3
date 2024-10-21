from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil

app = FastAPI()

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="../images"), name="images")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    file_size = 0
    try:
        # Create a temporary file to store the uploaded content
        with Path("temp_file").open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get the file size
        file_size = Path("temp_file").stat().st_size
    finally:
        # Clean up the temporary file
        if Path("temp_file").exists():
            Path("temp_file").unlink()
    
    return JSONResponse({
        "name": file.filename,
        "size": f"{file_size} bytes",
        "type": file.content_type
    })

@app.get("/check-image/{animal}")
async def check_image(animal: str):
    image_path = Path(f"../images/{animal}.jpg")
    if image_path.exists():
        return JSONResponse({"exists": True})
    else:
        return JSONResponse({"exists": False})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
