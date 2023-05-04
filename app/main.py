from fastapi import FastAPI, File
from PIL import Image
from fastapi.responses import FileResponse
from helper.execute import cartoonize, stylize
from io import BytesIO
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve(strict=True).parent
app = FastAPI()

def read_imagefile(file) -> Image.Image:
    return Image.open(BytesIO(file))
    
def listImageFile():
    files = os.listdir(f"{BASE_DIR}/result")
    files = [f for f in files]
    return {"files": files}

@app.get("/get-images")
async def getImage():
    return listImageFile()

@app.get("/del-images")
async def delImage():
    [os.remove(f"{BASE_DIR}/result/{file}") for file in os.listdir(f"{BASE_DIR}/result")]
    return {"fiels": listImageFile()}

@app.get("/")
async def test():
    return {"response": "working"}


@app.post("/cartoonize")
async def cartoon(imageFile: bytes = File(...)):
    input_image = read_imagefile(imageFile)
    return cartoonize(input_image)

@app.post("/apply-style")
async def style(imageFile: bytes = File(...)):
    input_image = read_imagefile(imageFile)
    return stylize(input_image)

@app.get("/image/{image_name}")
async def get_image_url(image_name: str):
    image_path = os.path.join("result", image_name)
    return FileResponse(image_path)

