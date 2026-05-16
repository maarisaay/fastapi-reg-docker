import uvicorn
from fastapi import FastAPI, HTTPException, Form
from pathlib import Path
from typing import Annotated
from model.point import Point
from libs.model import predict, train

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath("ml_models")
DATA_DIR = Path(BASE_DIR).joinpath("data")

app = FastAPI()

@app.get("/", tags=["intro"])
def index():
    return {"message": "Linear Regression ML"}

@app.post("/model/point", tags=["data"], response_model=Point, status_code=200)
async def point(x: Annotated[int, Form()], y: Annotated[float, Form()]):
    return Point(x=x, y=y)