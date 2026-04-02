from typing import Annotated
from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import keras
import pickle

with open("models/scaler_x.pickle", "rb") as f:
    x_scaler = pickle.load(f)
with open("models/scaler_y.pickle", "rb") as f:
    y_scaler = pickle.load(f)
model = keras.saving.load_model("models/speed_prediction.keras")


service = FastAPI(
    title="Generator APIary",
    description="""This is the core of the whole architecture where model blueprints \
                    (algorithms) resides. It exposes two endpoints to start a new training or inference \
                    process.
                    """,
)


class Request(BaseModel):
    quality: Annotated[float, Field(ge=0, le=1)]

class Response(BaseModel):
    speed: Annotated[int, Field(ge=1, le=20)]


@service.post(
    "/infer",
    responses={400: {"model": str}, 500: {"model": str}},
    response_class=Response,
    description="""Makes inference given a quality score
    """,
)
async def infer_data(request: Request):
    x = x_scaler.transform([[request.quality]])
    y = model.predict(x)
    y = y_scaler.inverse_transform(y)
    return Response(speed=int(y[0][0]))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(service, host="0.0.0.0", port=8010)
