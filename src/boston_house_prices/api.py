from __future__ import annotations

from fastapi import FastAPI

from boston_house_prices.predict import predict_house_price
from boston_house_prices.schemas import HouseFeatures, PredictionResponse

app = FastAPI(
    title="Boston House Prices API",
    description="Production-style API for house price prediction.",
    version="1.0.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures) -> PredictionResponse:
    payload = features.model_dump(by_alias=True)
    prediction = predict_house_price(payload)
    return PredictionResponse(predicted_price=round(prediction, 2))
