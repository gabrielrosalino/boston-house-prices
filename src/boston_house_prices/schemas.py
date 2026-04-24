from __future__ import annotations

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    crim: float = Field(..., alias="CRIM", description="Per capita crime rate by town.")
    zn: float = Field(..., alias="ZN", description="Proportion of residential land zoned for large lots.")
    indus: float = Field(..., alias="INDUS", description="Proportion of non-retail business acres.")
    chas: float = Field(..., alias="CHAS", description="Charles River dummy variable.")
    nox: float = Field(..., alias="NOX", description="Nitric oxides concentration.")
    rm: float = Field(..., alias="RM", description="Average number of rooms per dwelling.")
    age: float = Field(..., alias="AGE", description="Proportion of owner-occupied units built before 1940.")
    dis: float = Field(..., alias="DIS", description="Weighted distances to employment centers.")
    rad: float = Field(..., alias="RAD", description="Accessibility index to radial highways.")
    tax: float = Field(..., alias="TAX", description="Property-tax rate.")
    ptratio: float = Field(..., alias="PTRATIO", description="Pupil-teacher ratio by town.")
    b: float = Field(..., alias="B", description="Demographic transformation from original dataset.")
    lstat: float = Field(..., alias="LSTAT", description="Lower status population percentage.")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    predicted_price: float
    unit: str = "USD thousands"
