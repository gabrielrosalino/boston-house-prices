from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from boston_house_prices.predict import predict_house_price
from boston_house_prices.train import train_model

console = Console()
train_app = typer.Typer(help="Train the Boston house price prediction model.")
predict_app = typer.Typer(help="Generate predictions using a trained model.")


@train_app.callback(invoke_without_command=True)
def train(
    dataset_path: Annotated[Path | None, typer.Option(help="Optional local CSV dataset path.")] = None,
) -> None:
    metrics = train_model(dataset_path=dataset_path)

    table = Table(title="Training metrics")
    table.add_column("Metric")
    table.add_column("Value")

    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))

    console.print(table)


@predict_app.callback(invoke_without_command=True)
def predict(
    payload: Annotated[str, typer.Option(help="JSON string with the 13 model features.")],
) -> None:
    features = json.loads(payload)
    prediction = predict_house_price(features)
    console.print({"predicted_price": round(prediction, 2), "unit": "USD thousands"})
