from soulanalyzer.backend.MLData import MLData
from soulanalyzer.backend.SoulAnalyzer import SoulAnalyzer

import importlib.metadata

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def version() -> None:
    print(importlib.metadata.version("soulanalyzer"))


@app.command()
def test() -> None:
    print("test successful")

@app.command()
def train(csv: str = "data/Blabla.csv" ) -> None:
    print(f"Training model with {csv} data")
    #dataset = MLData(csv=csv)
    #analyzer = SoulAnalyzer(dataset)

