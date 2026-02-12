"""CLI entry point for Transskribo."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="transskribo",
    help="Batch audio transcription and speaker diarization using WhisperX.",
)


@app.command()
def run(
    config: str = typer.Option(..., "--config", help="Path to TOML config file"),
) -> None:
    """Run batch transcription processing."""
    typer.echo(f"Config: {config}")


if __name__ == "__main__":
    app()
