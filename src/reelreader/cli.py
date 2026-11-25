"""The ``reelreader`` command line.

v0.0.1 ships the configuration, provider, and diagnostic surface. The indexing
and retrieval commands are registered but point at the stories that fill them
in, so ``--help`` and ``doctor`` describe the product from day one.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from reelreader import __version__
from reelreader.config import CONFIG_NAME, ConfigError, Settings
from reelreader.providers import ProviderError, build_chat_provider

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Chat with recorded lectures and videos. Local-first, seekable citations.",
)
console = Console()

CONFIG_TEMPLATE = """\
# Reelreader configuration. Keys are never stored here.
library = "~/Reelreader"
tier = "text"  # text | visual | full

[chat]
# "openai" covers any OpenAI-compatible endpoint, including Ollama:
#   base_url = "http://localhost:11434/v1"  model = "llama3.2"
provider = "openai"
base_url = "https://api.openai.com/v1"
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"

[embed]
# Anthropic is not an option: it has no embeddings API.
provider = "local"
model = "BAAI/bge-small-en-v1.5"
dim = 384
"""


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"reelreader {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Reelreader: ask a video what it said, and jump to the moment it said it."""


@app.command()
def init(
    directory: Annotated[
        Path,
        typer.Option("--directory", "-d", help="Where to write the config file."),
    ] = Path(),
    force: Annotated[bool, typer.Option(help="Overwrite an existing config.")] = False,
) -> None:
    """Write a starter reelreader.toml and create the library folder."""
    target = (directory or Path()).resolve() / CONFIG_NAME
    if target.exists() and not force:
        console.print(f"[yellow]{target} already exists.[/] Use --force to overwrite.")
        raise typer.Exit(code=1)
    target.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    settings = Settings.load()
    settings.library.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Wrote[/] {target}")
    console.print(f"[green]Library[/] {settings.library}")


@app.command()
def config() -> None:
    """Show the resolved settings, with secrets redacted."""
    try:
        settings = Settings.load()
    except ConfigError as exc:
        console.print(f"[red]Config error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print_json(json.dumps(settings.as_dict(), indent=2))


@app.command()
def doctor() -> None:
    """Check the machine, configuration, and optional pieces."""
    try:
        settings = Settings.load()
    except ConfigError as exc:
        console.print(f"[red]Config error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Reelreader doctor", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Value")
    table.add_column("Status")

    def row(check: str, value: str, ok: bool | None) -> None:
        if ok is None:
            status = "[dim]info[/]"
        elif ok:
            status = "[green]ok[/]"
        else:
            status = "[yellow]missing[/]"
        table.add_row(check, value, status)

    row("Python", _python_version(), True)

    library_exists = settings.library.is_dir()
    row("Library", str(settings.library), library_exists)

    config_label = str(settings.config_path) if settings.config_path else "defaults"
    row("Config", config_label, settings.config_path is not None)

    chat_key = settings.chat.api_key() is not None
    row(
        "Chat",
        f"{settings.chat.provider} / {settings.chat.model}",
        True if settings.chat.provider == "openai" and not chat_key else chat_key,
    )
    key_note = "present" if chat_key else f"set {settings.chat.api_key_env}"
    row("Chat key", key_note, chat_key or settings.chat.provider == "openai")

    row(
        "Embeddings",
        f"{settings.embed.provider} / {settings.embed.model} ({settings.embed.dim}d)",
        True,
    )
    row("Retrieval tier", settings.tier, True)
    row("ffmpeg", shutil.which("ffmpeg") or "not on PATH", shutil.which("ffmpeg") is not None)

    for label, module, extra in (
        ("fastembed", "fastembed", "local"),
        ("faster-whisper", "faster_whisper", "asr"),
        ("scenedetect", "scenedetect", "scene"),
        ("fastapi", "fastapi", "serve"),
        ("sqlite-vec", "sqlite_vec", "index"),
    ):
        present = importlib.util.find_spec(module) is not None
        row(label, "installed" if present else f"pip install 'reelreader[{extra}]'", present)

    console.print(table)
    if settings.tier != "text":
        console.print(
            "[dim]Non-text tiers need a multimodal embedding model and, for "
            "'full', a GPU. See docs/SPEC.md: Retrieval tiers.[/]"
        )


@app.command()
def models() -> None:
    """List models the configured chat provider offers."""
    try:
        settings = Settings.load()
        provider = build_chat_provider(settings)
        with console.status(f"Listing models from {provider.name}..."):
            names = provider.list_models()
    except (ConfigError, ProviderError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    if not names:
        console.print("[yellow]The provider returned no models.[/]")
        return
    for name in names:
        console.print(name)


@app.command()
def add(
    source: Annotated[str, typer.Argument(help="File, folder, or URL to ingest.")],
) -> None:
    """Add a video to the library and queue the pipeline."""
    _planned(3, f"ingest {source!r}")


@app.command("index")
def index_command(
    view: Annotated[
        str,
        typer.Option(help="Embedding view: text, frame, video, audio, fused, or all."),
    ] = "all",
) -> None:
    """Transcribe, segment, and embed the library."""
    _planned(4, f"index view={view}")


@app.command("list")
def list_command() -> None:
    """List videos in the library."""
    _planned(5, "list the library")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    view: Annotated[str, typer.Option(help="Retrieval view.")] = "all",
) -> None:
    """Find moments, without calling a chat model."""
    _planned(6, f"search {query!r} view={view}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question about the video.")],
    video: Annotated[str | None, typer.Option(help="Restrict to one video id.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Ask a question and get a grounded answer with timestamps."""
    _planned(7, f"ask {question!r} video={video} json={as_json}")


@app.command()
def export(
    video_id: Annotated[str, typer.Argument(help="Video id.")],
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="md, srt, vtt, or anki."),
    ] = "md",
) -> None:
    """Export a transcript, study guide, or flashcard deck."""
    _planned(10, f"export {video_id} as {fmt}")


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="0 picks a free port.")] = 0,
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
) -> None:
    """Run the local HTTP API the desktop app talks to."""
    _planned(11, f"serve on {host}:{port}")


@app.command("eval")
def eval_command(
    dataset: Annotated[
        Path,
        typer.Option(help="JSONL of questions with expected timestamps."),
    ] = Path("data/eval/manual.jsonl"),
) -> None:
    """Measure recall@k for the configured tier and views."""
    _planned(12, f"eval {dataset}")


def _planned(story: int, action: str) -> NoReturn:
    console.print(f"[yellow]Not in v0.0.1:[/] {action}")
    console.print(f"Planned in docs/STORIES.md, story {story}.")
    raise typer.Exit(code=2)


def _python_version() -> str:
    import platform

    return f"{platform.python_version()} ({platform.system()} {platform.machine()})"


if __name__ == "__main__":
    app()
