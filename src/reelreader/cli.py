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
from reelreader.embed import EmbeddingError, ensure_embeddings
from reelreader.ingest import IngestError, IngestResult, ingest_path, ingest_url, is_url
from reelreader.library import Job, Library, LibraryError
from reelreader.pipeline import PipelineError, index_video
from reelreader.providers import ProviderError, build_chat_provider, build_embedder
from reelreader.rag import RagError
from reelreader.rag import ask as ask_video
from reelreader.search import SearchError, search_library

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
    title: Annotated[
        str | None,
        typer.Option("--title", "-t", help="Override the title."),
    ] = None,
    copy: Annotated[
        bool,
        typer.Option("--copy", help="Copy the file into the library folder."),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Add a video, a folder of videos, or a URL. Duplicates are ignored."""
    try:
        settings = Settings.load()
        library = Library.open(settings.library)
        if is_url(source):
            results = [ingest_url(library, source, title=title)]
        else:
            results = ingest_path(library, source, title=title, copy=copy)
    except (ConfigError, IngestError, LibraryError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(json.dumps([_result_dict(result) for result in results]))
        return
    for result in results:
        _report_result(result)
    if any(result.video.status == "no_audio" for result in results if result.created):
        console.print("[yellow]No audio track.[/] This video can only be indexed for visuals.")


@app.command("index")
def index_command(
    video: Annotated[
        str | None,
        typer.Option("--video", "-v", help="Index one video id (default: all)."),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="Whisper model: tiny, base, small, medium, large-v3."),
    ] = "small",
    device: Annotated[
        str,
        typer.Option(help="auto, cpu, or cuda."),
    ] = "auto",
    compute_type: Annotated[
        str,
        typer.Option(help="int8, int8_float16, float16, or float32."),
    ] = "int8",
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Ignore existing outputs and start over."),
    ] = False,
    no_embed: Annotated[
        bool,
        typer.Option("--no-embed", help="Stop after segmenting; skip embeddings."),
    ] = False,
) -> None:
    """Transcode, transcribe, segment, and embed videos into seekable moments."""
    try:
        settings = Settings.load()
        library = Library.open(settings.library)
        videos = [library.get_video(video)] if video else library.list_videos()
    except (ConfigError, LibraryError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    if not videos:
        console.print("No videos yet. Add one with: [bold]reelreader add FILE[/]")
        return

    failures = 0
    for item in videos:
        try:
            result = index_video(
                library,
                item,
                model=model,
                device=device,
                compute_type=compute_type,
                force=rebuild,
                log=lambda message: console.print(f"[dim]{message}[/]"),
            )
        except PipelineError as exc:
            console.print(f"[red]{item.title}:[/] {exc}")
            failures += 1
        else:
            changed = ", ".join(result.ran) or "up to date"
            console.print(f"[green]{item.title}[/] {result.segment_count} segments ({changed})")
    if failures:
        raise typer.Exit(code=1)
    if no_embed:
        return

    ready = [library.get_video(item.id) for item in videos]
    ready = [item for item in ready if item.status == "ready"]
    if not ready:
        return
    try:
        embedder = build_embedder(settings)
        summary = ensure_embeddings(
            library,
            ready,
            embedder,
            force=rebuild,
            log=lambda message: console.print(f"[dim]{message}[/]"),
        )
    except (EmbeddingError, ProviderError) as exc:
        console.print(f"[red]embedding failed:[/] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[dim]embeddings: {len(summary.embedded)} updated, {len(summary.skipped)} up to date[/]"
    )


@app.command("list")
def list_command(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List videos in the library."""
    try:
        settings = Settings.load()
        library = Library.open(settings.library)
    except (ConfigError, LibraryError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    videos = library.list_videos()
    if as_json:
        payload = []
        for video in videos:
            job = library.latest_job(video.id)
            payload.append(
                {
                    "id": video.id,
                    "title": video.title,
                    "path": str(video.path),
                    "duration": video.duration,
                    "status": "missing" if not video.exists else video.status,
                    "has_audio": video.has_audio,
                    "added_at": video.added_at,
                    "job": None if job is None else _job_dict(job),
                }
            )
        console.print_json(json.dumps(payload))
        return

    if not videos:
        console.print("No videos yet. Add one with: [bold]reelreader add FILE[/]")
        return

    table = Table(show_lines=False)
    table.add_column("Title", style="bold")
    table.add_column("Length", justify="right")
    table.add_column("Status")
    table.add_column("Job")
    table.add_column("Id", style="dim")
    for video in videos:
        state = "missing" if not video.exists else video.status
        job = library.latest_job(video.id)
        job_label = "" if job is None else f"{job.stage} {job.state} {job.progress:.0%}"
        table.add_row(
            video.title,
            video.duration_label,
            _status_cell(state),
            job_label,
            video.id,
        )
    console.print(table)
    console.print(f"[dim]{len(videos)} video(s) in {library.root}[/]")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for.")],
    view: Annotated[str, typer.Option(help="Retrieval view (text until M3).")] = "text",
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many moments.")] = 8,
    mode: Annotated[str, typer.Option(help="hybrid, dense, or bm25.")] = "hybrid",
    video: Annotated[
        str | None,
        typer.Option("--video", "-v", help="Restrict to one video id."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Find moments without calling a chat model."""
    if view != "text":
        console.print("[yellow]Only the text view is indexed until M3.[/]")
        raise typer.Exit(code=2)
    try:
        settings = Settings.load()
        library = Library.open(settings.library)
        embedder = build_embedder(settings)
        hits = search_library(
            library,
            query,
            embedder=embedder,
            limit=limit,
            mode=mode,
            video_id=video,
        )
    except (ConfigError, LibraryError, ProviderError, EmbeddingError, SearchError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(json.dumps([hit.to_dict() for hit in hits]))
        return
    if not hits:
        console.print("No moments matched.")
        return
    table = Table(show_lines=False)
    table.add_column("Time", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Video")
    table.add_column("Moment")
    for hit in hits:
        table.add_row(hit.label, f"{hit.score:.3f}", hit.video_title, _snippet(hit.text, 90))
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question about the video.")],
    video: Annotated[
        str | None,
        typer.Option("--video", "-v", help="Restrict to one video id."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Moments to consider.")] = 8,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Ask a question and get a grounded answer with timestamps."""
    try:
        settings = Settings.load()
        library = Library.open(settings.library)
        answer = ask_video(library, settings, question, limit=limit, video_id=video)
    except (
        ConfigError,
        LibraryError,
        ProviderError,
        EmbeddingError,
        RagError,
        SearchError,
    ) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(json.dumps(answer.to_dict()))
        return
    console.print(answer.text, style="yellow" if answer.abstained else "")
    if not answer.citations:
        return
    console.print()
    table = Table(title="Sources", show_lines=False)
    table.add_column("Time", justify="right")
    table.add_column("Video")
    table.add_column("Moment")
    for citation in answer.citations:
        table.add_row(citation.label, citation.video_title, citation.snippet)
    console.print(table)


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


def _snippet(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _result_dict(result: IngestResult) -> dict[str, object]:
    return {
        "id": result.video.id,
        "title": result.video.title,
        "path": str(result.video.path),
        "duration": result.video.duration,
        "status": result.video.status,
        "has_audio": result.video.has_audio,
        "created": result.created,
        "skipped": result.skipped,
    }


def _job_dict(job: Job) -> dict[str, object]:
    return {
        "id": job.id,
        "stage": job.stage,
        "state": job.state,
        "progress": job.progress,
        "message": job.message,
        "error": job.error,
    }


def _report_result(result: IngestResult) -> None:
    video = result.video
    if result.created:
        console.print(f"[green]Added[/] {video.title} ({video.duration_label}, {video.id})")
    else:
        reason = result.skipped or "already in library"
        console.print(f"[yellow]Skipped[/] {video.title}: {reason}")


def _status_cell(status: str) -> str:
    if status == "ready":
        return f"[green]{status}[/]"
    if status == "missing":
        return f"[yellow]{status}[/]"
    if status == "failed":
        return f"[red]{status}[/]"
    if status == "no_audio":
        return f"[magenta]{status}[/]"
    if status == "interrupted":
        return f"[yellow]{status}[/]"
    return status


def _planned(story: int, action: str) -> NoReturn:
    console.print(f"[yellow]Not in v0.0.1:[/] {action}")
    console.print(f"Planned in docs/STORIES.md, story {story}.")
    raise typer.Exit(code=2)


def _python_version() -> str:
    import platform

    return f"{platform.python_version()} ({platform.system()} {platform.machine()})"


if __name__ == "__main__":
    app()
