"""
Multimodal: helpers for building OpenAI-format message content from
text, image files/URLs, and audio files.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

# Content item type alias
ContentItem = dict[str, Any]

logger = logging.getLogger(__name__)


def text_item(text: str) -> ContentItem:
    return {"type": "text", "text": text}


def image_url_item(url: str, detail: str = "auto") -> ContentItem:
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


def image_file_item(path: str | Path, detail: str = "auto") -> ContentItem:
    """Encode a local image file as a base64 data URL."""
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return image_url_item(f"data:{mime};base64,{data}", detail=detail)


def audio_file_item(path: str | Path) -> ContentItem:
    """Encode a local audio file for OpenAI audio-capable models."""
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    fmt = _audio_format(mime or "")
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}


def _audio_format(mime: str) -> str:
    mapping = {
        "audio/mpeg": "mp3",
        "audio/mp4": "mp4",
        "audio/wav": "wav",
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/flac": "flac",
    }
    return mapping.get(mime, "mp3")


def build_user_message(
    text: str | None = None,
    images: list[str | Path] | None = None,   # file paths or URLs
    audio: list[str | Path] | None = None,    # file paths
    extra_items: list[ContentItem] | None = None,
) -> dict[str, Any]:
    """
    Build an OpenAI-format user message supporting text + images + audio.

    If only text is provided the message uses the simple string format;
    otherwise a list of content items is used.
    """
    items: list[ContentItem] = []

    if text:
        logger.debug("build_user_message – text item, %d char(s)", len(text))
        items.append(text_item(text))

    for img in images or []:
        s = str(img)
        if s.startswith(("http://", "https://", "data:")):
            logger.debug("build_user_message – image URL: %s", s[:80])
            items.append(image_url_item(s))
        else:
            size = Path(s).stat().st_size if Path(s).exists() else -1
            logger.debug("build_user_message – image file: %s (%d bytes)", s, size)
            items.append(image_file_item(img))

    for aud in audio or []:
        size = Path(str(aud)).stat().st_size if Path(str(aud)).exists() else -1
        logger.debug("build_user_message – audio file: %s (%d bytes)", aud, size)
        items.append(audio_file_item(aud))

    for item in extra_items or []:
        items.append(item)

    if not items:
        raise ValueError("build_user_message: at least one content item required")

    logger.debug(
        "build_user_message – %d item(s), format=%s",
        len(items),
        "plain_text" if (len(items) == 1 and items[0]["type"] == "text") else "content_list",
    )

    # Optimise: if there is only a single text item use plain string format
    if len(items) == 1 and items[0]["type"] == "text":
        return {"role": "user", "content": items[0]["text"]}

    return {"role": "user", "content": items}
