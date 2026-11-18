
import hashlib
import os
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs


VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"
}


def identify_source(source: str) -> dict:
    """
    Identify a YouTube URL or local audio/video file.

    Returns common metadata that the rest of the application
    can use regardless of the input type.
    """

    source = source.strip()

    if not source:
        raise ValueError("Source cannot be empty.")

    # Case 1: YouTube URL
    parsed_url = urlparse(source)

    if parsed_url.scheme in ("http", "https"):
        hostname = (parsed_url.hostname or "").lower()
        video_id = None

        if hostname in ("youtu.be", "www.youtu.be"):
            video_id = parsed_url.path.strip("/").split("/")[0]

        elif hostname in (
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
            "youtube-nocookie.com",
            "www.youtube-nocookie.com",
        ):
            if parsed_url.path == "/watch":
                video_id = parse_qs(parsed_url.query).get("v", [None])[0]

            else:
                match = re.match(
                    r"^/(shorts|embed|live)/([^/?]+)",
                    parsed_url.path
                )

                if match:
                    video_id = match.group(2)

        if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("Please provide a valid YouTube video URL.")

        return {
            "source_id": f"yt_{video_id}",
            "source_type": "youtube",
            "source_location": source,
            "display_name": video_id,
        }

    # Case 2: Local audio/video file
    file_path = Path(os.path.expanduser(source)).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError("The provided path is not a file.")

    extension = file_path.suffix.lower()

    if extension in VIDEO_EXTENSIONS:
        source_type = "local_video"
    elif extension in AUDIO_EXTENSIONS:
        source_type = "local_audio"
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    # Read the file in small blocks instead of loading the
    # entire video/audio file into memory.
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    file_hash = sha256.hexdigest()

    return {
        "source_id": f"local_{file_hash}",
        "source_type": source_type,
        "source_location": str(file_path),
        "display_name": file_path.name,
    }
