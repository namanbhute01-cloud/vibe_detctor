# auto_downloader.py
import os
import subprocess
from pathlib import Path

# --- CONFIG ---
OUT_ROOT = Path("OfflinePlayback")
GROUPS = {
    "Kids": [
        "https://www.youtube.com/watch?v=5qap5aO4i9A",
        "https://www.youtube.com/watch?v=1ZYbU82GVz4",
        "https://www.youtube.com/watch?v=2OEL4P1Rz04",
    ],
    "Teens": [
        "https://www.youtube.com/watch?v=kgx4WGK0oNU",
        "https://www.youtube.com/watch?v=7NOSDKb0HlU",
        "https://www.youtube.com/watch?v=jfKfPfyJRdk",
    ],
    "YoungAdults": [
        "https://www.youtube.com/watch?v=DWcJFNfaw9c",
        "https://www.youtube.com/watch?v=Eo-KmOd3i7s",
        "https://www.youtube.com/watch?v=MGk_Ba0KJ7U",
    ],
    "Adults": [
        "https://www.youtube.com/watch?v=2H5uWRjFsGc",
        "https://www.youtube.com/watch?v=hlWiI4xVXKY",
        "https://www.youtube.com/watch?v=-8NSoYMzibs",
    ],
    "Seniors": [
        "https://www.youtube.com/watch?v=Er6j8GzX8iQ",
        "https://www.youtube.com/watch?v=rqzkn-jX-JU",
        "https://www.youtube.com/watch?v=5KqJGtLC4g8",
    ],
    "Elderly": [
        "https://www.youtube.com/watch?v=8omw-Z2t1_s",
        "https://www.youtube.com/watch?v=NPjIuQ7unGI",
        "https://www.youtube.com/watch?v=2XKj6rst0V0",
    ],
}
# Path to yt-dlp binary/exe. If yt-dlp is on PATH, leave as "yt-dlp".
YTDLP_CMD = "yt-dlp"

# ytdlp options - change as desired
YTDLP_COMMON_OPTS = [
    "-f", "bestaudio",
    "--extract-audio",
    "--audio-format", "mp3",
    "--no-overwrites",
    "--yes-playlist",
    "--ignore-errors",
    "--no-post-overwrites",
]

# Create directories
OUT_ROOT.mkdir(parents=True, exist_ok=True)
for grp in GROUPS:
    (OUT_ROOT / grp).mkdir(parents=True, exist_ok=True)

def run_download(group, url):
    outdir = OUT_ROOT / group
    # Output template: use %(title)s and %(ext)s (yt-dlp placeholders)
    out_template = str(outdir / "%(title)s.%(ext)s")
    cmd = [YTDLP_CMD] + YTDLP_COMMON_OPTS + ["-o", out_template, url]
    print("\n[CMD] " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        print(proc.stdout)
        if proc.returncode == 0:
            print(f"[OK] Download attempt finished for group={group}")
        else:
            print(f"[WARN] yt-dlp returned code {proc.returncode} for url: {url}")
    except FileNotFoundError:
        print("[ERROR] yt-dlp not found. Install with 'pip install yt-dlp' or place yt-dlp.exe on PATH.")
        raise
    except Exception as e:
        print("[ERROR] Exception while running yt-dlp:", e)

def main():
    print("[INFO] Starting auto-downloader")
    # quick sanity check: yt-dlp exists?
    try:
        r = subprocess.run([YTDLP_CMD, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if r.returncode != 0:
            print("[WARN] yt-dlp may not be available or returned non-zero exit code. Output:")
            print(r.stdout, r.stderr)
        else:
            print(f"[INFO] yt-dlp version: {r.stdout.strip()}")
    except FileNotFoundError:
        print("[ERROR] yt-dlp not found. Install with 'pip install yt-dlp' or put yt-dlp.exe next to this script.")
        return

    for group, urls in GROUPS.items():
        print(f"\n=== Downloading group: {group} ({len(urls)} links) ===")
        for url in urls:
            if not url or url.strip() == "":
                continue
            try:
                run_download(group, url)
            except Exception as e:
                print("[ERROR] Download failed for:", url, e)
    print("\n[INFO] All requested downloads finished. Check OfflinePlayback/ folders.")

if __name__ == "__main__":
    main()
