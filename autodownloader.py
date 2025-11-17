import os
import subprocess
import time

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
BASE_DIR = os.path.join(os.getcwd(), "OfflinePlayback")

AGE_GROUPS = {
    "kids": [
        "https://www.youtube.com/playlist?list=PLc_C0GcxiJ8C5lS0ZkQpT2r8j04YfGLm_",
        "https://www.youtube.com/playlist?list=PLXd8g0k3uMZA4j7kGEXUX-uLLq5yRvCN-",
        "https://www.youtube.com/playlist?list=PLRjXY9OdnZRGlS0iWjC4w9hRuoF7waRM9",
        "https://www.youtube.com/playlist?list=PLh8a85xAp36J5Lsp0z5sPVXrj342XoPKL",
        "https://www.youtube.com/playlist?list=PLJm7MT9WczdvYVtwZUp3_LSSPFcAckS4i"
    ],

    "teens": [
        "https://www.youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmhvj7fax8fdxoj",
        "https://www.youtube.com/playlist?list=PLNxfttvZ2llG-jxwDyZaR0RoMPygfE6DS",
        "https://www.youtube.com/playlist?list=PLMC9KNkIncKtZfTQud6cA9zFu8Y5eoHz9",
        "https://www.youtube.com/playlist?list=PLmU8B4gZ41icn_5R16vuMJnqd7qV3CyMf",
        "https://www.youtube.com/playlist?list=PLIM7V63xmFvVS9Kf-Y6z3-b0JVpaz6RtL"
    ],

    "youths": [
        "https://www.youtube.com/playlist?list=PLFgquLnL59alCl_2TQvOiD5Vgm1hCaGSI",
        "https://www.youtube.com/playlist?list=PLDcnymzs18LVXfO_x0Ei0R24qDbVtyy66",
        "https://www.youtube.com/playlist?list=PLLMA7Sh3AIBrIIP_hYQB6S1YXWgE4jT_U",
        "https://www.youtube.com/playlist?list=PLS_oEMUyvA729x1NqH3Q2RBYeme2LzPFB",
        "https://www.youtube.com/playlist?list=PL4fGSI1pDJn6l5bBC0ynjGduV1JN89B6P"
    ],

    "adults": [
        "https://www.youtube.com/playlist?list=PLzAU6R0gtKdh7AWJ9w6-jGq-Lu9Y27Aor",
        "https://www.youtube.com/playlist?list=PL9bw4S5ePsEEb-NnyVnYueuUxMiVjC3wF",
        "https://www.youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmhvj7fax8fdxoj",
        "https://www.youtube.com/playlist?list=PLmU8B4gZ41ic6eKsrDMAYA9Yg3gCBXlI3",
        "https://www.youtube.com/playlist?list=PLS6o7gW8gZz1rB6g7YVOlJawVSHT6r30Y"
    ],

    "seniors": [
        "https://www.youtube.com/playlist?list=PLQ-dFfG9wP_lFy1gI2zH6RcVYgicuQeY3",
        "https://www.youtube.com/playlist?list=PL-9E0pLneY2vC68JrjYNewc19hXtOD87o",
        "https://www.youtube.com/playlist?list=PL3oW2tjiIxvSXoQGcW6HGz6E1YtZq9VHt",
        "https://www.youtube.com/playlist?list=PLV9Wb1iH5_BS1_Qyjp-1rssZCvGSqtEtD",
        "https://www.youtube.com/playlist?list=PLJNbijGz6wslZxvc9yZICEDypm9dNfTo1"
    ],

    "elderly": [
        "https://www.youtube.com/playlist?list=PLcFtoFQIj3A3W0xBW3ZrzXnFz3I1dZ7py",
        "https://www.youtube.com/playlist?list=PL3oW2tjiIxvTzCTED17SlRXKuX3s8YZdX",
        "https://www.youtube.com/playlist?list=PLV9Wb1iH5_BR0r8QwMCKa2QW2LaxhUJWk",
        "https://www.youtube.com/playlist?list=PLtqvK90f1RWyuk3F7S-MUz7Dnk3a1JpN4",
        "https://www.youtube.com/playlist?list=PLT7eHUdhOjk1w5EKeosFVJeFt3PcTJS3i"
    ]
}

# ----------------------------------------------------
# CREATE FOLDERS
# ----------------------------------------------------
def ensure_dirs():
    print("\n[INFO] Creating folders...")
    os.makedirs(BASE_DIR, exist_ok=True)

    for group in AGE_GROUPS:
        path = os.path.join(BASE_DIR, group)
        os.makedirs(path, exist_ok=True)
        print(f"[OK] {path}")

# ----------------------------------------------------
# DOWNLOAD FUNCTION
# ----------------------------------------------------
def download_playlist(group, url):
    print(f"\n[DOWNLOADING] {group} -> {url}\n")

    out_template = os.path.join(BASE_DIR, group, "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "-o", out_template,
        url
    ]

    subprocess.call(cmd)

# ----------------------------------------------------
# MAIN RUNNER
# ----------------------------------------------------
def run_downloader():
    ensure_dirs()

    for group, playlist_list in AGE_GROUPS.items():
        if not playlist_list:
            continue

        print("\n========================================")
        print(f"   DOWNLOADING: {group.upper()}")
        print("========================================")

        for url in playlist_list:
            download_playlist(group, url)

    print("\n========================================")
    print(" ALL PLAYLISTS DOWNLOADED / UPDATED ")
    print("========================================")
    time.sleep(1)

if __name__ == "__main__":
    run_downloader()
