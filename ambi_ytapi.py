#!/usr/bin/env python3
"""
Ambiance Offline System
- Fully offline playback by default (Option A: shuffle only within detected age folder)
- Optional downloader mode to populate folders via yt-dlp
- Camera snapshot -> age group -> play random file from group folder (shuffled)
- MPV playback (mpv must be installed and on PATH)
- Two preview windows: Camera Preview + Song Preview
- Keyboard controls for playback & UI

Usage:
  1) Ensure models are present next to script:
       deploy.prototxt
       res10_300x300_ssd_iter_140000.caffemodel
       deploy_age.prototxt
       age_net.caffemodel

  2) Install Python packages:
       pip install opencv-python numpy yt-dlp mutagen ytmusicapi pillow

  3) Install mpv and add to PATH (Windows: mpv.exe available).

  4) Create offline music folders (script will create them if missing):
       C:/Users/naman/VibeMusic/kids
       C:/Users/naman/VibeMusic/youths
       C:/Users/naman/VibeMusic/adults
       C:/Users/naman/VibeMusic/seniors

  5) Run in normal mode (offline playback):
       python ambiance_offline_system.py

     Or run downloader mode to populate folders (needs yt-dlp & internet):
       python ambiance_offline_system.py --download

Notes:
 - The script intentionally prefers offline files. If none exist it will wait and continue cycles.
 - Shuffle behaviour: only within detected group (Option A).
"""

import os
import sys
import time
import random
import threading
import subprocess
import platform
import traceback
from pathlib import Path
from datetime import datetime
import argparse

import cv2
import numpy as np
from mutagen import File as MutagenFile

# ---------------- CONFIG ----------------
# Camera source - USB camera index (0)
CAMERA_SOURCE = 0
ALTERNATE_CAMERA = 1  # set to RTSP string if you have one
CAPTURE_INTERVAL = 6.0  # seconds between cycles
TEMP_DIR = "temp_faces"
# Root folder for offline music
ROOT_MUSIC_DIR = Path(r"C:/Users/naman/OfflinePlayback")  # FIXED: correct folder name
MPV_BIN = "mpv"  # ensure mpv is on PATH
SHUFFLE = True  # when selecting from group folder, shuffle
CONTINUOUS = True  # run loop continuously

# Downloader config (optional). Fill with playlist URLs to auto-download.
PLAYLISTS_TO_DOWNLOAD = {
    # "kids": ["https://www.youtube.com/playlist?list=..."],
    # "youths": [...],
    # "adults": [...],
    # "seniors": [...],
}

# Acceptable offline extensions
OFFLINE_EXTS = {".mp3", ".m4a", ".mp4", ".wav", ".flac", ".ogg", ".webm", ".mkv", ".avi", ".opus", ".mpeg"}

# DNN model filenames (must be in same folder)
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO = "deploy_age.prototxt"
AGE_MODEL = "age_net.caffemodel"
AGE_MIDPOINTS = np.array([1,5,10,18,28,40,50,70], dtype=np.float32)

# mpv launch settings
MPV_VOLUME = 80

# retry constants
MPV_LAUNCH_RETRIES = 2
MPV_FAIL_SLEEP = 2.0

# ---------------- Sanity / prepare dirs ----------------
os.makedirs(TEMP_DIR, exist_ok=True)
ROOT_MUSIC_DIR.mkdir(parents=True, exist_ok=True)
for g in ("kids","youths","adults","seniors"):
    (ROOT_MUSIC_DIR / g).mkdir(parents=True, exist_ok=True)

IS_WINDOWS = platform.system().lower().startswith("windows")

# ---------------- Helpers: models, detection, age ----------------
def check_models():
    missing = [p for p in (FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"[ERROR] Missing DNN model files: {missing}")

def load_nets():
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    return face_net, age_net

def detect_faces(frame, net, conf_threshold=0.5):
    h,w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)
    dets = net.forward()
    boxes = []
    for i in range(dets.shape[2]):
        conf = float(dets[0,0,i,2])
        if conf > conf_threshold:
            box = dets[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype("int")
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            boxes.append((x1,y1,x2,y2,conf))
    return boxes

def predict_age(age_net, face_img):
    if face_img.size == 0:
        return None
    face_img = cv2.resize(face_img, (227,227))
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227,227),
                                 (78.4263377603,87.7689143744,114.895847746),
                                 swapRB=False, crop=False)
    age_net.setInput(blob)
    preds = age_net.forward()[0]
    probs = np.exp(preds - np.max(preds))
    probs /= probs.sum()
    expected = float((probs * AGE_MIDPOINTS).sum())
    return float(np.clip(expected, 0, 100))

def age_to_group(age):
    if age <= 12: return "kids"
    if age <= 25: return "youths"
    if age <= 60: return "adults"
    return "seniors"

def cleanup_temp():
    for f in os.listdir(TEMP_DIR):
        try: os.remove(os.path.join(TEMP_DIR,f))
        except: pass

# ---------------- Offline music helpers ----------------
def list_offline_files(group):
    p = ROOT_MUSIC_DIR / group
    if not p.exists(): return []
    files = [str(x.resolve()) for x in p.iterdir() if x.is_file() and x.suffix.lower() in OFFLINE_EXTS]
    return files

def pick_offline_file_from_group(group):
    files = list_offline_files(group)
    if not files:
        return None
    if SHUFFLE:
        return random.choice(files)
    else:
        return files[0]

def extract_duration(path):
    try:
        m = MutagenFile(path)
        if not m or not getattr(m,'info',None): return 0
        return int(getattr(m.info,'length',0) or 0)
    except Exception:
        return 0

# ---------------- MPV wrapper ----------------
class MPVPlayer:
    def __init__(self, mpv_bin=MPV_BIN):
        self.mpv_bin = mpv_bin
        self.proc = None
        self.ipc_path = None
        if IS_WINDOWS:
            self.ipc_path = rf"\\.\pipe\mpvpipe_{os.getpid()}"
        else:
            self.ipc_path = f"/tmp/mpv_socket_{os.getpid()}.sock"
        self.sock = None

    def play(self, uri, volume=MPV_VOLUME):
        self.stop()
        # Build mpv command — keep it simple & robust
        cmd = [
            self.mpv_bin, uri,
            "--no-video",
            "--force-window=no",
            "--idle=no",
            "--really-quiet",
            "--no-terminal",
            f"--volume={int(volume)}"
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[ERROR] mpv not found on PATH. Install mpv and retry.")
            self.proc = None
            return False
        except Exception as e:
            print("[ERROR] mpv launch failed:", e)
            self.proc = None
            return False

        # small wait to check immediate exit
        time.sleep(0.25)
        if self.proc and (self.proc.poll() is not None):
            rc = self.proc.returncode
            print(f"[ERROR] mpv exited immediately with code {rc}. URI may be invalid or mpv error.")
            self.proc = None
            return False
        return True

    def is_playing(self):
        return self.proc is not None and (self.proc.poll() is None)

    def stop(self):
        try:
            if self.proc:
                try:
                    self.proc.terminate()
                except:
                    pass
        finally:
            self.proc = None

# ---------------- Song preview UI ----------------
def song_preview_ui(title, artist, duration, player, control):
    win = "Song Preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 700, 160)
    start = time.time()
    while not control.get("stop_event").is_set() and player.is_playing():
        img = np.zeros((160,700,3), dtype=np.uint8)
        cv2.putText(img, "Now Playing", (18,30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)
        cv2.putText(img, title[:60], (18,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(img, f"by {artist[:60]}", (18,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        elapsed = int(time.time() - start)
        if duration and duration>0:
            frac = min(1.0, max(0.0, elapsed/duration))
            text = f"{elapsed//60}:{elapsed%60:02d} / {duration//60}:{duration%60:02d}"
        else:
            frac = (elapsed % 10) / 10.0
            text = f"{elapsed//60}:{elapsed%60:02d} / --:--"
        bar_w = int(frac * 640)
        cv2.rectangle(img,(30,125),(670,135),(60,60,60),-1)
        cv2.rectangle(img,(30,125),(30+bar_w,135),(0,200,0),-1)
        cv2.putText(img, text, (520,98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200),1)
        cv2.putText(img, "[p]=Pause [n]=Next [s]=Stop [q]=Quit", (18,145), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,180,180),1)
        cv2.imshow(win, img)
        k = cv2.waitKey(200) & 0xFF
        if k == ord('q'):
            control['action'] = 'quit'; control['stop_event'].set(); player.stop(); break
        if k == ord('n'):
            control['action'] = 'next'; control['stop_event'].set(); player.stop(); break
        if k == ord('s'):
            control['action'] = 'stop'; control['stop_event'].set(); player.stop(); break
        if k == ord('p'):
            # mpv IPC pause may not be available; warn user
            print("[CTRL] Pause requested — mpv IPC not implemented for pause here.")
    try: cv2.destroyWindow(win)
    except: pass

# ---------------- Downloader helper (optional) ----------------
def run_downloader_for_playlists(playlists: dict):
    """
    playlists: dict group -> list of youtube links (playlist or video).
    Uses yt-dlp to download best audio into ROOT_MUSIC_DIR/<group>/
    """
    if not playlists:
        print("[DL] No playlists defined. Populate PLAYLISTS_TO_DOWNLOAD in the script first.")
        return
    # check yt-dlp availability
    try:
        subprocess.run(["yt-dlp", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        print("[DL] yt-dlp not found. Install via 'pip install yt-dlp' or place yt-dlp.exe on PATH.")
        return

    for group, urls in playlists.items():
        outdir = ROOT_MUSIC_DIR / group
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[DL] Downloading into {outdir} ...")
        for url in urls:
            print(f"[DL] -> {url}")
            # yt-dlp options: extract audio to mp3, keep filename as title
            cmd = [
                "yt-dlp",
                "-f", "bestaudio",
                "--extract-audio",
                "--audio-format", "mp3",
                "--no-overwrites",
                "--yes-playlist",
                "--ignore-errors",
                "-o", str(outdir / "%(title)s.%(ext)s"),
                url
            ]
            try:
                subprocess.run(cmd, check=False)
            except Exception as e:
                print("[DL] yt-dlp error:", e)
    print("[DL] Downloader finished.")

# ---------------- MAIN LOOP ----------------
def main_loop():
    print("[INFO] Ambiance offline system starting (Option A: shuffle within group).")
    check_models()
    face_net, age_net = load_nets()

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_SOURCE}. Try changing CAMERA_SOURCE or check camera.")
        return

    player = MPVPlayer(mpv_bin=MPV_BIN)

    try:
        while True:
            # take single snapshot
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera capture failed; retrying in 1s.")
                time.sleep(1)
                continue

            boxes = detect_faces(frame, face_net)
            if not boxes:
                overlay = frame.copy()
                cv2.putText(overlay, "No faces detected — waiting...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,200),2)
                cv2.imshow("Camera Preview", overlay)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                time.sleep(CAPTURE_INTERVAL)
                continue

            cleanup_temp()
            ts = int(time.time())
            ages = []
            for i, (x1,y1,x2,y2,conf) in enumerate(boxes):
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                fname = os.path.join(TEMP_DIR, f"face_{ts}_{i}.jpg")
                cv2.imwrite(fname, face_img)
                try:
                    age = predict_age(age_net, face_img)
                except Exception:
                    age = None
                if age is not None:
                    ages.append(age)
            if not ages:
                print("[WARN] Age prediction failed; skipping this cycle.")
                time.sleep(CAPTURE_INTERVAL)
                continue

            avg_age = float(np.mean(ages))
            group = age_to_group(avg_age)
            print(f"[INFO] Avg age {avg_age:.1f} -> group '{group}'")

            # pick offline file from the detected group ONLY (Option A)
            chosen = pick_offline_file_from_group(group)
            if not chosen:
                print(f"[WARN] No offline files found in {group}. Please add files to {ROOT_MUSIC_DIR/group} or run downloader.")
                time.sleep(CAPTURE_INTERVAL)
                continue

            title = Path(chosen).name
            artist = "Local"
            duration = extract_duration(chosen)

            # try mpv play with a couple attempts
            ok = False
            for attempt in range(1, MPV_LAUNCH_RETRIES+1):
                ok = player.play(chosen, volume=MPV_VOLUME)
                if ok: break
                else:
                    print(f"[WARN] mpv start failed (attempt {attempt}). Retrying in {MPV_FAIL_SLEEP}s")
                    time.sleep(MPV_FAIL_SLEEP)
            if not ok:
                print("[ERROR] mpv couldn't play the file. Skipping this track.")
                time.sleep(CAPTURE_INTERVAL)
                continue

            # spawn song preview thread
            control = {"stop_event": threading.Event(), "action": None}
            ui_t = threading.Thread(target=song_preview_ui, args=(title, artist, duration, player, control), daemon=True)
            ui_t.start()

            # while playing show camera snapshot + overlay + handle keys
            while player.is_playing():
                vis = frame.copy()
                for (x1,y1,x2,y2,_) in boxes:
                    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(vis, f"Group:{group} Age:{avg_age:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                cv2.putText(vis, f"Now: {title}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                cv2.imshow("Camera Preview", vis)
                k = cv2.waitKey(120) & 0xFF
                if k == ord('v'):
                    # toggle camera source if you configured ALTERNATE_CAMERA (not switching cv2.VideoCapture here)
                    print("[UI] Toggle camera action requested (not implemented live in this simple script).")
                elif k == ord('r'):
                    print("[UI] Recalibrate requested - stopping playback and continuing.")
                    player.stop()
                    control["stop_event"].set()
                    break
                elif k == ord('z'):
                    global SHUFFLE
                    SHUFFLE = not SHUFFLE
                    print(f"[UI] Shuffle set to {SHUFFLE}")
                elif k == 27:
                    print("[UI] ESC pressed — exiting.")
                    control["stop_event"].set()
                    player.stop()
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                # check song preview actions
                if control.get("action"):
                    act = control["action"]
                    if act in ("next","stop","quit"):
                        print("[UI] action from song preview:", act)
                        control["action"] = None
                        player.stop()
                        control["stop_event"].set()
                        break

            control["stop_event"].set()
            player.stop()
            print("[INFO] Playback finished. Waiting", CAPTURE_INTERVAL, "s before next snapshot.")
            time.sleep(CAPTURE_INTERVAL)
            if not CONTINUOUS:
                break

    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    except Exception as e:
        print("[ERROR] Exception in main loop:", e)
        traceback.print_exc()
    finally:
        try: player.stop()
        except: pass
        try: cap.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass

# ---------------- CLI / run ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Ambiance offline system (Option A: shuffle inside age group).")
    ap.add_argument("--download", action="store_true", help="Run downloader for PLAYLISTS_TO_DOWNLOAD then exit (requires yt-dlp).")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.download:
        run_downloader_for_playlists(PLAYLISTS_TO_DOWNLOAD)
        print("[MAIN] Downloader run complete. Exiting.")
        sys.exit(0)
    main_loop()
