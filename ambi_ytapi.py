# ambi_hybrid_mpv_retry.py
"""
Hybrid ambience player (online-first, offline fallback) with limited retries and multi-invidious fallback.

Keyboard controls:
 p = pause/resume
 n = next (skip)
 s = stop
 q = quit

Requirements:
 pip install opencv-python numpy yt-dlp ytmusicapi mutagen Pillow
 Install mpv and add to PATH.
 Place DNN models next to script:
  - deploy.prototxt
  - res10_300x300_ssd_iter_140000.caffemodel
  - deploy_age.prototxt
  - age_net.caffemodel
"""

import os
import time
import random
import threading
import subprocess
import platform
import socket
import traceback
from pathlib import Path
import io
import json

import cv2
import numpy as np
import yt_dlp
from ytmusicapi import YTMusic
from mutagen import File as MutagenFile
from PIL import Image

# ---------------- User config ----------------
CAMERA_SOURCE = 0                        # USB camera index (0)
CAPTURE_INTERVAL = 6.0                   # seconds between snapshots after each song cycle
TEMP_DIR = "temp_faces"
ROOT_MUSIC_DIR = r"C:/Users/naman/VibeMusic"  # offline root (Windows path)
MPV_BIN = "mpv"                          # mpv on PATH
SHUFFLE = True
CONTINUOUS = True

# Per-track limited retries (you confirmed limited retry behavior)
MAX_PER_TRACK_RETRIES = 3
RETRY_BACKOFF = 1.0                      # seconds multiplier between attempts

# Background online retry when in offline mode
ONLINE_RETRY_INTERVAL = 30.0             # background tries every N seconds

# ytmusic / search configuration
YT_SEARCH_CANDIDATES = 8

# Invidious instances (multi-server fallback)
INVIDIOUS_INSTANCES = [
    "https://iv.nadeko.net/latest_version",
    "https://invidious.snopyta.org/latest_version",
    "https://inv.tux.pizza/latest_version",
    "https://iv.ggtyler.dev/latest_version"
]
INVIDIOUS_ITAGS = [140, 251]  # try 140 (m4a) then 251 (webm opus)

# Acceptable offline file extensions (play any media files)
OFFLINE_EXTS = {".mp3", ".m4a", ".mp4", ".wav", ".flac", ".ogg", ".webm", ".mkv", ".avi", ".mov", ".opus", ".mpeg"}

# DNN model filenames (must be present)
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO = "deploy_age.prototxt"
AGE_MODEL = "age_net.caffemodel"
AGE_MIDPOINTS = np.array([1,5,10,18,28,40,50,70], dtype=np.float32)

# Ensure directories exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ROOT_MUSIC_DIR, exist_ok=True)
for g in ("kids","youths","adults","seniors"):
    os.makedirs(Path(ROOT_MUSIC_DIR)/g, exist_ok=True)

# platform / optional pywin32 for IPC
IS_WINDOWS = platform.system().lower().startswith("windows")
try:
    if IS_WINDOWS:
        import win32file  # type: ignore
        HAVE_PYWIN32 = True
    else:
        HAVE_PYWIN32 = False
except Exception:
    HAVE_PYWIN32 = False

# Init YTMusic
ytmusic = YTMusic()

# ---------------- DNN helpers ----------------
def check_models():
    missing = [p for p in (FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL) if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"[ERROR] Missing DNN models: {missing}")

def load_nets():
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    return face_net, age_net

def detect_faces(frame, net, conf_threshold=0.5):
    h,w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes=[]
    for i in range(detections.shape[2]):
        conf = float(detections[0,0,i,2])
        if conf>conf_threshold:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype("int")
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            boxes.append((x1,y1,x2,y2,conf))
    return boxes

def predict_age(age_net, face_img):
    if face_img.size==0:
        return None
    face_img = cv2.resize(face_img,(227,227))
    blob = cv2.dnn.blobFromImage(face_img,1.0,(227,227),(78.4263377603,87.7689143744,114.895847746), swapRB=False, crop=False)
    age_net.setInput(blob)
    preds = age_net.forward()[0]
    probs = np.exp(preds - np.max(preds))
    probs /= probs.sum()
    expected_age = float((probs * AGE_MIDPOINTS).sum())
    return float(np.clip(expected_age,0,100))

def age_to_group(age):
    if age <= 12: return "kids"
    if age <= 25: return "youths"
    if age <= 60: return "adults"
    return "seniors"

def cleanup_temp():
    for f in os.listdir(TEMP_DIR):
        try: os.remove(os.path.join(TEMP_DIR,f))
        except: pass

# ---------------- Offline helpers ----------------
def list_offline_files_in_group(group):
    p = Path(ROOT_MUSIC_DIR)/group
    if not p.exists(): return []
    files = [str(x.resolve()) for x in p.iterdir() if x.is_file() and x.suffix.lower() in OFFLINE_EXTS]
    if SHUFFLE:
        random.shuffle(files)
    else:
        files.sort()
    return files

def pick_offline_file(group):
    files = list_offline_files_in_group(group)
    if not files: return None
    return random.choice(files)

def extract_offline_duration(path):
    try:
        m = MutagenFile(path)
        if not m or not getattr(m,'info',None): return 0
        dur = int(getattr(m.info,'length',0) or 0)
        return dur
    except Exception:
        return 0

# ---------------- YTMusic + extraction ----------------
def find_youtube_candidate(group, candidates=YT_SEARCH_CANDIDATES):
    queries = {
        "kids":"Kids songs",
        "youths":"Pop music",
        "adults":"Relaxing instrumental music",
        "seniors":"Retro classic songs"
    }
    q = queries.get(group, "Popular music")
    try:
        results = ytmusic.search(q, filter="songs")[:candidates]
        if not results: return None, None, None
        choice = random.choice(results)
        video_id = choice.get('videoId') or choice.get('id')
        title = choice.get('title', 'YouTube')
        artists = ", ".join([a.get('name','') for a in choice.get('artists',[])]) if choice.get('artists') else (choice.get('artist') or 'YouTube')
        return video_id, title, artists
    except Exception as e:
        print("[WARN] ytmusic search failed:", e)
        return None, None, None

def extract_direct_audio_url(video_id, format_preference="bestaudio"):
    if not video_id: return None, 0
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {"quiet": True, "skip_download": True, "no_warnings": True, "format": format_preference}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = int(info.get("duration") or 0)
            direct = info.get("url")
            if direct:
                return direct, duration
            formats = info.get("formats") or []
            audio_formats = [f for f in formats if f.get("acodec") and f.get("url") and f.get("acodec")!="none"]
            if audio_formats:
                audio_formats.sort(key=lambda f: ((f.get("abr") or 0) + (f.get("tbr") or 0)), reverse=True)
                return audio_formats[0].get("url"), duration
    except Exception as e:
        print("[WARN] yt-dlp extract failed:", e)
    return None, 0

def build_invidious_stream(video_id):
    """Try multiple invidious instances and itags. Return first working stream URL or None."""
    if not video_id: return None, 0
    for inst in INVIDIOUS_INSTANCES:
        for itag in INVIDIOUS_ITAGS:
            candidate = f"{inst}?id={video_id}&itag={itag}"
            # quick basic check using yt-dlp to ensure it's playable (extract_info will usually succeed)
            opts = {"quiet": True, "skip_download": True, "no_warnings": True}
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(candidate, download=False)
                    dur = int(info.get("duration") or 0)
                    direct = info.get("url") or candidate
                    # return direct or candidate; mpv tends to accept invidious direct formats directly
                    return direct, dur
            except Exception:
                # try next itag/instance
                continue
    return None, 0

# ---------------- MPV wrapper ----------------
class MPVPlayer:
    def __init__(self, mpv_bin=MPV_BIN):
        self.mpv_bin = mpv_bin
        self.proc = None
        pid = os.getpid()
        if IS_WINDOWS:
            self.ipc_path = rf"\\.\pipe\mpvpipe_{pid}"
        else:
            self.ipc_path = f"/tmp/mpv_socket_{pid}.sock"
        self.sock = None
        self.pipe_handle = None

    def play(self, uri, volume=80):
        self.stop()
        cmd = [
            self.mpv_bin, uri,
            "--no-video",
            "--idle=yes",
            "--keep-open=no",
            "--no-terminal",
            f"--volume={int(volume)}",
            "--really-quiet",
            f"--input-ipc-server={self.ipc_path}"
        ]
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[ERROR] mpv not found on PATH.")
            self.proc = None
            return False
        except Exception as e:
            print("[ERROR] launching mpv failed:", e)
            self.proc = None
            return False

        # best-effort IPC connect
        for _ in range(40):
            try:
                if IS_WINDOWS:
                    if not HAVE_PYWIN32:
                        break
                    handle = win32file.CreateFile(self.ipc_path,
                                                  win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                                                  0, None, win32file.OPEN_EXISTING, 0, None)
                    self.pipe_handle = handle
                    break
                else:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.ipc_path)
                    self.sock = s
                    break
            except Exception:
                time.sleep(0.05)

        time.sleep(0.3)
        if self.proc and (self.proc.poll() is not None):
            rc = self.proc.returncode
            print(f"[ERROR] mpv exited immediately with code {rc}. URI probably invalid or mpv error.")
            self.proc = None
            return False
        return True

    def is_playing(self):
        return self.proc is not None and (self.proc.poll() is None)

    def send(self, cmd_list):
        payload = json.dumps({"command": cmd_list}) + "\n"
        try:
            if IS_WINDOWS:
                if not HAVE_PYWIN32 or not self.pipe_handle:
                    return False
                win32file.WriteFile(self.pipe_handle, payload.encode("utf-8"))
                return True
            else:
                if not self.sock:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.ipc_path)
                    self.sock = s
                self.sock.sendall(payload.encode("utf-8"))
                return True
        except Exception:
            return False

    def toggle_pause(self):
        return self.send(["cycle","pause"])

    def stop(self):
        try:
            if self.proc:
                try:
                    self.send(["quit"])
                except:
                    pass
                try:
                    self.proc.wait(timeout=1)
                except:
                    try:
                        self.proc.terminate()
                    except:
                        pass
        finally:
            self.proc = None
            try:
                if self.sock:
                    self.sock.close(); self.sock=None
            except:
                pass
            try:
                if self.pipe_handle:
                    win32file.CloseHandle(self.pipe_handle); self.pipe_handle=None
            except:
                pass
            if (not IS_WINDOWS) and os.path.exists(self.ipc_path):
                try: os.remove(self.ipc_path)
                except: pass

# ---------------- Background retry worker ----------------
class OnlineRetryWorker(threading.Thread):
    def __init__(self, shared_state, interval=ONLINE_RETRY_INTERVAL):
        super().__init__(daemon=True)
        self.shared = shared_state
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            if self.shared.get("mode") == "offline":
                grp = self.shared.get("last_group")
                if grp:
                    try:
                        vid, title, artist = find_youtube_candidate(grp)
                        if vid:
                            direct, dur = extract_direct_audio_url(vid)
                            if direct:
                                self.shared["online_candidate"] = {"uri": direct, "title": title, "artist": artist, "duration": dur}
                                print("[RETRY] Background found online candidate:", title)
                                # let main loop pick it
                    except Exception as e:
                        print("[RETRY] background exception:", e)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()

# ---------------- Song preview UI + keyboard (OpenCV) ----------------
def song_preview(title, artist, duration, player, control):
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
            frac = (elapsed%10)/10.0
            text = f"{elapsed//60}:{elapsed%60:02d} / --:--"
        bar_w = int(frac*640)
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
            ok = player.toggle_pause()
            if not ok:
                print("[WARN] Pause/resume IPC not available.")
    try: cv2.destroyWindow(win)
    except: pass

# ---------------- Main controller ----------------
def main():
    print("[INFO] Starting hybrid player (limited per-track retries).")
    check_models()
    face_net, age_net = load_nets()
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera:", CAMERA_SOURCE); return

    player = MPVPlayer(mpv_bin=MPV_BIN)
    shared = {"mode":"online","last_group":None,"online_candidate":None}

    retry_worker = OnlineRetryWorker(shared_state=shared)
    retry_worker.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Camera read failed; retrying..."); time.sleep(1); continue

            boxes = detect_faces(frame, face_net)
            if not boxes:
                overlay = frame.copy()
                cv2.putText(overlay,"No faces detected — waiting...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,200),2)
                cv2.imshow("Camera Preview", overlay)
                if cv2.waitKey(1) & 0xFF == 27: break
                time.sleep(CAPTURE_INTERVAL)
                continue

            cleanup_temp()
            ts = int(time.time())
            ages=[]
            for i,(x1,y1,x2,y2,conf) in enumerate(boxes):
                face_img = frame[y1:y2, x1:x2]
                if face_img.size==0: continue
                fname = os.path.join(TEMP_DIR,f"face_{ts}_{i}.jpg")
                cv2.imwrite(fname, face_img)
                age = predict_age(age_net, face_img)
                if age is not None: ages.append(age)
            if not ages:
                print("[WARN] Age prediction failed; skipping."); time.sleep(CAPTURE_INTERVAL); continue

            avg_age = float(np.mean(ages)); group = age_to_group(avg_age)
            shared['last_group'] = group
            print(f"[INFO] Avg age {avg_age:.1f} -> group '{group}'")

            chosen_uri=None; chosen_title=None; chosen_artist=None; chosen_duration=0

            # if background supplied candidate, use it first
            candidate = shared.get("online_candidate")
            if candidate:
                chosen_uri = candidate.get("uri"); chosen_title = candidate.get("title"); chosen_artist = candidate.get("artist"); chosen_duration = candidate.get("duration",0)
                shared["online_candidate"] = None
                shared["mode"] = "online"
                print("[INFO] Using background online candidate:", chosen_title)

            # else attempt online with limited retries (MAX_PER_TRACK_RETRIES)
            if not chosen_uri:
                retries = 0
                while retries < MAX_PER_TRACK_RETRIES and not chosen_uri:
                    retries += 1
                    print(f"[TRY] Online attempt {retries}/{MAX_PER_TRACK_RETRIES} for group {group}")
                    vid, title, artist = find_youtube_candidate(group)
                    if vid:
                        # try yt-dlp extraction first (direct)
                        direct, dur = extract_direct_audio_url(vid)
                        if direct:
                            chosen_uri = direct; chosen_title = title; chosen_artist = artist; chosen_duration = dur; shared["mode"]="online"
                            print("[INFO] Found online direct stream:", chosen_title); break
                        # else try invidious fallback across instances/itags
                        direct_iv, dur_iv = build_invidious_stream(vid)
                        if direct_iv:
                            chosen_uri = direct_iv; chosen_title = title; chosen_artist = artist; chosen_duration = dur_iv; shared["mode"]="online"
                            print("[INFO] Found invidious stream:", chosen_title); break
                        else:
                            print("[WARN] Extraction failed for vid", vid)
                    else:
                        print("[WARN] No youtube candidate found in attempt", retries)
                    # backoff before next attempt
                    time.sleep(RETRY_BACKOFF * retries)
                # end retries

            # If still no chosen_uri => offline fallback
            if not chosen_uri:
                offline_file = pick_offline_file(group)
                if offline_file:
                    chosen_uri = offline_file; chosen_title = Path(offline_file).name; chosen_artist = "Local"; chosen_duration = extract_offline_duration(offline_file); shared["mode"]="offline"
                    print("[INFO] Offline fallback selected:", chosen_title)
                else:
                    print("[ERROR] No offline files either. Will wait and continue (background retry running).")
                    time.sleep(CAPTURE_INTERVAL)
                    continue

            # Attempt mpv play; if fails and was online, do limited re-attempts (already tried extraction though)
            ok = player.play(chosen_uri)
            if not ok:
                print("[ERROR] mpv failed to start for track. If online, try offline fallback.")
                if shared.get("mode")=="online":
                    offline_file = pick_offline_file(group)
                    if offline_file:
                        print("[INFO] Trying offline after mpv failure.")
                        chosen_uri = offline_file; chosen_title = Path(offline_file).name; chosen_artist="Local"; chosen_duration=extract_offline_duration(offline_file); shared["mode"]="offline"
                        ok = player.play(chosen_uri)
                        if not ok:
                            print("[ERROR] Offline mpv also failed. Skipping cycle.")
                            time.sleep(CAPTURE_INTERVAL); continue
                    else:
                        print("[ERROR] No offline files; skipping.")
                        time.sleep(CAPTURE_INTERVAL); continue
                else:
                    print("[ERROR] mpv couldn't start offline file as well; skipping.")
                    time.sleep(CAPTURE_INTERVAL); continue

            # Launch song preview UI thread
            control = {"stop_event": threading.Event(), "action": None}
            ui_th = threading.Thread(target=song_preview, args=(chosen_title or "Unknown", chosen_artist or "", chosen_duration, player, control), daemon=True)
            ui_th.start()

            # While playing show camera snapshot + handle keyboard controls
            while player.is_playing():
                vis = frame.copy()
                for (x1,y1,x2,y2,_) in boxes:
                    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(vis, f"Group:{group} Age:{avg_age:.1f} Mode:{shared.get('mode')}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)
                cv2.putText(vis, f"Now: {chosen_title or '---'}",(10,45),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
                cv2.imshow("Camera Preview", vis)
                k = cv2.waitKey(120) & 0xFF

                if k == ord('p'):
                    ok = player.toggle_pause()
                    if not ok:
                        print("[CTRL] Pause/resume IPC not available.")
                elif k == ord('n'):
                    print("[CTRL] Next pressed."); player.stop(); control["stop_event"].set(); break
                elif k == ord('s'):
                    print("[CTRL] Stop pressed."); player.stop(); control["stop_event"].set(); break
                elif k == ord('q'):
                    print("[CTRL] Quit pressed. Exiting."); player.stop(); control["stop_event"].set(); cap.release(); cv2.destroyAllWindows(); retry_worker.stop(); return

                if control.get("action"):
                    action = control["action"]
                    if action in ("next","skip","stop","quit"):
                        print("[CTRL] action from song preview:", action)
                        control["action"] = None
                        player.stop()
                        control["stop_event"].set()
                        break

            control["stop_event"].set()
            player.stop()
            print("[INFO] Playback finished; waiting", CAPTURE_INTERVAL, "s before next cycle.")
            time.sleep(CAPTURE_INTERVAL)
            if not CONTINUOUS:
                break

    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt; stopping.")
    except Exception as e:
        print("[ERROR] exception in main:", e); traceback.print_exc()
    finally:
        try: player.stop()
        except: pass
        retry_worker.stop()
        try: cap.release()
        except: pass
        try: cv2.destroyAllWindows()
        except: pass

if __name__ == "__main__":
    main()
