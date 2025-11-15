# ambi_dashboard_mpv_ipc.py
"""
Ambience detection + Flask dashboard + mpv JSON IPC + offline fallback.

Highlights:
- mpv controlled via JSON IPC socket (Unix) or named pipe (Windows via pywin32).
- Reliable Play / Pause / Stop / Volume control from dashboard and preview window.
- Offline local playlist fallback when YouTube streaming fails.
- Detection pauses while a track is selected; staff resumes detection via dashboard button.
- Preview OpenCV window displays camera feed and small overlay with current track info; keyboard controls included.

Requirements:
- Python 3.8+
- mpv on PATH
- Python packages:
    pip install opencv-python numpy flask google-api-python-client yt-dlp

- Optional on Windows (for full mpv IPC):
    pip install pywin32

Usage:
- Put Caffe models (deploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel,
  deploy_age.prototxt, age_net.caffemodel) in the same folder.
- Create offline_music/ and add mp3 files.
- Set YOUTUBE_API_KEY in the script.
- Run: python ambi_dashboard_mpv_ipc.py
- Open dashboard: http://127.0.0.1:5000
"""

import os
import sys
import time
import json
import random
import threading
import subprocess
from pathlib import Path
from urllib.parse import quote_plus

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from googleapiclient.discovery import build
import yt_dlp
import socket
import platform

# Try import pywin32 modules (Windows IPC). If unavailable we'll fallback.
IS_WINDOWS = platform.system().lower().startswith("windows")
try:
    if IS_WINDOWS:
        import win32file
        import win32pipe
        import pywintypes
        HAVE_PYWIN32 = True
    else:
        HAVE_PYWIN32 = False
except Exception:
    HAVE_PYWIN32 = False

# ---------------- Config ----------------
CAMERA_SOURCE = 0  # replace with RTSP string if using IP camera
CAPTURE_INTERVAL = 5.0
TEMP_DIR = "temp_faces"
OFFLINE_DIR = "offline_music"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OFFLINE_DIR, exist_ok=True)

# Models
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO  = "deploy_age.prototxt"
AGE_MODEL  = "age_net.caffemodel"

AGE_BUCKETS = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
AGE_MIDPOINTS = np.array([1,5,10,18,28,40,50,70], dtype=np.float32)

# mpv binary (must be on PATH)
MPV_BIN = "mpv"

# YouTube API key (put your key here)
YOUTUBE_API_KEY = "AIzaSyBolLJYQd3VQtyw0GChK8y78E3JWGgz76k"
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY) if YOUTUBE_API_KEY != "PASTE_YOUR_KEY_HERE" else None

# Shared state
shared = {
    "playing": False,
    "current_video_id": None,
    "current_title": None,
    "current_artist": None,
    "shuffle": False,
    "offline_active": False,
    "volume": 70,
}
state_lock = threading.Lock()

# ---------------- Helper functions ----------------
def check_models():
    missing = [p for p in (FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL) if not os.path.exists(p)]
    if missing:
        print("[ERROR] Missing model files:", missing)
        raise SystemExit(1)

def load_nets():
    face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
    return face_net, age_net

def detect_faces_dnn(net, frame, conf_threshold=0.5):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = float(detections[0,0,i,2])
        if conf > conf_threshold:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype("int")
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(w-1,x2), min(h-1,y2)
            boxes.append((x1,y1,x2,y2,conf))
    return boxes

def predict_age(age_net, face_img):
    if face_img.size == 0:
        return None
    face_img = cv2.resize(face_img,(227,227))
    blob = cv2.dnn.blobFromImage(face_img,1.0,(227,227),(78.4263377603,87.7689143744,114.895847746), swapRB=False, crop=False)
    age_net.setInput(blob)
    preds = age_net.forward()[0]
    probs = np.exp(preds - np.max(preds))
    probs /= probs.sum()
    expected_age = float((probs * AGE_MIDPOINTS).sum())
    expected_age = np.clip(expected_age,0,100)
    return expected_age

def cleanup_temp():
    for f in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR,f))
        except:
            pass

def age_to_group(age):
    if age <= 12: return "kids"
    if age <= 25: return "youth"
    if age <= 40: return "adult"
    if age <= 60: return "mid"
    return "senior"

def fetch_song_for_group(group, max_results=6):
    if youtube is None:
        return None, None, None
    queries = {
        "kids":"popular kids songs",
        "youth":"latest pop hits",
        "adult":"relaxing instrumental music",
        "mid":"soft rock classics",
        "senior":"retro old hindi songs"
    }
    query = queries.get(group,"popular songs")
    try:
        req = youtube.search().list(part="snippet", q=query, type="video", maxResults=max_results, videoCategoryId="10")
        res = req.execute()
        items = res.get("items", [])
        if not items:
            return None,None,None
        # choose depending on shuffle flag
        item = random.choice(items) if shared.get("shuffle",False) else items[0]
        title = item["snippet"]["title"]
        artist = item["snippet"]["channelTitle"]
        video_id = item["id"]["videoId"]
        return video_id, title, artist
    except Exception as e:
        print("[YT ERROR]", e)
        return None,None,None

def get_offline_tracks():
    p = Path(OFFLINE_DIR)
    files = [str(x.resolve()) for x in p.iterdir() if x.suffix.lower() in (".mp3",".wav",".m4a",".flac",".ogg")]
    files.sort()
    return files

def choose_offline_track(shuffle=False):
    tracks = get_offline_tracks()
    if not tracks: return None
    return random.choice(tracks) if shuffle else tracks[0]

# ---------------- mpv JSON IPC Player ----------------
class MPVIPC:
    def __init__(self, mpv_bin=MPV_BIN):
        self.mpv_bin = mpv_bin
        self.process = None
        self.ipc_path = None
        self.sock = None
        self.pipe_handle = None
        self.lock = threading.Lock()
        self.volume = shared.get("volume",70)
        self.platform = platform.system().lower()

    def _make_ipc_path(self):
        pid = os.getpid()
        if IS_WINDOWS:
            # named pipe path
            name = f"\\\\.\\pipe\\mpv_pipe_{pid}"
            return name
        else:
            # unix domain socket path
            return f"/tmp/mpv_socket_{pid}.sock"

    def _start_mpv(self, uri, is_local=False):
        # create unique ipc path
        self.ipc_path = self._make_ipc_path()
        cmd = [self.mpv_bin,
               "--no-video",
               "--idle=no",
               f"--volume={int(self.volume)}",
               f"--input-ipc-server={self.ipc_path}",
               "--force-window=no",
               uri]
        # start mpv
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[ERROR] mpv not found. Install mpv and ensure it's on PATH.")
            self.process = None
            return False
        # connect to ipc
        connected = False
        for _ in range(50):
            try:
                if IS_WINDOWS:
                    if not HAVE_PYWIN32:
                        # can't use windows pipe without pywin32
                        break
                    # attempt to open named pipe
                    self.pipe_handle = win32file.CreateFile(self.ipc_path,
                                                           win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                                                           0, None, win32file.OPEN_EXISTING, 0, None)
                    connected = True
                    break
                else:
                    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self.sock.connect(self.ipc_path)
                    connected = True
                    break
            except Exception:
                time.sleep(0.05)
        if not connected:
            if IS_WINDOWS and not HAVE_PYWIN32:
                print("[WARN] pywin32 not installed — mpv Windows IPC unavailable. Falling back to stdin/key controls.")
            else:
                print("[WARN] Could not connect to mpv IPC.")
        return True

    def start(self, uri, is_local=False, title=None, artist=None):
        with self.lock:
            self.stop()
            ok = self._start_mpv(uri, is_local=is_local)
            if ok:
                print(f"[MPV] started {'local' if is_local else 'stream'} -> {title or uri}")
            return ok

    def stop(self):
        with self.lock:
            if self.process:
                # try graceful quit via IPC
                try:
                    self._send_command(["quit"])
                except Exception:
                    pass
                # give it a moment
                try:
                    self.process.wait(timeout=2)
                except Exception:
                    try: self.process.kill()
                    except: pass
                self.process = None
            # clean up sockets / handles
            try:
                if self.sock:
                    try: self.sock.close()
                    except: pass
                    self.sock = None
                if self.pipe_handle:
                    try: win32file.CloseHandle(self.pipe_handle)
                    except: pass
                    self.pipe_handle = None
                # remove unix socket file
                if self.ipc_path and (not IS_WINDOWS) and os.path.exists(self.ipc_path):
                    try:
                        os.remove(self.ipc_path)
                    except:
                        pass
            except Exception:
                pass
            print("[MPV] stopped")

    def _send_command(self, cmd_list):
        payload = json.dumps({"command": cmd_list}) + "\n"
        data = payload.encode("utf-8")
        if IS_WINDOWS:
            if not HAVE_PYWIN32 or not self.pipe_handle:
                raise RuntimeError("Windows pipe not available (pywin32 missing or not connected)")
            # WriteFile requires bytes; wrap in try
            win32file.WriteFile(self.pipe_handle, data)
            # Not reading response; that's fine
            return
        else:
            if not self.sock:
                raise RuntimeError("IPC socket not connected")
            self.sock.sendall(data)
            # try to read a response non-blocking (optional)
            try:
                self.sock.settimeout(0.1)
                resp = self.sock.recv(4096)
                return resp
            except Exception:
                return None

    def set_property(self, prop, value):
        try:
            self._send_command(["set_property", prop, value])
        except Exception as e:
            print("[MPV] set_property error:", e)

    def cycle_property(self, prop):
        try:
            self._send_command(["cycle", prop])
        except Exception as e:
            print("[MPV] cycle error:", e)

    def play_pause(self):
        try:
            self._send_command(["cycle", "pause"])
        except Exception as e:
            print("[MPV] play_pause error:", e)

    def set_volume(self, vol):
        with self.lock:
            self.volume = int(max(0,min(100,vol)))
            try:
                self._send_command(["set_property", "volume", self.volume])
            except Exception as e:
                print("[MPV] set_volume error (IPC):", e)
                # If IPC failed, restart process with new vol (fallback)
                if self.process:
                    cur = shared.get("current_video_id")
                    offline = shared.get("offline_active", False)
                    uri = cur if offline else (f"https://www.youtube.com/watch?v={cur}" if cur else None)
                    if uri:
                        print("[MPV] restarting mpv to apply volume")
                        self.start(uri, is_local=offline, title=shared.get("current_title"))

    def is_playing(self):
        return self.process is not None and (self.process.poll() is None)

# Create mpv ipc player instance
mpv_player = MPVIPC(mpv_bin=MPV_BIN)

# ---------------- Flask dashboard ----------------
app = Flask(__name__)
INDEX_HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><title>Ambience Dashboard</title>
<style>body{font-family:Arial;padding:12px;background:#f5f5f5}button{padding:8px 12px;margin:4px}</style>
</head><body>
<h3>Ambience Dashboard (mpv IPC)</h3>
<div><b>Current:</b> <span id="cur">No track</span></div>
<div>
  <button onclick="cmd('/play')">Play</button>
  <button onclick="cmd('/pause')">Pause</button>
  <button onclick="cmd('/next')">Next</button>
  <button onclick="cmd('/stop')">Stop</button>
  <button onclick="cmd('/resume')">Resume Detection</button>
  <button onclick="cmd('/open_youtube')">Open YouTube</button>
  <button onclick="toggleShuffle()">Shuffle: <span id='sh'>Off</span></button>
</div>
<div style="margin-top:8px;">Volume: <input id="vol" type="range" min="0" max="100" value="70" oninput="volChange(this.value)"><span id='v'>70</span></div>
<pre id="st">loading...</pre>

<script>
function cmd(path){ fetch(path, {method:'POST'}).then(()=>setTimeout(status,300)); }
function status(){ fetch('/status').then(r=>r.json()).then(j=>{ document.getElementById('st').textContent = JSON.stringify(j,null,2); document.getElementById('cur').textContent = j.current_title ? (j.current_title + ' — ' + j.current_artist) : 'No track'; document.getElementById('sh').textContent = j.shuffle ? 'On' : 'Off'; document.getElementById('vol').value = j.volume; document.getElementById('v').textContent = j.volume; }); }
function toggleShuffle(){ fetch('/toggle_shuffle',{method:'POST'}).then(()=>setTimeout(status,200)); }
function volChange(v){ document.getElementById('v').textContent = v; fetch('/set_volume',{method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({volume:parseInt(v)})}).then(()=>setTimeout(status,200)); }
setInterval(status,2000); status();
</script>
</body></html>
"""

@app.route("/")
def index():
    return INDEX_HTML

@app.route("/status")
def status():
    with state_lock:
        return jsonify(shared)

@app.route("/play", methods=["POST"])
def route_play():
    with state_lock:
        vid = shared.get("current_video_id")
        offline = shared.get("offline_active", False)
        if not vid:
            return ("No track",400)
        if offline:
            uri = vid
        else:
            uri = f"https://www.youtube.com/watch?v={vid}"
    # start mpv with IPC
    mpv_player.start(uri, is_local=offline, title=shared.get("current_title"), artist=shared.get("current_artist"))
    return ("",204)

@app.route("/pause", methods=["POST"])
def route_pause():
    mpv_player.play_pause()
    return ("",204)

@app.route("/stop", methods=["POST"])
def route_stop():
    mpv_player.stop()
    with state_lock:
        shared["playing"]=False
        shared["current_video_id"]=None
        shared["current_title"]=None
        shared["current_artist"]=None
        shared["offline_active"]=False
    return ("",204)

@app.route("/next", methods=["POST"])
def route_next():
    # pick next adult online or fallback to local
    with state_lock:
        shared["playing"]=False
    video_id, title, artist = fetch_song_for_group("adult")
    if video_id:
        with state_lock:
            shared["current_video_id"]=video_id
            shared["current_title"]=title
            shared["current_artist"]=artist
            shared["offline_active"]=False
            shared["playing"]=True
        uri = f"https://www.youtube.com/watch?v={video_id}"
        mpv_player.start(uri, is_local=False, title=title, artist=artist)
        return ("",204)
    # offline fallback
    track = choose_offline_track(shuffle=shared.get("shuffle",False))
    if track:
        with state_lock:
            shared["current_video_id"]=track
            shared["current_title"]=os.path.basename(track)
            shared["current_artist"]="Local playlist"
            shared["offline_active"]=True
            shared["playing"]=True
        mpv_player.start(track, is_local=True, title=shared["current_title"], artist=shared["current_artist"])
        return ("",204)
    return ("No track",500)

@app.route("/resume", methods=["POST"])
def route_resume():
    with state_lock:
        shared["playing"] = False
    return ("",204)

@app.route("/toggle_shuffle", methods=["POST"])
def route_shuffle():
    with state_lock:
        shared["shuffle"] = not shared["shuffle"]
    return ("",204)

@app.route("/open_youtube", methods=["POST"])
def route_open_yt():
    with state_lock:
        vid = shared.get("current_video_id")
        offline = shared.get("offline_active", False)
    if offline or not vid:
        return ("No online video",400)
    import webbrowser
    webbrowser.open(f"https://www.youtube.com/watch?v={vid}")
    return ("",204)

@app.route("/set_volume", methods=["POST"])
def route_set_volume():
    data = request.get_json() or {}
    vol = int(data.get("volume", shared.get("volume",70)))
    with state_lock:
        shared["volume"]=vol
    mpv_player.set_volume(vol)
    return ("",204)

# ---------------- Detection loop ----------------
def draw_overlay(frame):
    with state_lock:
        title = shared.get("current_title") or ""
        artist = shared.get("current_artist") or ""
        offline = shared.get("offline_active", False)
    h,w = frame.shape[:2]
    overlay = np.zeros((60,w,3),dtype=np.uint8)
    overlay[:] = (40,40,40)
    cv2.putText(overlay, f"{title[:70]}", (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230),1)
    cv2.putText(overlay, f"{artist[:60]} {'(offline)' if offline else ''}", (8,45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)
    try:
        frame[-60:,:,:] = cv2.addWeighted(frame[-60:,:,:], 0.4, overlay, 0.6, 0)
    except:
        pass

def wait_player_end_or_staff():
    while True:
        with state_lock:
            if not shared.get("playing"):
                mpv_player.stop()
                break
        if not mpv_player.is_playing():
            with state_lock:
                shared["playing"]=False
                shared["offline_active"]=False
            break
        time.sleep(0.4)

def detection_loop(camera_source=CAMERA_SOURCE, capture_interval=CAPTURE_INTERVAL):
    print("[DETECT] starting")
    check_models()
    face_net, age_net = load_nets()
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print("[ERROR] cannot open camera:", camera_source)
        return
    last_capture = 0.0
    while True:
        with state_lock:
            if shared.get("playing"):
                # show preview but don't detect heavy ops
                ret, f = cap.read()
                if ret:
                    draw_overlay(f)
                    cv2.imshow("Camera Feed (Testing)", f)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                time.sleep(0.5)
                continue
        ret, frame = cap.read()
        if not ret:
            time.sleep(1); continue
        preview = frame.copy()
        boxes = detect_faces_dnn(face_net, frame)
        for (x1,y1,x2,y2,_) in boxes:
            cv2.rectangle(preview, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("Camera Feed (Testing)", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[EXIT] quitting")
            break
        elif key == ord('p'):
            mpv_player.play_pause()
        elif key == ord('n'):
            # call next
            _ = route_next()
        elif key == ord('s'):
            _ = route_stop()
        elif key == ord('+') or key == ord('='):
            with state_lock:
                shared["volume"]=min(100, shared.get("volume",70)+5)
            mpv_player.set_volume(shared["volume"])
        elif key == ord('-') or key == ord('_'):
            with state_lock:
                shared["volume"]=max(0, shared.get("volume",70)-5)
            mpv_player.set_volume(shared["volume"])
        now = time.time()
        if now - last_capture < capture_interval:
            time.sleep(0.05)
            continue
        last_capture = now
        # full detection
        boxes = detect_faces_dnn(face_net, frame)
        print(f"[STEP] {len(boxes)} faces detected.")
        if not boxes:
            cleanup_temp(); continue
        ages=[]; cleanup_temp()
        ts = int(time.time())
        for i,(x1,y1,x2,y2,conf) in enumerate(boxes):
            face_img = frame[y1:y2, x1:x2]
            if face_img.size==0: continue
            path = os.path.join(TEMP_DIR, f"face_{ts}_{i}.jpg")
            cv2.imwrite(path, face_img)
            age_pred = predict_age(age_net, face_img)
            if age_pred is not None: ages.append(age_pred)
        if not ages:
            print("[WARN] no ages predicted"); continue
        avg_age = float(np.mean(ages))
        group = age_to_group(avg_age)
        print(f"[RESULT] Avg age: {avg_age:.1f} -> {group}")
        # fetch song
        video_id,title,artist = fetch_song_for_group(group)
        used_offline = False
        if video_id:
            yt_url = f"https://www.youtube.com/watch?v={video_id}"
            # try extract stream via yt-dlp
            try:
                with yt_dlp.YoutubeDL({'quiet': True,'format':'bestaudio'}) as ydl:
                    info = ydl.extract_info(yt_url, download=False)
                    stream = info.get('url') or yt_url
            except Exception as e:
                print("[YT-DLP] failed:", e); stream = None
            if stream:
                with state_lock:
                    shared["current_video_id"]=video_id
                    shared["current_title"]=title
                    shared["current_artist"]=artist
                    shared["offline_active"]=False
                    shared["playing"]=True
                # start mpv with IPC
                mpv_player.start(stream, is_local=False, title=title, artist=artist)
                wait_player_end_or_staff()
            else:
                used_offline=True
        else:
            used_offline=True
        if used_offline:
            track = choose_offline_track(shuffle=shared.get("shuffle",False))
            if track:
                with state_lock:
                    shared["current_video_id"]=track
                    shared["current_title"]=os.path.basename(track)
                    shared["current_artist"]="Local playlist"
                    shared["offline_active"]=True
                    shared["playing"]=True
                mpv_player.start(track, is_local=True, title=shared["current_title"], artist=shared["current_artist"])
                wait_player_end_or_staff()
            else:
                print("[FALLBACK] no offline tracks found")
                with state_lock:
                    shared["playing"]=False
    cap.release(); cv2.destroyAllWindows()

# ---------------- Run Flask + detection ----------------
def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    # Warn about pywin32 on Windows if missing
    if IS_WINDOWS and not HAVE_PYWIN32:
        print("[WARN] pywin32 not found. For best mpv control on Windows install pywin32: pip install pywin32")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    try:
        detection_loop()
    except KeyboardInterrupt:
        print("[EXIT] Keyboard interrupt")
        mpv_player.stop()
        os._exit(0)
