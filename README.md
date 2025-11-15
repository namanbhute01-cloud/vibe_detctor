Ambiance Detection & Auto-Music System (Face-Age + MPV Player + Offline Fallback + Preview Window)
This project automatically detects faces from a live camera stream, estimates average age, classifies the crowd into an age group, and plays YouTube music matched to the audience.
If internet or yt-dlp fails, the system automatically switches to offline local MP3 playlist fallback.

It includes:

✅ Real-time face detection
✅ Average age estimation
✅ Automatic music selection per age group
✅ MPV player for lightweight playback
✅ Preview window (camera + now playing)
✅ Buttons for music controls (Play / Pause / Stop / Skip / Volume Up / Down)
✅ Offline playlist fallback
✅ Flask API backend
✅ Auto face-capture and cleaned-up temp directory

📸 Features
🎯 Age Group Classification
Age	Group
0–17	Kids
18–35	Youth
36–55	Adults
56+	Seniors
🎵 Matching Auto-Playlist
YouTube playlist or offline music folder per group.

🎥 Camera Preview Window
Shows:
✔ Live RTSP/IP cam frames
✔ Detection status
✔ Current song playing

🎚️ Music Controls
Buttons using MPV JSON IPC:

Pause / Resume

Stop

Next song

Volume Up

Volume Down

📦 Requirements
Install Python 3.10+
https://www.python.org/downloads/

Install System Dependencies (Windows)
Install mpv (recommended portable build):
https://sourceforge.net/projects/mpv-player-windows/

Add MPV folder to PATH.

Install Python Packages
pip install opencv-python flask yt-dlp deepface numpy psutil
📁 Project Structure
ambiance_detection_system/
│
├── ambi_system.py            # Main script
├── playlists/
│     ├── kids/               # Local fallback MP3s
│     ├── youth/
│     ├── adults/
│     └── seniors/
│
├── temp_faces/               # Auto-created temp images
└── README.md
⚙️ Setup (Café / Restaurant Installation)
1. Connect the RTSP Camera
Most café CCTV cameras output RTSP.

Common RTSP format:

rtsp://username:password@CAMERA_IP:554/Streaming/Channels/101
Add it inside the script:

CAMERA_URL = "rtsp://user:pass@192.168.1.10/stream"
If using USB webcam:

CAMERA_URL = 0
🟦 Running the System
Inside the project folder:

python ambi_system.py
You will see:

A preview window

Console logs for detection

Web API running at

http://127.0.0.1:5000
🔊 Music Control Buttons (Preview Window)
The preview GUI includes:

Button	Function
▶ Play / Pause	Toggle playback
⏹ Stop	Stop MPV
⏭ Skip	Play next song
🔊 Vol +	Increase volume
🔉 Vol –	Decrease volume
Controls communicate with MPV over JSON IPC.

🎶 Offline Playlist Fallback
Place MP3 files like this:

playlists/
    kids/
        song1.mp3
        song2.mp3
    youth/
        ...
    adults/
        ...
    seniors/
        ...
If yt-dlp fails, the system plays random local MP3 automatically.

🧪 API Endpoints (Optional)
Endpoint	Method	Description
/status	GET	Get current detection & music info
/skip	POST	Skip song
/stop	POST	Stop playback
🛠 Troubleshooting
❗ MPV not found
Add the mpv.exe folder to PATH.

❗ yt-dlp errors (signature extract, SABR)
Use:

yt-dlp -U
If still failing → offline fallback will kick in.

❗ Blank camera preview
Check RTSP URL with VLC first.

🥤 Ideal For
Cafés

Restaurants

Lounges

Hotels

College canteens

Retail spaces with CCTV

Creates automatic smart ambiance music based on the customers present
