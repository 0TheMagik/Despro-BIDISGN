import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

## Script untuk merekam Video dataset BISINDO

os.environ["EGL_PLATFORM"] = "surfaceless" # Error Prevention untuk Linux

DATA_PATH = os.path.join('dataset')

num_frames = 90

action_name = input("Masukkan nama aksi/label (contoh: hello) \t: ").strip()
try:
    num_videos_to_collect = int(input("Jumlah video yang ingin direkam \t\t: "))
except ValueError:
    print("Input tidak valid, mengatur ke default (1 video).")
    num_videos_to_collect = 1
    
# Membuat folder/label jika belum ada
action_path = os.path.join(DATA_PATH, action_name)
os.makedirs(action_path, exist_ok=True)

cap = cv2.VideoCapture(1)
if (cap.isOpened() == False):
    print("Error opening video stream or file")


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

print("Frame Width: ", frame_width)
print("Frame Height: ", frame_height)
print("Frame Rate: ", frame_rate)

# Video Codec for .MP4
codec = cv2.VideoWriter_fourcc(*'mp4v')

# Cek file .mp4 yang sudah ada di folder untuk mendapatkan nomor urut terakhir
existing_videos = [
    int(f.split('.')[0]) 
    for f in os.listdir(action_path) 
    if f.endswith('.mp4') and f.split('.')[0].isdigit()
]
start_sequence = max(existing_videos) + 1 if existing_videos else 0
end_sequence = start_sequence + num_videos_to_collect

# Video Dimension
videodimension = (frame_width, frame_height)

# ... (Kode Video capture, codec, start end sequence tetap sama)

# --- 1. SET UP MOUSE CALLBACK SEBELUM LOOP RECORDING ---
cv2.namedWindow("Frame")
mouse_state = {"clicked": False}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state["clicked"] = True

cv2.setMouseCallback("Frame", mouse_callback)
# --------------------------------------------------------

# Mulai perulangan berdasarkan jumlah video yang ingin direkam
for sequence in range(start_sequence, end_sequence):
    filename = os.path.join(action_path, f"{sequence}.mp4")
    out = cv2.VideoWriter(filename, codec, frame_rate, videodimension)
    recording = False
    frames_recorded = 0 # Inisialisasi penghitung frame
    mouse_state["clicked"] = False # Reset status klik untuk sequence baru
    
    print(f"\n--- Persiapkan diri Anda untuk video urutan: {sequence} ---")
    print("Klik kiri pada window 'Frame' untuk mulai merekam, otomatis berhenti setelah 60 frame")
    
    while True:
        ret, frame = cap.read()

        if ret == False:
            print("Error retrieving frame")
            break

        # 1. Simpan frame ASLI dan hitung
        if recording:
            out.write(frame)
            frames_recorded += 1 # Tambahkan counter

        # 2. Buat Copy untuk tampilan layar
        display_frame = frame.copy() 
        
        # 3. Menambahkan indikator text & Hitungan Frame di Layar
        if recording:
            # Beritahu di layar progres framenya
            cv2.putText(display_frame, f'Recording Seq: {sequence} | Frame: {frames_recorded}/60', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Ubah teks indikator layar
            cv2.putText(display_frame, f'Ready Seq: {sequence} | Left Click to Start', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # 4. Tampilkan frame
        cv2.imshow("Frame", display_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # --- 2. UBAH LOGIKA TRIGGER ---
        # Mulai merekam jika window diklik (kiri)
        if mouse_state["clicked"] and not recording:
            print(f"Mulai merekam {sequence}.mp4...")
            recording = True
            mouse_state["clicked"] = False # Reset flag klik

        # if key == ord('s') and not recording:
        #     print(f"Mulai merekam {sequence}.mp4...")
        #     recording = True 
            
        # Otomatis berhenti merekam jika sudah 60 frame
        if recording and frames_recorded >= num_frames:
            print(f"Selesai merekam {sequence}.mp4! (60 / 60 frames)")
            break
            
        # Bisa juga berhenti paksa pakai q (opsional)
        elif key == ord('q'):
            print(f"Dihentikan paksa (Merekam {frames_recorded} frame)")
            break

    out.release() # Tutup akses file untuk video saat ini

print("\nSeluruh video berhasil direkam!")
cap.release()
cv2.destroyAllWindows()