import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


os.environ["EGL_PLATFORM"] = "surfaceless" # Error Prevention untuk Linux

# Folder dataset
DATA_PATH = os.path.join('dataset')
# 60 frames per-video
video_frames = 60

# Input Label
action_name = input("Masukkan nama aksi/label (contoh: hello) \t: ").strip()
try:
    num_videos_to_collect = int(input("Jumlah video yang ingin direkam \t\t: "))
except ValueError:
    print("Input tidak valid, mengatur ke default (1 video).")
    num_videos_to_collect = 1

# Membuat folder/label jika belum ada
action_path = os.path.join(DATA_PATH, action_name)
os.makedirs(action_path, exist_ok=True)

existing_videos = [int(f) for f in os.listdir(action_path) if f.isdigit()]
start_sequence = max(existing_videos) + 1 if existing_videos else 0
end_sequence = start_sequence + num_videos_to_collect

print(f"\nSequence dimulai dari folder '{start_sequence}' hingga '{end_sequence - 1}'")

for sequence in range(start_sequence, end_sequence):
    os.makedirs(os.path.join(action_path, str(sequence)), exist_ok=True)

# 3. Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='handlandmarker/hand_landmarker.task',
                             delegate=BaseOptions.Delegate.GPU),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2) 

def extract_keypoints(results):
    if results and getattr(results, "hand_landmarks", None):
        landmarks_list = []
        for hand in results.hand_landmarks:
            for landmark in hand:
                landmarks_list.extend([landmark.x, landmark.y, landmark.z])
        # Standarisasi array ke 126 elemen (2 tangan x 21 titik x 3 koordinat XYZ)
        arr = np.array(landmarks_list)
        if arr.shape[0] < 126:
            arr = np.concatenate([arr, np.zeros(126 - arr.shape[0])])
        return arr[:126]
    else:
        return np.zeros(21 * 3 * 2) 

# 4. Pengambilan Video

# Mintalah input Enter DULU sebelum menyalakan kamera
input(f"Tekan Enter untuk mulai merekam {num_videos_to_collect} video untuk '{action_name}'...")

# Kamera baru dinyalakan SETELAH Enter ditekan agar tidak terjadi timeout/buffer overflow
cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # Coba pakai (0) saja tanpa cv2.CAP_V4L2 jika masih ada error

with HandLandmarker.create_from_options(options) as landmarker:
    if not cap.isOpened():
        print("Cannot open camera")
    else:
        start_time = time.time()  # Waktu mulai untuk timestamp video
        # Loop sekuens video
        for sequence in range(start_sequence, end_sequence):
            for frame_num in range(video_frames):
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Camera/Frame error.")
                    break
                
                # Proses frame dengan MediaPipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Hitung timestamp dalam milidetik (harus terus bertambah)
                frame_timestamp_ms = int((time.time() - start_time) * 1000)
                
                # Gunakan detect_for_video() dan masukkan timestamp-nya
                results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                keypoints = extract_keypoints(results)
                
                # Tampilan UI di layar
                if frame_num == 0: 
                    cv2.putText(frame, 'BERSIAP...', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Collecting [{action_name}] Seq: {sequence}', (15,20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    cv2.waitKey(1000) # Jeda 1 detik sebelum video baru mulai
                else: 
                    cv2.putText(frame, f'Collecting [{action_name}] Seq: {sequence}', (15,20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                
                # Menyimpan Koordinat Landmark ke folder
                npy_path = os.path.join(action_path, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Perekaman dihentikan paksa.")
                    break

        print("\nPerekaman Selesai!")

cap.release()
cv2.destroyAllWindows()