import tkinter as tk
from tkinter import filedialog
import cv2
import threading
from ultralytics import YOLO
from process_frame import process_frame_with_models

model_path = "worksafety.pt"  # İş güvenliği modelinin doğru yolu
model_path_yolov8n = "yolov8n.pt"  # Önceden eğitilmiş modelin yolu

#0=helmet 1=vest

# İlk model: insan tespiti
model_yolov8 = YOLO(model_path_yolov8n)

# İkinci model: iş güvenliği tespiti
model_worksafe = YOLO(model_path)  # Modelin doğru yolu kullanın


def open_camera():
    def start_camera():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = process_frame_with_models(frame)
                cv2.imshow("Pocessing Image", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    # Kamerayı ayrı bir thread'de çalıştırmak için:
    threading.Thread(target=start_camera).start()

def open_video_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = process_frame_with_models(frame)
                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

# Arayüz oluşturma
root = tk.Tk()
root.title("Video Arayüzü")
root.geometry("600x400")

# Butonlar
camera_button = tk.Button(root, text="Kamerayı Aç", command=open_camera)
camera_button.pack(pady=80)

video_button = tk.Button(root, text="Video Seç", command=open_video_file)
video_button.pack(pady=20)

# Arayüz başlatma
root.mainloop()
