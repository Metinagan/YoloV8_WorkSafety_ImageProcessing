import cv2
import imutils
import torch
from ultralytics import YOLO

video_path = "test5.mp4"  # Video dosyasının yolu
model_path = "worksafety.pt"  # İş güvenliği modelinin doğru yolu
model_path_yolov8n = "human.pt"  # Önceden eğitilmiş modelin yolu

#0=helmet 1=vest

# Video açma
cap = cv2.VideoCapture(video_path)

# İlk model: insan tespiti
model_yolov8 = YOLO(model_path_yolov8n).to("cuda")

# İkinci model: iş güvenliği tespiti
model_worksafe = YOLO(model_path).to("cuda")  # Modelin doğru yolu kullanın
threshold = 0.5

while cap.read():
    ret, img = cap.read()  # Kareyi oku
    if not ret:
        break  # Eğer video sonlanmışsa döngüyü kır

    img = imutils.resize(img, width=720)

    # İnsan tespiti yap
    results = model_yolov8(img)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        vest = False
        helmet = False
        
        if score > threshold:
            if int(class_id) == 0:  # İnsan sınıfı (class_id 0, COCO veri seti)
                #cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (225, 225, 225), 2)

                # İnsan tespiti yapılan bölgeyi iş güvenliği modeline ver
                person_img = img[int(y1):int(y2), int(x1):int(x2)]  # Tespit edilen kişiyi kırp
                saferesults = model_worksafe(person_img)[0]  # İş güvenliği tespiti

                # Kask ve yelek durumu
                helmet_ok = False
                vest_ok = False

                for saferesult in saferesults.boxes.data.tolist():
                    sx1, sy1, sx2, sy2, sscore, sclass_id = saferesult
                    
                    if sscore > threshold:
                        if int(sclass_id) == 0:  # Kask sınıfı (class_id 0)
                            helmet_ok = True
                        if int(sclass_id) == 1:  # Yelek sınıfı (class_id 1)
                            vest_ok = True

                # İnsan üstüne "Kask: OK" ve "Yelek: OK" yazdır
                font_scale = (x2 - x1) / 200  # Font boyutunu insanın yüksekliğine göre ayarla (y2 - y1)
                if helmet_ok:
                    cv2.putText(img, "Helmet: OK", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                    helmet = True
                else:
                    cv2.putText(img, "Helmet: Not Detected", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

                if vest_ok:
                    cv2.putText(img, "Vest: OK", (int(x1), int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                    vest = True
                else:
                    cv2.putText(img, "Vest: Not Detected", (int(x1), int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                    
                #kask ve yelek aynı anda kullanılıyorsa    
                if helmet==True and vest==True:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                else:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    

    cv2.imshow('Processed Frame', img)

    # Çıkmak için ESC tuşu
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
