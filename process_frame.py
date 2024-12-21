import cv2
import imutils
from ultralytics import YOLO
import torch

model_path = "worksafety.pt"  # worksafety model path
model_path_yolov8n = "yolov8n.pt"  # yolov8n model path

#yolov8n model with gpu
human_model = YOLO(model_path_yolov8n).to("cuda")
   
# worksafety model with gpu
worksafety_model = YOLO(model_path).to("cuda")


def process_frame_with_models(frame):

    # resize frame
    frame = imutils.resize(frame, width=640)

    # detect person
    results = human_model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        vest = False
        helmet = False
        threshold=0.5

        if score > threshold:
            if int(class_id) == 0:  # person class (class_id 0, COCO data set)
                person_img = frame[int(y1):int(y2), int(x1):int(x2)]  # crop the detect person
                saferesults = worksafety_model(person_img)[0]  # worksafe detect

                helmet_ok = False
                vest_ok = False

                for saferesult in saferesults.boxes.data.tolist():
                    sx1, sy1, sx2, sy2, sscore, sclass_id = saferesult

                    if sscore > threshold:
                        if int(sclass_id) == 0:  # helmet sınıfı
                            helmet_ok = True
                        if int(sclass_id) == 1:  # vest sınıfı
                            vest_ok = True

                # show results on screen
                font_scale = (x2 - x1) / 200
                if helmet_ok:
                    cv2.putText(frame, "Helmet: OK", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                    helmet = True
                else:
                    cv2.putText(frame, "Helmet: Not Detected", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

                if vest_ok:
                    cv2.putText(frame, "Vest: OK", (int(x1), int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                    vest = True
                else:
                    cv2.putText(frame, "Vest: Not Detected", (int(x1), int(y1) + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)

                # red or green rectangle for person
                if helmet and vest:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    return frame