import numpy as np
import cv2
import time
import json
from sort import *
import random
from datetime import datetime
import csv

# Fungsi untuk meload ROI dari file JSON
def load_roi_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Inisialisasi tracker dan warna
mot_tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3)
warna = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(50)]
cap = cv2.VideoCapture(r"kereta_tampakbelakang.mp4") #input video/livestream source

net = cv2.dnn.readNetFromONNX(r"best.onnx") #input modelnya disini
classes = ["Orang"]

# Load ROI dari file JSON
roi_file = "ROI1.json"  # Input file parimeter/ROI
area = load_roi_from_json(roi_file)

# Set untuk menyimpan ID yang telah dihitung
counted_ids = set()

# Set untuk menyimpan semua ID yang pernah berada di dalam area
total_ids = set()

# Dictionary untuk menyimpan waktu masuk dan waktu total di ROI
entry_time = {}  # Menyimpan waktu masuk terbaru ke ROI
total_time = {}  # Menyimpan total waktu yang telah dihabiskan dalam ROI

# Membuka file CSV untuk menyimpan log deteksi
with open('detection_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Class", "Object ID", "Duration (seconds)"])

    try:
        while True:
            img = cap.read()[1]
            if img is None:
                break

            img = cv2.resize(img, (1280, 720))
            blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()[0]

            classes_ids = []
            confidences = []
            boxes = []
            rows = detections.shape[0]

            img_width, img_height = img.shape[1], img.shape[0]
            x_scale = img_width / 640
            y_scale = img_height / 640

            for i in range(rows):
                row = detections[i]
                confidence = row[4]
                if confidence > 0.2:
                    classes_score = row[5:]
                    ind = np.argmax(classes_score)
                    if classes_score[ind] > 0.2:
                        classes_ids.append(ind)
                        confidences.append(confidence)
                        cx, cy, w, h = row[:4]
                        x1 = int((cx - w / 2) * x_scale)
                        y1 = int((cy - h / 2) * y_scale)
                        width = int(w * x_scale)
                        height = int(h * y_scale)
                        box = np.array([x1, y1, width, height])
                        boxes.append(box)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
            hasil = []
            masuk_poly = 0

            for i in indices:
                x1, y1, w, h = boxes[i]
                cx = int(x1 + (w // 2))
                cy = int(y1 + (h // 2))
                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 1:
                    masuk_poly += 1
                    label = classes[classes_ids[i]]
                    conf = confidences[i]
                    hasil.append(np.array([x1, y1, x1 + w, y1 + h, conf, 0]))

            if masuk_poly > 0:
                track_bbs_ids = mot_tracker.update(np.array(hasil))
            else:
                track_bbs_ids = mot_tracker.update(np.empty((0, 5)))

            current_ids_in_roi = set()

            for i in range(len(track_bbs_ids.tolist())):
                coords = track_bbs_ids.tolist()[i]
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                obj_id = int(coords[4])
                cv2.rectangle(img, (x1, y1), (x2, y2), warna[obj_id % len(warna)], 2)
                current_ids_in_roi.add(obj_id)

                # Jika objek masuk kembali atau objek baru, catat waktu masuknya
                if obj_id not in entry_time:
                    entry_time[obj_id] = time.time()  # Catat waktu masuk baru
                    if obj_id not in total_time:
                        total_time[obj_id] = 0  # Jika objek baru, inisialisasi waktu total

                # Tampilkan ID di atas bounding box
                cv2.putText(img, str(obj_id), (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.6, (71, 150, 255), 2)

                # Tampilkan waktu total objek berada dalam ROI
                duration = int(total_time[obj_id] + (time.time() - entry_time[obj_id]))  # Potong koma
                cv2.putText(img, f"Time: {duration}s", (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

            # Periksa apakah ada objek yang keluar dari ROI
            for obj_id in list(entry_time.keys()):
                if obj_id not in current_ids_in_roi:
                    # Update total waktu di dalam ROI
                    total_time[obj_id] += time.time() - entry_time[obj_id]  # Tambahkan waktu terakhir di ROI

                    # Simpan data ke CSV
                    detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([detection_time, classes[0], obj_id, int(total_time[obj_id])])

                    # Hapus waktu masuk karena objek telah keluar dari ROI
                    del entry_time[obj_id]


            cv2.polylines(img, [np.array(area, np.int32)], True, (0, 0, 255), 2)
            cv2.imshow("VIDEO", img)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break

    except KeyboardInterrupt:
        pass  # Untuk menangani interupsi seperti Ctrl+C

    finally:
        # Saat program berhenti, simpan semua data objek yang masih dalam ROI ke CSV
        for obj_id in list(entry_time.keys()):
            total_time[obj_id] += time.time() - entry_time[obj_id]  # Tambahkan waktu terakhir di ROI

            # Simpan data ke CSV
            detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([detection_time, classes[0], obj_id, int(total_time[obj_id])])

        cap.release()
        cv2.destroyAllWindows()
