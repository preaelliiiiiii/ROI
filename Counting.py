import numpy as np
import cv2
import json
from sort import *
import random
from datetime import datetime
import csv
import os

# Fungsi untuk meload ROI dari file JSON
def load_roi_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Inisialisasi tracker dan warna
mot_tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3)
cap = cv2.VideoCapture(r"kereta_tampakdepan.mp4")

net = cv2.dnn.readNetFromONNX(r"best.onnx")
classes = ["Pelanggan"]

# Load ROI dari file JSON
roi_file = "ROI1.json"  # Ganti dengan nama file JSON yang Anda simpan
area = load_roi_from_json(roi_file)

# Set untuk menyimpan ID yang telah dihitung
counted_ids = set()

# Set untuk menyimpan semua ID yang pernah berada di dalam area
total_ids = set()

# Memuat logo dan mengubah ukurannya
logo_path = "logo.png"  # Sesuaikan dengan jalur file logo Anda
logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
logo = cv2.resize(logo, (100, 100))  # Ukuran logo yang lebih kecil

# Jika logo memiliki channel alpha, konversi ke 3 channel (RGB)
if logo.shape[2] == 4:
    b, g, r, a = cv2.split(logo)
    logo_rgb = cv2.merge((b, g, r))
    # Membuat mask untuk logo
    logo_mask = a
else:
    logo_rgb = logo
    logo_mask = np.ones(logo.shape[:2], dtype=np.uint8) * 255  # Mask untuk logo tanpa alpha

# Nama institusi
institusi_name = "Perimeter Protection Stasiun Peron"

# Membuka file CSV untuk menyimpan log deteksi
with open('detection_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Class", "Object ID"])

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
            label = classes[classes_ids[i]]
            conf = confidences[i]
            hasil.append(np.array([x1, y1, x1 + w, y1 + h, conf, 0]))

        if len(hasil) > 0:
            track_bbs_ids = mot_tracker.update(np.array(hasil))
        else:
            track_bbs_ids = mot_tracker.update(np.empty((0, 5)))

        for i in range(len(track_bbs_ids.tolist())):
            coords = track_bbs_ids.tolist()[i]
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            obj_id = int(coords[4])

            # Hitung posisi tengah dari bounding box
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Tentukan apakah objek di dalam ROI
            inside_roi = cv2.pointPolygonTest(np.array(area, np.int32), (center_x, center_y), False) >= 0

            if inside_roi:
                # Jika di dalam ROI, beri bounding box dengan warna merah dan isian transparan 50%
                overlay = img.copy()  # Salinan gambar untuk overlay
                
                # Menambahkan isian merah dengan opacity 50%
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 192), -1)  # Warna merah dengan isian
                cv2.addWeighted(overlay, 0.5, img, 1 - 0.5, 0, img)  # Alpha blending (50% transparansi)
                
                # Menambahkan outline untuk bounding box (warna merah dengan outline)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 192), 2)  # Outline merah
                
                # Tambah ID ke daftar ID yang dihitung
                if obj_id not in counted_ids:
                    counted_ids.add(obj_id)
                    total_ids.add(obj_id)
                    detection_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([detection_time, classes[0], obj_id])
            else:
                # Jika di luar ROI, tampilkan bounding box dengan outline ungu
                cv2.rectangle(img, (x1, y1), (x2, y2), (154, 91, 191), 2)


        # Tampilkan jumlah orang yang terdeteksi
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # Ukuran font
        font_thickness = 1
        font_color = (255, 255, 255)

        # Tampilkan jumlah orang dengan latar belakang transparan
        text = "Orang: " + str(len(total_ids))
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = 10
        text_y = img.shape[0] - 10

        overlay = img.copy()
        background_color = (255, 255, 255)
        alpha = 0.5
        cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), background_color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Menambahkan logo dengan transparansi dan 50% opacity di pojok kanan bawah
        logo_h, logo_w = logo.shape[:2]
        overlay = img.copy()
        position = (img.shape[1] - logo_w - 10, img.shape[0] - logo_h - 10)  # Posisi logo di pojok kanan bawah

        if logo.shape[2] == 4:  # Logo memiliki channel alpha
            # Memisahkan channel logo
            b, g, r, a = cv2.split(logo)
            
            # Membuat mask dan inverse mask dari alpha channel
            alpha_mask = (a / 255.0) * 0.5  # Mengatur alpha menjadi 50% opacity
            alpha_inv = 1.0 - alpha_mask
            
            # Memasukkan logo pada frame dengan alpha blending
            for c in range(0, 3):  # Mengiterasi setiap channel (BGR)
                img[position[1]:position[1] + logo_h, position[0]:position[0] + logo_w, c] = (
                    alpha_mask * logo[:, :, c] + alpha_inv * img[position[1]:position[1] + logo_h, position[0]:position[0] + logo_w, c]
                )

        # Menambahkan teks nama institusi di bagian atas gambar dengan latar belakang kotak hitam
        text = "Perimeter Protection Stasiun Peron"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Posisi teks (di pojok kiri atas dengan sedikit jarak dari pinggir)
        text_x = 10
        text_y = 30  # Sedikit di bawah batas atas gambar

        # Membuat kotak latar belakang hitam di sekitar teks
        cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + 5), 
                      (0, 0, 0), -1)

        # Menambahkan teks pada gambar
        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Menampilkan gambar
        cv2.imshow("Perimeter Protection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
