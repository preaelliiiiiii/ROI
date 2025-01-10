import cv2
import json
import numpy as np
from tkinter import *
from tkinter import filedialog as fd
from tkinter import messagebox
from tkinter.simpledialog import askstring

data_point = []
root = Tk()
root.config(background="lightyellow")
Label(root, text="ROI Point", font="Normal 50 bold", bg="lightyellow").grid(column=0, row=0, padx=20, pady=20)

def reset_roi():
    global data_point
    data_point = []
    messagebox.showinfo("Informasi", "ROI telah direset")

def POINTS(event, x, y, flags, param):
    global data_point
    if event == cv2.EVENT_LBUTTONDOWN:  # Klik kiri untuk menambahkan titik
        point = [x, y]
        data_point.append(point)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Klik kanan untuk reset ROI
        reset_roi()

def process_video_stream(cap):
    cv2.namedWindow('ROI')
    cv2.setMouseCallback('ROI', POINTS)
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (1280, 720))
        image = img.copy()
        if not len(data_point) == 0:
            for i in data_point:
                cv2.circle(image, (i[0], i[1]), 7, (255, 0, 0), -1)
        cv2.polylines(image, [np.array(data_point, np.int32)], True, (0, 0, 255), 2)
        cv2.imshow("ROI", image)
        keycode = cv2.waitKey(30)  # Menampilkan frame setiap 30 ms
        if cv2.getWindowProperty("ROI", cv2.WND_PROP_VISIBLE) < 1:
            break
        if keycode == ord('s'):
            save_file = fd.asksaveasfilename(defaultextension=".json", filetypes=[("Json Files", "*.json")])
            if save_file:
                with open(save_file, 'w') as file:
                    json.dump(data_point, file)
                messagebox.showinfo("Informasi", f"Data telah tersimpan sebagai {save_file}")
            break
        elif keycode == ord('q'):  # Tombol 'q' untuk keluar
            break
    cap.release()
    cv2.destroyAllWindows()

def pilih_file():
    global data_point
    file_name = fd.askopenfilename(filetypes=[("Gambar", ("*.jpg", "*.jpeg", "*.png")), ("Video", ("*.mp4", "*.avi"))])
    if not file_name == "":
        if file_name.endswith(".mp4") or file_name.endswith(".avi"):
            cap = cv2.VideoCapture(file_name)
            process_video_stream(cap)
        else:
            img = cv2.imread(file_name)
            if img is not None:
                process_image(img)

def process_image(img):
    cv2.namedWindow('ROI')
    cv2.setMouseCallback('ROI', POINTS)
    while True:
        img = cv2.resize(img, (1280, 720))
        image = img.copy()
        if not len(data_point) == 0:
            for i in data_point:
                cv2.circle(image, (i[0], i[1]), 7, (255, 0, 0), -1)
        cv2.polylines(image, [np.array(data_point, np.int32)], True, (0, 0, 255), 2)
        cv2.imshow("ROI", image)
        keycode = cv2.waitKey(1)
        if cv2.getWindowProperty("ROI", cv2.WND_PROP_VISIBLE) < 1:
            break
        if keycode == ord('s'):
            save_file = fd.asksaveasfilename(defaultextension=".json", filetypes=[("Json Files", "*.json")])
            if save_file:
                with open(save_file, 'w') as file:
                    json.dump(data_point, file)
                messagebox.showinfo("Informasi", f"Data telah tersimpan sebagai {save_file}")
            break
    cv2.destroyAllWindows()

def input_rtsp_hls():
    global data_point
    link = askstring("Input RTSP/HLS", "Masukkan link RTSP atau HLS:")
    if link:
        cap = cv2.VideoCapture(link)
        if cap.isOpened():
            process_video_stream(cap)
        else:
            messagebox.showerror("Error", "Tidak dapat membuka stream video dari link ini.")

def input_webcam():
    global data_point
    cap = cv2.VideoCapture(0)  # Buka webcam
    if cap.isOpened():
        process_video_stream(cap)
    else:
        messagebox.showerror("Error", "Tidak dapat membuka kamera.")

# Tombol untuk memilih gambar/video
b1 = Button(root, text="Cari Gambar/Video", font="Normal 30", bd=10, relief=RIDGE, command=pilih_file, activebackground='blue')
b1.grid(column=0, row=1, pady=20, padx=30)

# Tombol untuk input dari RTSP/HLS
b2 = Button(root, text="Input RTSP/HLS", font="Normal 30", bd=10, relief=RIDGE, command=input_rtsp_hls, activebackground='green')
b2.grid(column=0, row=2, pady=20, padx=30)

# Tombol untuk input dari webcam
b3 = Button(root, text="Buka Webcam", font="Normal 30", bd=10, relief=RIDGE, command=input_webcam, activebackground='red')
b3.grid(column=0, row=3, pady=20, padx=30)

root.mainloop()
