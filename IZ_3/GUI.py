import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
import nibabel as nib
from scipy import ndimage

# Получение размеров экрана
def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # Скрыть основное окно Tkinter
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

# Определение размеров окна
screen_width, screen_height = get_screen_size()
window_width = screen_width // 2
window_height = screen_height // 2

# Загрузите вашу модель
model = keras.models.load_model("iz_3_image_3")

def read_nifti_file(filepath):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    min_value = -1000
    max_value = 400
    volume[volume < min_value] = min_value
    volume[volume > max_value] = max_value
    volume = (volume - min_value) / (max_value - min_value)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    volume = read_nifti_file(path)
    volume = normalize(volume)
    volume = resize_volume(volume)
    return volume

def predict_scan(filepath):
    scan = process_scan(filepath)
    scan = np.expand_dims(scan, axis=0)  # Добавление размерности пакета
    prediction = model.predict(scan)
    score = [1 - prediction[0][0], prediction[0][0]]
    class_names = ["normal", "abnormal"]
    result_text.set(
        "Модель уверена на {:.2f}% что CT-скан {}.".format(
            100 * score[1], class_names[int(np.round(prediction[0][0]))]
        )
    )

def choose_file():
    filepath = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])
    if filepath:
        predict_scan(filepath)
        display_image(filepath)

def display_image(filepath):
    img = Image.open(filepath)
    img = img.resize((300, 300))  # Размеры изображения в окне
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Создание основного окна Tkinter
root = tk.Tk()
root.title("Проверка CT-сканов")
root.geometry(f"{window_width}x{window_height}+{screen_width // 4}+{screen_height // 4}")

# Элементы управления
choose_button = tk.Button(root, text="Выбрать файл", command=choose_file)
choose_button.pack(pady=10)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Helvetica", 14))
result_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

# Запуск главного цикла Tkinter
root.mainloop()
