import tkinter as tk
from tkinter import filedialog
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NiiViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NII Viewer")

        self.file_path = None
        self.nii_data = None

        # Создаем интерфейс
        self.create_widgets()

    def create_widgets(self):
        # Кнопка для выбора файла
        self.btn_open = tk.Button(self.root, text="Открыть файл", command=self.open_file)
        self.btn_open.pack(pady=10)

        # Кнопка для отображения изображения
        self.btn_show = tk.Button(self.root, text="Показать изображение", command=self.show_image, state=tk.DISABLED)
        self.btn_show.pack(pady=10)

    def open_file(self):
        # Открываем диалоговое окно для выбора файла
        file_path = filedialog.askopenfilename(filetypes=[("NIfTI files", "*.nii")])

        if file_path:
            self.file_path = file_path
            self.nii_data = nib.load(file_path)
            self.btn_show.config(state=tk.NORMAL)

    def show_image(self):
        if self.nii_data is not None:
            # Извлекаем данные изображения
            img_data = self.nii_data.get_fdata()

            # Отображаем изображение с использованием matplotlib
            fig, ax = plt.subplots()
            ax.imshow(img_data[:, :, img_data.shape[2] // 2], cmap='gray')

            # Создаем холст Tkinter для встраивания графика Matplotlib
            canvas = FigureCanvasTkAgg(fig, master=self.root)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = NiiViewerApp(root)
    root.mainloop()
