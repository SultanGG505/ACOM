import nibabel as nib
import matplotlib.pyplot as plt

# Замените 'path/to/your/file.nii' на путь к вашему файлу NIfTI
file_path = r'C:\Users\User\Documents\GitHub\ACOM\IZ_3\parsed_data\bad\1.2.643.5.1.13.13.12.2.77.8252.00000312101100040009080805131200.nii'

# Загрузка файла NIfTI
img = nib.load(file_path)

# Получение данных изображения
data = img.get_fdata()

# Вывод среза изображения (можно настроить срез по вашему усмотрению)
slice_number = 1
plt.imshow(data[:, :, :], cmap='gray')
plt.title('NIfTI Image')
plt.show()
