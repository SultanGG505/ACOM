
import subprocess
import os

def open_nii_file_in_mricrogl(file_path, mricrogl_path):
    if not os.path.isfile(file_path):
        print(f"Файл {file_path} не найден.")
        return

    mricrogl_executable = os.path.join(mricrogl_path, 'MRIcroGL.exe')

    command = [mricrogl_executable, file_path, "-cm 4hot"]

    subprocess.run(command)

# Пример использования
nii_file_path = r'C:\Users\User\Documents\GitHub\ACOM\IZ_3\parsed_data\bad\1.2.643.5.1.13.13.12.2.77.8252.00000312101100040009080805131200.nii'
mricrogl_installation_path = r'C:\Users\User\Desktop\MRIcroGL_windows\MRIcroGL'  # Замените на ваш путь

open_nii_file_in_mricrogl(nii_file_path, mricrogl_installation_path)
