import os


def rename_files_in_folder(folder_path):
    # Проверяем, существует ли указанная папка
    if not os.path.exists(folder_path):
        print(f"Папка '{folder_path}' не существует.")
        return

    # Получаем список файлов в папке
    files = os.listdir(folder_path)

    # Переименовываем файлы с расширением ".nil" в ".nii"
    for file_name in files:
        if file_name.endswith(".nil"):
            # Формируем новое имя файла с расширением ".nii"
            new_name = os.path.splitext(file_name)[0] + ".nii"

            # Формируем полные пути к файлам
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)

            # Переименовываем файл
            os.rename(old_path, new_path)
            print(f"Файл '{file_name}' переименован в '{new_name}'.")


# Укажите путь к папке, в которой нужно произвести переименование
folder_path = r"C:\Users\User\Documents\GitHub\ACOM\IZ_3\parsed_data\bad"
rename_files_in_folder(folder_path)
folder_path = r"C:\Users\User\Documents\GitHub\ACOM\IZ_3\parsed_data\good"
rename_files_in_folder(folder_path)
