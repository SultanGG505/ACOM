import gzip
import os
import shutil



import dicom2nifti



def unpack_single_gzip_in_folder(folder_path):

    files = os.listdir(folder_path)


    gz_files = [file for file in files if file.endswith('.gz')]


    if len(gz_files) == 1:
        gz_file_path = os.path.join(folder_path, gz_files[0])


        output_file_path = os.path.splitext(gz_file_path)[0]

        with gzip.open(gz_file_path, 'rb') as f_in, open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_file_path)
        print(f"Распаковано: {gz_file_path} -> {output_file_path}")
    else:
        print("Ошибка: Не удалось определить единственный файл .gz в указанной папке.")


for x in os.listdir(r"D:\data\bad\500_600_studies"):
    dicom2nifti.convert_directory(os.path.join(r"D:\data\bad\500_600_studies", x,
                                   os.listdir(os.path.join(
                                                       r"D:\data\bad\500_600_studies",
                                                       x))[0]), os.path.join(r"D:\data\bad\500_600_studies", x))
    unpack_single_gzip_in_folder(os.path.join(r"D:\data\bad\500_600_studies", x))


for x in os.listdir(r"D:\data\bad\600_700_studies"):
    dicom2nifti.convert_directory(os.path.join(r"D:\data\bad\600_700_studies", x,
                                   os.listdir(os.path.join(
                                                       r"D:\data\bad\600_700_studies",
                                                       x))[0]), os.path.join(r"D:\data\bad\600_700_studies", x))
    unpack_single_gzip_in_folder(os.path.join(r"D:\data\bad\600_700_studies", x))

for x in os.listdir(r"D:\data\bad\700_800_studies"):
    dicom2nifti.convert_directory(os.path.join(r"D:\data\bad\700_800_studies", x,
                                   os.listdir(os.path.join(
                                                       r"D:\data\bad\700_800_studies",
                                                       x))[0]), os.path.join(r"D:\data\bad\700_800_studies", x))
    unpack_single_gzip_in_folder(os.path.join(r"D:\data\bad\700_800_studies", x))

for x in os.listdir(r"D:\data\good\0_100_studies"):
    dicom2nifti.convert_directory(os.path.join(r"D:\data\good\0_100_studies", x,
                                   os.listdir(os.path.join(
                                                       r"D:\data\good\0_100_studies",
                                                       x))[0]), os.path.join(r"D:\data\good\0_100_studies", x))
    unpack_single_gzip_in_folder(os.path.join(r"D:\data\good\0_100_studies", x))

for x in os.listdir(r"D:\data\good\100_200_studies"):
    dicom2nifti.convert_directory(os.path.join(r"D:\data\good\100_200_studies", x,
                                   os.listdir(os.path.join(
                                                       r"D:\data\good\100_200_studies",
                                                       x))[0]), os.path.join(r"D:\data\good\100_200_studies", x))
    unpack_single_gzip_in_folder(os.path.join(r"D:\data\good\100_200_studies", x))

for x in os.listdir(r"D:\data\good\200_300_studies"):
    dicom2nifti.convert_directory(os.path.join(r"D:\data\good\200_300_studies", x,
                                   os.listdir(os.path.join(
                                                       r"D:\data\good\200_300_studies",
                                                       x))[0]), os.path.join(r"D:\data\good\200_300_studies", x))
    unpack_single_gzip_in_folder(os.path.join(r"D:\data\good\200_300_studies", x))

for x in os.listdir(r"D:\data\good\300_400_studies"):
    dicom2nifti.convert_directory(os.path.join(r"D:\data\good\300_400_studies", x,
                                   os.listdir(os.path.join(
                                                       r"D:\data\good\300_400_studies",
                                                       x))[0]), os.path.join(r"D:\data\good\300_400_studies", x))
    unpack_single_gzip_in_folder(os.path.join(r"D:\data\good\300_400_studies", x))

for x in os.listdir(r"D:\data\good\400_500_studies"):
    dicom2nifti.convert_directory(os.path.join(r"D:\data\good\400_500_studies", x,
                                   os.listdir(os.path.join(
                                                       r"D:\data\good\400_500_studies",
                                                       x))[0]), os.path.join(r"D:\data\good\400_500_studies", x))
    unpack_single_gzip_in_folder(os.path.join(r"D:\data\good\400_500_studies", x))