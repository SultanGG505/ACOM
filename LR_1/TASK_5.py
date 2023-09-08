# Задание 5 Прочитать изображение, перевести его в формат HSV.
# Вывести на экран два окна, в одном изображение в формате HSV, в другом –
# исходное изображение.

import cv2

image_path = r'..\media\1.jpg'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Чтение в оттенках серого

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Original", img)
cv2.imshow("HSV", hsv)
cv2.waitKey(0)
