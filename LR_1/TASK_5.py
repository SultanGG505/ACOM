# Задание 5 Прочитать изображение, перевести его в формат HSV.
# Вывести на экран два окна, в одном изображение в формате HSV, в другом –
# исходное изображение.

import cv2

image_path = r'..\media\1.jpg'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Чтение в оттенках серого
resize = cv2.resize(img, (300, 300))

hsv = cv2.cvtColor(resize, cv2.COLOR_BGR2HSV)
hsv_res = cv2.resize(hsv, (300, 300))
cv2.imshow("Original", resize)
cv2.imshow("HSV", hsv_res)
cv2.waitKey(0)
