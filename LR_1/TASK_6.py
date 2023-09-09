# Задание 6 (самостоятельно) Прочитать изображение с камеры. Вывести
# в центре на экране Красный крест в формате, как на изображении. Указать
# команды, которые позволяют это сделать.

import cv2

img = cv2.imread(r'..\media\1.jpg')
height, width, _ = img.shape


rect_width = 100  # Ширина прямоугольника
rect_height = 300  # Высота прямоугольника


top_left_x = (width - rect_width) // 2
top_left_y = (height - rect_height) // 2
bottom_right_x = top_left_x + rect_width
bottom_right_y = top_left_y + rect_height
cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)

rect_width = 300  # Ширина прямоугольника
rect_height = 100  # Высота прямоугольника

top_left_x = (width - rect_width) // 2
top_left_y = (height - rect_height) // 2
bottom_right_x = top_left_x + rect_width
bottom_right_y = top_left_y + rect_height
cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 0, 255), 2)
ROI = img[top_left_y:top_left_y+rect_height, top_left_x:top_left_x+rect_width]
blur = cv2.GaussianBlur(ROI, (51,51), 0)
img[top_left_y:top_left_y+rect_height, top_left_x:top_left_x+rect_width] = blur
cv2.namedWindow('image', cv2.WINDOW_NORMAL)  # Окно с изменяемым размером
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
