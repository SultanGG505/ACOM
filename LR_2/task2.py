# Задание 2 Применить фильтрацию изображения с помощью команды
# inRange и оставить только красную часть, вывести получившееся изображение
# на экран(treshold), выбрать красный объект и потестировать параметры
# фильтрации, подобрав их нужного уровня.

import cv2
import numpy as np

image = cv2.imread( r'..\media\2.jpg')

while True:

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255]) # оттенок насыщенность яркость(значение)

    mask = cv2.inRange(hsv, lower_red, upper_red)
    onlyRed_frame = cv2.bitwise_and(image, image, mask=mask)

    combined = np.hstack([image, onlyRed_frame])

    cv2.imshow('Red Filtered Image', combined)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break


cv2.destroyAllWindows()