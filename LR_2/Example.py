import cv2
import numpy as np
import time

# Инициализация видеозахвата с камеры
cap = cv2.VideoCapture(0)

# Установка разрешения
cap.set(3, 320)
cap.set(4, 240)

# Создание окна для настроек
cv2.namedWindow("Control", cv2.WINDOW_NORMAL)

# Начальные значения для трекбаров
iLowH = 170
iHighH = 179
iLowS = 150
iHighS = 255
iLowV = 60
iHighV = 255

# Создание трекбаров
cv2.createTrackbar("LowH", "Control", iLowH, 179)
cv2.createTrackbar("HighH", "Control", iHighH, 179)
cv2.createTrackbar("LowS", "Control", iLowS, 255)
cv2.createTrackbar("HighS", "Control", iHighS, 255)
cv2.createTrackbar("LowV", "Control", iLowV, 255)
cv2.createTrackbar("HighV", "Control", iHighV, 255)

iLastX = -1
iLastY = -1

# Чтение временного изображения и создание черного изображения для линий
ret, imgTmp = cap.read()
imgLines = np.zeros_like(imgTmp)

# Начало замера времени
start = time.time()
frames = 0

while True:
    # Чтение кадра с камеры
    ret, imgOriginal = cap.read()

    # Преобразование BGR в HSV
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    # Применение цветового фильтра
    lower_bound = np.array([iLowH, iLowS, iLowV])
    upper_bound = np.array([iHighH, iHighS, iHighV])
    imgThresholded = cv2.inRange(imgHSV, lower_bound, upper_bound)

    # Морфологическая обработка
    kernel = np.ones((5, 5), np.uint8)
    imgThresholded = cv2.erode(imgThresholded, kernel, iterations=1)
    imgThresholded = cv2.dilate(imgThresholded, kernel, iterations=1)
    imgThresholded = cv2.dilate(imgThresholded, kernel, iterations=1)
    imgThresholded = cv2.erode(imgThresholded, kernel, iterations=1)

    # Вычисление моментов
    moments = cv2.moments(imgThresholded)
    area = moments['m00']

    if area > 10000:
        posX = int(moments['m10'] / area)
        posY = int(moments['m01'] / area)

        if iLastX >= 0 and iLastY >= 0 and posX >= 0 and posY >= 0:
            cv2.line(imgLines, (posX, posY), (iLastX, iLastY), (0, 0, 255), 2)

        iLastX = posX
        iLastY = posY

    # Отображение изображений
    cv2.imshow("Thresholded Image", imgThresholded)
    imgOriginal = cv2.add(imgOriginal, imgLines)
    cv2.imshow("Original", imgOriginal)

    # Ожидание клавиши 'Esc' для выхода
    if cv2.waitKey(30) == 27:
        print("Esc key is pressed by the user")
        break

    frames += 1

# Завершение замера времени
end = time.time()
elapsed_time = end - start
fps = frames / elapsed_time
print(f"FPS: {fps:.2f}")

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()
