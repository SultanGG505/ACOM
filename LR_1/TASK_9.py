# Задание 9 подключить телефон
import cv2

# cap = cv2.VideoCapture(r'..\media\lol.mp4')
# если указать 0, то будет вебка
cap = cv2.VideoCapture(0)


# cv2.CAP_PROP_BRIGHTNESS - яркость (0-1)
# cv2.CAP_PROP_CONTRAST - контраст (0-1)
# cv2.CAP_PROP_SATURATION - насыщенность (0-1)
# cv2.CAP_PROP_HUE - оттенок (0-1)

while True:
    # Захват кадра из видеопотока.
    ret, frame = cap.read()

    if not ret:
        print("Конец видео.")
        break

    # Отображение кадра на экране.
    # resize = cv2.resize(frame, (640, 360))
    # изменение расширения
    # cv2.normalize(frame, frame, 0, 277, cv2.NORM_MINMAX)
    cv2.imshow("Video", frame) #если mp4, то поменять frame на resize

    # Выход на q.
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Освобождение памяти
cap.release()
cv2.destroyAllWindows()


