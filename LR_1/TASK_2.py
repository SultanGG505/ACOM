import cv2

cap = cv2.VideoCapture(r'..\media\lol.mp4')
# если указать 0, то будет вебка
# cap = cv2.VideoCapture(0)


# cv2.CAP_PROP_BRIGHTNESS - яркость (0-1)
# cv2.CAP_PROP_CONTRAST - контраст (0-1)
# cv2.CAP_PROP_SATURATION - насыщенность (0-1)
# cv2.CAP_PROP_HUE - оттенок (0-1)

cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
cap.set(cv2.CAP_PROP_CONTRAST, 0.8)

while True:
    # Захват кадра из видеопотока.
    ret, frame = cap.read()

    if not ret:
        print("Конец видео.")
        break

    # Отображение кадра на экране.
    resize = cv2.resize(frame, (640, 360))  # изменение расширения
    cv2.normalize(frame, frame, 0, 277, cv2.NORM_MINMAX)
    cv2.imshow("Video", resize)

    # Выход на q.
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Освобождение памяти
cap.release()
cv2.destroyAllWindows()

# # создадим объект VideoCapture для захвата видео
# cap = cv2.VideoCapture(r'C:\Users\AlmaZ\Videos\papka\11.mp4', cv2.CAP_ANY)
#
# # Если не удалось открыть файл, выводим сообщение об ошибке
# if cap.isOpened() == False:
#     print('Не возможно открыть файл')
#
# # Пока файл открыт
# while cap.isOpened():
#     # поочередно считываем кадры видео
#     fl, img = cap.read()
#     # если кадры закончились, совершаем выход
#     if img is None:
#         break
#     # выводим текущий кадр на экран
#     cv2.imshow("Cat", img)
#     # при нажатии клавиши "q", совершаем выход
#     if cv2.waitKey(25) == ord('q'):
#         break
#
# # освобождаем память от переменной cap
# cap.release()
# # закрываем все открытые opencv окна
# cv2.destroyAllWindows()

#


# cap = cv2.VideoCapture(0)
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# video_writer = cv2.VideoWriter("output2.mp4", fourcc, 25, (w, h))
# while True:
#     ret, img = cap.read()
#     cv2.imshow("camera", img)
#     if cv2.waitKey(25) == ord('q'): # Клавиша Esc
#         break
# cap.release()
# cv2.destroyAllWindows()


# hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
# cv2.namedWindow("Display window",cv2.WINDOW_KEEPRATIO)
# cv2.imshow("Display window", hsv)
# cv2.waitKey(0)
