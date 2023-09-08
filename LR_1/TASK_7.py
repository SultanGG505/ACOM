# Задание 7 (самостоятельно) Отобразить информацию с вебкамеры,
# записать видео в файл, продемонстрировать видео.

import cv2
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Ошибка при открытии вебкамеры.")
    exit()


output_file = "task_7.avi"  # Имя выходного файла
frame_width = int(cap.get(3))  # Ширина кадра
frame_height = int(cap.get(4))  # Высота кадра
fps = 30.0  # Количество кадров в секунду


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:

    ret, frame = cap.read()

    if not ret:
        print("Ошибка при захвате кадра.")
        break


    rect_width = 100
    rect_height = 300
    top_left_x1 = (frame_width - rect_width) // 2
    top_left_y1 = (frame_height - rect_height) // 2
    bottom_right_x1 = top_left_x1 + rect_width
    bottom_right_y1 = top_left_y1 + rect_height
    cv2.rectangle(frame, (top_left_x1, top_left_y1), (bottom_right_x1, bottom_right_y1), (0, 0, 255), 2)

    rect_width = 300
    rect_height = 100
    top_left_x2 = (frame_width - rect_width) // 2
    top_left_y2 = (frame_height - rect_height) // 2
    bottom_right_x2 = top_left_x2 + rect_width
    bottom_right_y2 = top_left_y2 + rect_height
    cv2.rectangle(frame, (top_left_x2, top_left_y2), (bottom_right_x2, bottom_right_y2), (0, 0, 255), 2)

    out.write(frame)


    cv2.imshow("Webcam Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()