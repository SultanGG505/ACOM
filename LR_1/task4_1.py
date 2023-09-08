import cv2

input_video_file = r'..\media\lol.mp4'
output_video_file = "output_video2.avi"
cap = cv2.VideoCapture(input_video_file)

if not cap.isOpened():
    print("Ошибка при открытии входного видео.")
    exit()

#  (размер кадра, частота кадров, кодек и другие параметры)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))


fourcc = cv2.VideoWriter_fourcc(*'XVID')
#(XVID для AVI)
out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))


if not out.isOpened():
    print("Ошибка при создании выходного видео.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    out.write(frame)

cap.release()
out.release()

print("Запись видео завершена.")