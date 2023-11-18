import cv2
import numpy as np

# Создание изображения с заливкой
image = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (200, 200), 255, -1)
cv2.rectangle(image, (150, 150), (300, 300), 255, -1)

# Создание маски для области пересечения
mask = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(mask, (150, 150), (200, 200), 255, -1)

# Удаление заливки с области пересечения
result = cv2.subtract(image, mask)

# Визуализация результатов
cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()