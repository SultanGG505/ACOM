import cv2
import numpy as np
from keras.models import load_model


model = load_model('../models/cnn_model_2.keras')


image_path = '../input_picture/1.jpg'
img_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


img_cv = cv2.resize(img_cv, (28, 28))

print('Размер тестируемого изображения:',img_cv.shape)

image = img_cv / 255.0

image = image.reshape(1, 28, 28, 1)


predictions = model.predict(image)

print(predictions)
predicted_percentages = predictions * 100
print("Предсказанные вероятности:")
for i, percentage in enumerate(predicted_percentages[0]):
    print(f"Цифра {i}: {percentage:.2f}%")


predicted_class = np.argmax(predictions)


print(f"Предсказанная цифра: {predicted_class}")
