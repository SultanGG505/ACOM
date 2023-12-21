import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")

img = cv2.imread('test1.png')
text = pytesseract.image_to_string(img, lang='rus+eng')
print(text)