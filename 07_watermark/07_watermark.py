import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

img = cv2.imread(r'C:\Users\USER\Desktop\develop\AI-study\07_watermark\image.jpg')

plt.figure(figsize=(16, 10))
plt.imshow(img[:, :, ::-1])
cv2.waitKey(5000)