import cv2
from matplotlib import pyplot as plt
img = cv2.imread('C:\\Users\\Aravind Prasad\\Desktop\\MRICE\\Bacterial leaf blight\\Bacterial8.JPG')
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(grey, (5,5), 0)
edge = cv2.Canny(blur,100,200)
dil = cv2.dilate(edge,None)
plt.subplot(231),plt.imshow(img,cmap ='gray')
plt.title('original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(grey,cmap = 'gray')
plt.title('Grey Image'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(img,cmap ="gray")
plt.title('Blur Image'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(img,cmap ="gray")
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(img,cmap ="gray")
plt.title('Dilate Image'), plt.xticks([]), plt.yticks([])
plt.show()


