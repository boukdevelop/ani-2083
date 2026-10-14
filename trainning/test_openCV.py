import cv2

image = cv2.imread('trainning\dikoume-stephane-gitfut.png',cv2.IMREAD_COLOR)

cv2.imshow('exemple', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('dikoume-stephane-gitfut.png', image)