import cv2
# lesture de l'image
img = cv2.imread('trainning/sun.png')
img = cv2.resize(img, (200,200))

print('image lue')

color = (255,255,255)

image = cv2.line(img,(196,2), (2,100), color, 20)

cv2.imshow('exemple', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('new_sun.png', image)