import cv2
import urllib.request
import numpy as np

req = urllib.request.urlopen('https://')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr,-1)
#print(img)

image_width = 500
image_height = int(image_width*(img.shape[0]/img.shape[1]))

print(image_width)
print(image_height)

img = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#img2 = cv2.medianBlur(img2,3)
#img2 = cv2.bilateralFilter(img2,3,25,25)
#img2[:, :, 0] = cv2.equalizeHist(img2[:, :, 0])
#img2[:, :, 1] = cv2.equalizeHist(img2[:, :, 1])
#img2[:, :, 0] = cv2.GaussianBlur(img2[:, :, 0],(9,9),0)


#img2[:, :, 2] = cv2.equalizeHist(img2[:, :, 2])

clahe1 = cv2.createCLAHE(clipLimit=1, tileGridSize=(5,5))
clahe2 = cv2.createCLAHE(clipLimit=1.250, tileGridSize=(25,25))
img2[:, :, 1] = clahe1.apply(img2[:, :, 1])
img2[:, :, 2] = clahe2.apply(img2[:, :, 2])
#img2[:, :, 1] = cv2.medianBlur(img2[:, :, 1],9)
#img2[:, :, 2] = cv2.bilateralFilter(img2[:, :, 2],3,25,25)
#img2[:, :, 1] = cv2.GaussianBlur(img2[:, :, 1],(9,9),0)

img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)

img3 = np.hstack((img, img2))

cv2.imshow('img3',img3)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()