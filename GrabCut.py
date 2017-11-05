import numpy as np
import cv2
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)

img = cv2.imread(r'images\messi5.jpg')
mask = np.zeros(img.shape[:2],np.uint8)
print(mask)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# print(bgdModel)
rect = (50,50,450,290)

# cv2.rectangle(img,(50,50),(450,290),(0,255,0),2)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

print(mask)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

print(mask2)
print(bgdModel)
print(fgdModel)


img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()