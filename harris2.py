import cv2
import numpy as np

filename = 'simple.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.7*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

#print(corners)

# Now draw them
res = np.hstack((centroids,corners))
print(res)
res = np.int0(res)
#print(res)
# img[res[:,1],res[:,0]]=[0,0,255]
# img[res[:,3],res[:,2]] = [0,255,0]

for pt in res:
    cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), 2)
    cv2.circle(img, (int(pt[2]), int(pt[3])), 5, (0, 255, 0), 2)


#cv2.imwrite('subpixel5.png',img)

cv2.imshow('subpixel',img)
#cv2.imshow('img2',img2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()