import cv2
import numpy as np

filename = 'pattern.png'
img = cv2.imread(filename)
img2 = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

dst_max = dst.max()

print(dst_max)
#print(dst)


#img[dst>0.4*dst.max()]=[0,0,255]

print(len(dst))

for row in range(len(dst)):
    for col in range(len(dst[row])):
        if dst[row][col] > 0.4*dst_max:
            cv2.circle(img, (int(col),int(row)), 5, (0,0,255), 2)


goodcorners = cv2.goodFeaturesToTrack(gray,95,0.005,10)

print(len(goodcorners))
print(goodcorners[0][0][0])

# for pt in goodcorners:
#     cv2.circle(img2, (int(pt[0][0]), int(pt[0][1])), 10, (255, 0, 255), 2)


cv2.imshow('dst',img)
#cv2.imshow('img2',img2)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()