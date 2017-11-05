import cv2
import numpy as np


img = cv2.imread(r'images\lightning.jpg',0)

ret,thresh = cv2.threshold(img,127,255,0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv2.moments(cnt)
print( M )

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

area = cv2.contourArea(cnt)

print('Area: ',area)

perimeter = cv2.arcLength(cnt,True)

print('perimeter: ',perimeter)

epsilon = 0.045*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)


img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#cv2.drawContours(img,[approx],-1,(0,0,255),thickness=2)

hull = cv2.convexHull(cnt, returnPoints=True)

cv2.drawContours(img,[hull],-1,(0,0,255),thickness=2)

#print(cv2.isContourConvex(cnt))

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

rect = cv2.minAreaRect(cnt)
#print(rect)
box = cv2.boxPoints(rect)
#print(box)
box = np.int0(box)
#print(box)
cv2.drawContours(img,[box],0,(255,0,0),2)

rows,cols = img.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

print([vx,vy,x,y])

cv2.circle(img,(x,y),30,(255,0,255),thickness=2)

lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
# cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

slope = vy/vx
intercept = y-slope*x

print(slope)
print(intercept)

leftx = 0
lefty = intercept
rightx = cols-1
righty = slope*rightx+intercept

print([leftx,lefty, rightx, righty])

cv2.line(img,(leftx,lefty),(rightx,righty),(0,255,0),2)


cv2.imshow('img3', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()