import cv2
import numpy as np
import math

img = cv2.imread(r'images\25deg.png',0)


#mgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(img, 150, 200, 0)
# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

ret,thresh = cv2.threshold(img,127,255,0)
im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)

print(len(contours))

# cv2.imshow('img3', thresh)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()


# ret, thresh = cv2.threshold(img,127,255,0)
# im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cnt = contours[0]

cv2.drawContours(img,cnt,-1,(255,100,0), 2)

# #print(contours[1])
#cnt = contours[6]
#print(cnt)
M = cv2.moments(cnt)
print(M)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

mu_20_prime = M['m20']/M['m00'] - cx*cx
mu_02_prime = M['m02']/M['m00'] - cy*cy
mu_11_prime = M['m11']/M['m00'] - cx*cy


angle = 0.5 * math.atan(2*mu_11_prime/(mu_20_prime-mu_02_prime))

print('Angle from Image Moments:')
print(angle*180/math.pi)
print()

rect = cv2.minAreaRect(cnt)
print('rect: ',rect)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)
angle = rect[2]*math.pi/180

print('Angle from minAreaRect:')
print(angle*180/math.pi)
print()

line_length = 200
x1 = cx - ((line_length/2) * math.cos(angle))
y1 = cy - ((line_length/2) * math.sin(angle))
x2 = cx + ((line_length/2) * math.cos(angle))
y2 = cy + ((line_length/2) * math.sin(angle))

# print(cx)
# print(cy)
# print(x2)
# print(y2)

cv2.circle(img,(cx,cy),30,(255,0,255),thickness=2)
cv2.line(img,(int(x1),int(y1)), (int(x2),int(y2)), (150,50,100), thickness=2)


x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img,ellipse,(0,255,0),2)
print(ellipse)

angle = ellipse[2]#*math.pi/180

print('Angle from fitEllipse:')
print(angle)
print()


cv2.imshow('img3', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()