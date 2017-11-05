import cv2
import numpy as np

# Load the images
img =cv2.imread('mccaws_90.jpg')

# Convert them to grayscale
imgg =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# SURF extraction
surf = cv2.xfeatures2d.SURF_create(400)
surf.setHessianThreshold(5000)
kp, descritors = surf.detectAndCompute(imgg,None)

# Setting up samples and responses for kNN
samples = np.array(descritors)

print(len(samples))

responses = np.arange(len(kp),dtype = np.float32)

# kNN training
#knn = cv2.KNearest()
knn = cv2.ml.KNearest_create()
knn.train(samples,cv2.ml.ROW_SAMPLE,responses)

# Now loading a template image and searching for similar keypoints
template = cv2.imread('mccaw.PNG')
templateg= cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
keys,desc = surf.detectAndCompute(templateg,None)

for h,des in enumerate(desc):
    #print(len(des))
    des = np.array(des,np.float32).reshape((1,64))
    retval, results, neigh_resp, dists = knn.findNearest(des,1)
    res,dist =  int(results[0][0]),dists[0][0]

    if dist<0.1: # draw matched keypoints in red color
        color = (0,0,255)
    else:  # draw unmatched in blue color
        print(dist)
        color = (255,0,0)

    #Draw matched key points on original image
    x,y = kp[res].pt
    center = (int(x),int(y))
    cv2.circle(img,center,5,color,-1)

    #Draw matched key points on template image
    x,y = keys[h].pt
    center = (int(x),int(y))
    cv2.circle(template,center,2,color,-1)

cv2.imshow('img',img)
cv2.imshow('tm',template)
cv2.waitKey(0)
cv2.destroyAllWindows()