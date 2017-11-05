import numpy as np
import math

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-4

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])



t = np.array([[0.25581, -0.77351, 0.57986], [-0.85333, -0.46255, -0.24057], [0.45429, -0.43327, -0.77839]])

print(rotationMatrixToEulerAngles(t))






# t_trans = np.transpose(t)
#
#
# print(t)
# print()
# print(np.dot(t_trans,t))
#
#
# print()
# t_trans = np.transpose(t)
# shouldBeIdentity = np.dot(t_trans, t)
# I = np.identity(3, dtype = t.dtype)
# n = np.linalg.norm(I - shouldBeIdentity)
# print(n)