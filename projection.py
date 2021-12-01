# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:15:12 2021

@author: harshitm
"""
import numpy as np
import cv2
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
    #......
    
    threeD_points = np.zeros((1, 32, 3), np.float32)
    
    #Reading the Image and converting it to grayscale to detect corners using findChessboardCorners function
    img = imread(imgname)
    gray_imgname = cvtColor(img, COLOR_BGR2GRAY)
    found, corners = findChessboardCorners(gray_imgname, (4, 9), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 32, 0.01)
    corners2 = cornerSubPix(gray_imgname, corners, (11, 11), (-1, -1), criteria)

    # World coordinates of the Object in 3D
    threeD_points = np.array([[40 ,0, 40], [40, 0, 30], [40, 0, 20], [40, 0, 10],
                     [30, 0, 40], [30, 0, 30], [30, 0, 20], [30, 0, 10], 
                     [20, 0, 40], [20, 0, 30], [20, 0, 20], [20, 0, 10],
                     [10, 0, 40], [10, 0, 30], [10, 0, 20], [10, 0, 10],
                     [0, 10, 40], [0, 10, 30], [0, 10, 20], [0, 10, 10],
                     [0, 20, 40], [0, 20, 30], [0, 20, 20], [0, 20, 10],
                     [0, 30, 40], [0, 30, 30], [0, 30, 20], [0, 30, 10],
                     [0, 40, 40], [0, 40, 30], [0, 40, 20], [0, 40, 10]])
    
    # Removing extra points lying on z axis
    index = [32, 33, 34, 35, 36, 37, 38, 39]
    corners3 = np.delete(corners2, index)
    corners3 = corners3.reshape(32, 2)   # This is our final matrix for Image coordinates
    
    # Drawing the detected corners back on the image and saving the image
    img = drawChessboardCorners(img, (4, 9), corners3, found)
    cv2.imwrite('img.png', img)
    cv2.waitKey(500)

    # Defining the A matrix to compute the Ax=0 form 
    a_matrix = np.zeros((64, 12), np.float32)
    j=0
    for i in range(64):
        if( i % 2 == 0):
            a_matrix[i][0] = threeD_points[j][0] # X1
            a_matrix[i][1] = threeD_points[j][1] # Y1
            a_matrix[i][2] = threeD_points[j][2] # Z1
            a_matrix[i][3] = 1
            a_matrix[i][8] = -corners3[j][0]*threeD_points[j][0] # -x1*X1
            a_matrix[i][9] = -corners3[j][0]*threeD_points[j][1] # -x1*Y1
            a_matrix[i][10] = -corners3[j][0]*threeD_points[j][2] # -x1*Z1
            a_matrix[i][11] = -corners3[j][0] # -x1
            
        else:
            a_matrix[i][4] = threeD_points[j][0] # X1 
            a_matrix[i][5] = threeD_points[j][1] # Y1
            a_matrix[i][6] = threeD_points[j][2] # Z1
            a_matrix[i][7] = 1
            a_matrix[i][8] = -corners3[j][1]*threeD_points[j][0] # -y1*X1
            a_matrix[i][9] = -corners3[j][1]*threeD_points[j][1] # -y1*Y1
            a_matrix[i][10] = -corners3[j][1]*threeD_points[j][2] # -y1*Z1
            a_matrix[i][11] = -corners3[j][1] # -y1
            j+=1

    # Using SVD decomposition to solve the Ax=0 equation
    u, v, sh = np.linalg.svd(a_matrix)
    
    # Extracting m vector from the solution
    x_vector = sh[-1]

    """
    Calculating Lambda Scalar from preliminary 1. Since m_matrix = lambda* x_matrix.
    Lambda is calculated by relating thrid row of M matrix in prelimary 1 to rotation matrix.
    Since ri.ri = 1 and m21 = r21, m22 = r22, m23 = r23, we can say r22.r22 = 1, m22*m22 = 1, 
    (lambda*x)**2= 1, lambda = sqrt(1/x**2), and for all three values, we can say lambda = l2 norm of m21, ,22, m23
    """
    x_matrix = x_vector.reshape(3, 4)
    lambda_scalar = 1/(np.sqrt((x_matrix[2][0] * x_matrix[2][0]) + (x_matrix[2][1] * x_matrix[2][1]) + (x_matrix[2][2] * x_matrix[2][2])))

                       
    # Getting the M matrix now.
    m_matrix = lambda_scalar*x_matrix
    
    # defining m1, m2, m3, m4 vectors for calculation of intrinsic parameters
    m1 = m_matrix[0][0:3]
    m2 = m_matrix[1][0:3]
    m3 = m_matrix[2][0:3]
    m4 = m_matrix[0:3][1]

    # Calculating Ox first
    ox = m1.T.dot(m3)
    
    # Calculationg Oy
    oy = m2.T.dot(m3)
    
    # Calculating fx
    fx = np.sqrt(m1.T.dot(m1) - (ox*ox))
    
    # Calculating fy
    fy = np.sqrt(m2.T.dot(m2) - (oy*oy))
    
    # Creating Intrinsic parameter array
    intrinsic_params_array = [fx, fy, ox, oy]
    
    intrinsic_matrix = np.zeros((3, 3))
    intrinsic_matrix[0][0] = fx
    intrinsic_matrix[0][2] = ox
    intrinsic_matrix[1][1] = fy
    intrinsic_matrix[1][2] = oy
    intrinsic_matrix[2][2] = 1

    #Verifying the projection matrix on world point 0
    #w_point = threeD_points[0]
    #w_point = np.append(w_point, 1)
    #w_point = w_point.reshape(1, 4)
    #res = m_matrix.dot(w_point)
    #print(res)
    #print("\nImage Coordinates computes is : ")
    #print(res[0]/res[2], res[1]/res[2])
    #print(corners3[0])

    return intrinsic_params_array, True

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate(r'C:\Users\harsh\.spyder-py3\checkboard.png')
    print(intrinsic_params)
    print(is_constant)
