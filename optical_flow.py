import numpy as np
import cv2


def optical_flow(old_frame, new_frame, window_size, min_quality=0.01):

    max_corners = 10000
    min_distance = 0.1
    cv_features = cv2.goodFeaturesToTrack(old_frame, max_corners, min_quality, min_distance)

    w = int(window_size/2)

    #Convolve to get gradients w.r.to X, Y and T dimensions
    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(old_frame, -1, kernel_x)              #Gradient over X
    fy = cv2.filter2D(old_frame, -1, kernel_y)              #Gradient over Y
    ft = cv2.filter2D(new_frame, -1, kernel_t) - cv2.filter2D(old_frame, -1, kernel_t)  #Gradient over Time


    u = np.zeros(old_frame.shape)
    v = np.zeros(old_frame.shape)

    for feature in cv_features:        
            j, i = feature.ravel()		
            i, j = int(i), int(j)		

            I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(I_t, (I_t.shape[0],1))
            A = np.vstack((I_x, I_y)).T

            U = np.matmul(np.linalg.pinv(A), b)   

            u[i,j] = U[0][0]
            v[i,j] = U[1][0]
 
    return (u,v)
