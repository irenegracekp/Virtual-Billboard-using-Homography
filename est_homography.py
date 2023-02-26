import numpy as np


def est_homography(X, X_prime):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by X_prime. In this assignment, X are the coordinates of the
    four corners of the soccer goal while X_prime are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        X_prime: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. X_prime ~ H*X

    """

    ##### STUDENT CODE START #####
    ax = np.zeros((4,9))
    ay = np.zeros((4,9))
    A = np.zeros((8,9))
    k = 0
    for i in range(4):
        ax[i] = [-X[i][0], -X[i][1], -1, 0, 0, 0, X[i][0]*X_prime[i][0], X[i][1]*X_prime[i][0], X_prime[i][0]]
        ay[i] = [0, 0, 0, -X[i][0], -X[i][1], -1, X[i][0]*X_prime[i][1], X[i][1]*X_prime[i][1], X_prime[i][1]]
        A[k] = ax[i]
        k += 1
        A[k] = ay[i]
        k += 1

    [U, S , Vt ] = np.linalg.svd(A) 

    h = Vt[-1]
    H = np.reshape(h, (3,3))

    ##### STUDENT CODE END #####

    return H
