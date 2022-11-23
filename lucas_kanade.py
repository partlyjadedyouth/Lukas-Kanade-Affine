import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold
from tqdm import tqdm

def lucas_kanade_affine(T, I):
    """
    This functions is for getting an affine warp vector p
    which makes the difference between T and W(I;p) the smallest
    by Generalized Lucas-Kanade Algorithm.

    T : Template image
    I : Image to be warped
    p : Affine warp vector [dxx, dxy, dyx, dyy, dx, dy]
    dp : increment of p
    """
    # These codes are for calculating image gradient by using sobel filter
    Ix = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=5)  # do not modify this
    Iy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=5)  # do not modify this
    
    p = np.zeros(6) # initializer p
    
    ### START CODE HERE ###
    # [Caution] You should use only numpy and RectBivariateSpline functions
    # Never use opencv

    # Normalize gradients
    Ix /= 22.0
    Iy /= 22.0

    # Basic constants
    dp = np.ones(6).T
    I_row, I_col = I.shape
    T_row, T_col = T.shape

    # Coordinates
    x = np.arange(0, I_col)
    y = np.arange(0, I_row)
    
    # Warp function
    W = np.array([
        [1.0 + p[0], p[1], p[4]],
        [p[2], 1.0 + p[3], p[5]]
    ])

    # Warped coordinates
    x1_warped = W[0, 2]
    x2_warped = W[0, 0] * T_col + W[0, 1] * T_row + W[0, 2]
    y1_warped = W[1, 2]
    y2_warped = W[1, 0] * T_col + W[1, 1] * T_row + W[1, 2]
    c_warped = np.linspace(x1_warped, x2_warped, T_col)
    r_warped = np.linspace(y1_warped, y2_warped, T_row)
    c_mesh_warped, r_mesh_warped = np.meshgrid(c_warped, r_warped)

    # Compute warped image
    I_spline = RectBivariateSpline(y, x, I)
    I_warped = I_spline.ev(r_mesh_warped, c_mesh_warped)

    # Compute error image (n x 1)
    error_img = (T - I_warped).reshape(-1, 1)

    # Compute warped image gradient (n x 2)
    Ix_spline = RectBivariateSpline(y, x, Ix)
    Ix_warped = Ix_spline.ev(r_mesh_warped, c_mesh_warped)
    Iy_spline = RectBivariateSpline(y, x, Iy)
    Iy_warped = Iy_spline.ev(r_mesh_warped, c_mesh_warped)
    img_grad = np.hstack((Ix_warped.reshape(-1, 1), Iy_warped.reshape(-1, 1)))

    # Compute sum of (image gradient @ Jacobian) (n x 6)
    sum_of_grad_jacob = np.zeros((T_col * T_row, 6))
    for i in range(T_row):
        for j in range(T_col):
            grad_sample = np.array([img_grad[i * T_col + j]])
            jacob = np.array([
                [j, 0, i, 0, 1, 0],
                [0, j, 0, i, 0, 1]
            ])
            sum_of_grad_jacob[i * T_col + j] = grad_sample @ jacob
    
    # Compute Hessian (6 x 6)
    H = sum_of_grad_jacob.T @ sum_of_grad_jacob
    
    # Compute dp (6 x 1)
    dp = np.linalg.inv(H) @ sum_of_grad_jacob.T @ error_img

    # Update p
    for i in range(6):
        p[i] += dp[i, 0]
        
    ### END CODE HERE ###
    return p
    
def subtract_dominant_motion(It, It1):
    """
    This function is for getting an area of moving objects.
    Pixels where difference between It and W(It1;p) is over the threshold
    can be considered as part of moving objects.

    It : Image at t
    It1 : Image at t+1 (Template)
    motion_image : Difference of It and warped It1
    mask : Hysteresis-thresholded motion_image
    """
    ### START CODE HERE ###
    # [Caution] You should use only numpy and RectBivariateSpline functions
    # Never use opencv

    # Get p and warp function
    p = lucas_kanade_affine(It1, It)
    W = np.array([
            [1.0 + p[0], p[1], p[4]],
            [p[2], 1.0 + p[3], p[5]]
        ])
    
    # Coordinates
    It1_row, It1_col = It.shape
    x = np.arange(0, It1_col)
    y = np.arange(0, It1_row)
    x1_warped = W[0, 2]
    x2_warped = W[0, 0] * It1_col + W[0, 1] * It1_row + W[0, 2]
    y1_warped = W[1, 2]
    y2_warped = W[1, 0] * It1_col + W[1, 1] * It1_row + W[1, 2]
    c_warped = np.linspace(x1_warped, x2_warped, It1_col)
    r_warped = np.linspace(y1_warped, y2_warped, It1_row)
    c_mesh_warped, r_mesh_warped = np.meshgrid(c_warped, r_warped)

    # Spline and warped It1
    It1_spline = RectBivariateSpline(y, x, It1)
    It1_warped = It1_spline.ev(r_mesh_warped, c_mesh_warped)

    # Compute motion image
    motion_image = np.abs(It - It1_warped)

    ### START CODE HERE ###
    th_hi = 0.3 * 256 # you can modify this
    th_lo = 0.1 * 256 # you can modify this
    
    mask = apply_hysteresis_threshold(motion_image, th_lo, th_hi)
    
    return mask

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    data_dir = 'data/motion'
    video_path = 'results/motion_best.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))
    img_path = os.path.join(data_dir, "{}.jpg".format(0))
    It = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    for i in tqdm(range(1, 61)):
    # for i in tqdm(range(1, 20)):
        img_path = os.path.join(data_dir, "{}.jpg".format(i))
        It1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        It_clone = It.copy()
        mask = subtract_dominant_motion(It, It1)
        It_clone = cv2.cvtColor(It_clone, cv2.COLOR_GRAY2BGR)
        It_clone[mask, 2] = 255 # set color red for masked pixels
        out.write(It_clone)
        It = It1
    out.release()