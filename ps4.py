"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2

# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale = 1.0 / 8.0)

 #   grad_x = normalize_and_scale(sobelx)

    return grad_x
#    raise NotImplementedError


def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """

    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale = 1.0 / 8.0)

  # grad_y = normalize_and_scale(sobely)
    return grad_y
#    raise NotImplementedError


def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    Ix = gradient_x(img_a)
    Iy = gradient_y(img_a)
    It = img_b - img_a

    Ix_s = cv2.boxFilter(Ix*Ix, cv2.CV_64F, (k_size, k_size))
    Iy_s = cv2.boxFilter(Iy*Iy, cv2.CV_64F, (k_size, k_size))
    Ixy_s = cv2.boxFilter(Ix*Iy, cv2.CV_64F, (k_size, k_size))
    Itx_s = cv2.boxFilter(Ix*It, cv2.CV_64F, (k_size, k_size))
    Ity_s = cv2.boxFilter(Iy*It, cv2.CV_64F, (k_size, k_size))



    U = (Ixy_s * Ity_s - Iy_s * Itx_s)/(Ix_s * Iy_s - Ixy_s *Ixy_s)
    V = (Ixy_s * Itx_s - Ix_s * Ity_s)/(Ix_s * Iy_s - Ixy_s *Ixy_s)

##    A = np.array([[Ix_s, Ixy_s],
##                  [Ixy_s, Iy_s]])
##    
##
##    A_sub = 1 / ((A[0][0] * A[1][1]) - (A[0][1] * A[1][0]))
##
##    A_new = np.array([[Iy_s, (-1)*Ixy_s],[(-1)*Ixy_s, Ix_s]])
##
##    print('shape of A_new', A_new.shape)
##    A_T = A_sub * A_new
##
##    print('shape of A_new', A_T.shape)
##    B = np.array([[(-1)* Itx_s],
##                  [(-1)* Ity_s]])
##
##    print('shape B', B.shape)
##
##    print('shape AT', A_T.shape)
##
##    U = A_T[0][0] * B[0][0] + A_T[0][1] * B[1][0]
##    V = A_T[1][0] * B[0][0] + A_T[1][1] * B[1][0]

    

    return U, V #u.astype(np.uint8), v.astype(np.uint8) #


def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    kernel = np.array([1, 4, 6, 4, 1]) / 16
    r_img = cv2.sepFilter2D(image, -1, kernel, kernel)[::2, ::2]

##    print('reduce1',img_bd.shape)

    
    return r_img
    #raise NotImplementedError


def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    l = [None] * levels
    print(len(l))
    
    t_img = image

    for i in range(len(l)):
        if(i == 0):
            l[0] = image
        else:
            l[i] = reduce_image(t_img)
            t_img = l[i]

    print(len(l))
    return l
    #raise NotImplementedError


def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    print(len(img_list))

    img1 = img_list[0]
    img1_m = normalize_and_scale(img1, scale_range=(0, 255))

    for i in range(len(img_list)):
        if (i == 0):
            print('Do Nothing')
            output = img1_m
        else:
            img = img_list[i]
            img_m = np.zeros((img1.shape[0],img.shape[1]))
            img_m[0:img.shape[0], 0:img.shape[1]] = normalize_and_scale(img, scale_range=(0, 255))
            output = cv2.hconcat([output,img_m])

##    cv2.imshow('output',output)
##    cv2.waitKey(0)

##    img1 = img_list[0]
##    img2 = img_list[1]
##    img3 = img_list[2]
##    img4 = img_list[3]
##    print(img1.shape, img2.shape, img3.shape, img4.shape)
##    img1_m = normalize_and_scale(img1, scale_range=(0, 255))
##    
##    img2_m = np.zeros((img1.shape[0],img2.shape[1]))
##
##    img2_m[0:img2.shape[0], 0:img2.shape[1]] = normalize_and_scale(img2, scale_range=(0, 255))
##
##
##    img3_m = np.zeros((img1.shape[0],img3.shape[1]))
##
##    img3_m[0:img3.shape[0], 0:img3.shape[1]] = normalize_and_scale(img3, scale_range=(0, 255))
##
##    img4_m = np.zeros((img1.shape[0],img4.shape[1]))
##
##    img4_m[0:img4.shape[0], 0:img4.shape[1]] = normalize_and_scale(img4, scale_range=(0, 255))
##
##    
##
##    output = cv2.hconcat([img1_m,img2_m])
##    output1 = cv2.hconcat([output,img3_m])
##    output2 = cv2.hconcat([output1,img4_m])
##    print('output2',output2.shape)
##    
##    cv2.imshow('output',output2)
##    cv2.waitKey(0)

    return output
    #raise NotImplementedError


def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """



    kernel = np.array([1, 4, 6, 4, 1]) / 8
    e_img = np.zeros((image.shape[0] * 2,image.shape[1] * 2))
    e_img[::2, ::2] = image
    newImage = cv2.sepFilter2D(e_img, -1, kernel, kernel)

##
    return newImage
#    raise NotImplementedError


def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    l_pyr = []

    for i in range(len(g_pyr) - 1):
        img = expand_image(g_pyr[i+1])
        l_img = g_pyr[i] - img[:g_pyr[i].shape[0], :g_pyr[i].shape[1]]
        l_pyr.append(l_img)

    l_pyr.append(g_pyr[-1])
    return l_pyr
 #   raise NotImplementedError


def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """


    M, N = image.shape
    X, Y = np.meshgrid(range(N), range(M))

 #   print('X,Y',X,Y)
    X = (X + U).astype(np.float32)
    Y = (Y + V).astype(np.float32)

    img_n = cv2.remap(image, X, Y, interpolation = interpolation, borderMode = border_mode)

    return img_n
    
#    raise NotImplementedError


def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """

    pyd_a = gaussian_pyramid(img_a, levels)
    pyd_b = gaussian_pyramid(img_b, levels)

    h, w = pyd_a[-1].shape

    U = np.zeros((h, w), np.float64)
    V = np.zeros((h, w), np.float64)

    for i in range(levels-1, -1, -1):
        c_img_a = pyd_a[i]
        c_img_b = pyd_b[i]

        c_h, c_w = c_img_a.shape

        U = expand_image(U) * 2
        V = expand_image(V) * 2
        
        U = U[:c_h, :c_w]
        V = V[:c_h, :c_w]

        warp_b = warp(c_img_b, U, V, interpolation, border_mode)

        u, v = optic_flow_lk(c_img_a, warp_b, k_size, k_type, sigma)

        U = U + u
        V = V + v



    

##    print('length of pyd_a',len(pyd_a))
##    print('pyd_a[-1]',pyd_a[-1])
##    print('pyd_a[0]',pyd_a[0])
##    print('pyd_a[1]',pyd_a[1])
##    print('pyd_a[2]',pyd_a[2])
##    print('pyd_a[3]',pyd_a[3])
    
    return U, V
    #raise NotImplementedError
