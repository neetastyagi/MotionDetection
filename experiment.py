"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np

import ps4

# I/O directories
input_dir = "input_images"
output_dir = "./"
VID_DIR = "input_videos"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in range(0, v.shape[0], stride):

        for x in range(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out


# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    print('level',level)
    print('pyr',len(pyr))

    for i in np.arange(level-1, -1, -1):

        u_new = ps4.expand_image(u) * 2
        v_new = ps4.expand_image(v) * 2

        h_c, w_c = pyr[i].shape
        u_new = u_new[:h_c, :w_c]
        v_new = v_new[:h_c, :w_c]

    return u_new, v_new
    # TODO: Your code here
    #raise NotImplementedError


def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.

    

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 30  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    shift_0_smooth = cv2.GaussianBlur(shift_0, (7,7), 2)

    k_size = 40 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0 # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    shift_0_smooth = cv2.GaussianBlur(shift_0, (5,5), 2)
    shift_r10_smooth = cv2.GaussianBlur(shift_r10, (5,5), 2)
    shift_r20_smooth = cv2.GaussianBlur(shift_r20, (5,5), 2)
    shift_r40_smooth = cv2.GaussianBlur(shift_r40, (5,5), 2)

    k_size = 35  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0_smooth, shift_r10_smooth, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)

    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0_smooth, shift_r20_smooth, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)


    k_size = 50  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0_smooth, shift_r40_smooth, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)





    #raise NotImplementedError


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 4  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 1  # TODO: Select the level number (or id) you wish to use
    k_size = 20 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02)) #diff_yos_img))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 4  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 1 # TODO: Select the level number (or id) you wish to use
    k_size = 20 # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0 # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    levels = 3  # TODO: Define the number of levels
    k_size = 40  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 3  # TODO: Define the number of levels
    k_size = 80  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))

def insert_images(img_a, img_b, k_size):

    
    levels = 6
    k_size = k_size  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

   

    u, v = ps4.hierarchical_lk(img_b, img_a, levels, k_size,k_type, sigma, interpolation, border_mode)

    u_r, v_r = ps4.hierarchical_lk(img_a, img_b, levels, k_size,k_type, sigma, interpolation, border_mode)

    img_1 = ps4.warp(img_a,  u * 0.2, v * 0.2, interpolation, border_mode)

    img_4 = ps4.warp(img_b, -u_r * 0.2 , v_r * 0.2 , interpolation, border_mode)
    

    u1, v1 = ps4.hierarchical_lk(img_1, img_b, levels, k_size, k_type, sigma, interpolation, border_mode)
    

    img_2 = ps4.warp(img_1, u1 * 0.4, v1 * 0.4, interpolation, border_mode)


    u2, v2 = ps4.hierarchical_lk(img_b, img_4, levels, k_size, k_type, sigma, interpolation, border_mode)


    img_3 = ps4.warp(img_4, -u2 * 0.4, -v2 * 0.4, interpolation, border_mode)


 #   u3, v3 = ps4.hierarchical_lk(img_b, img_3, levels, k_size,k_type, sigma, interpolation, border_mode)


    

    output1 = cv2.hconcat([ps4.normalize_and_scale(img_a), ps4.normalize_and_scale(img_1), ps4.normalize_and_scale(img_2)])

    output2 = cv2.hconcat([ps4.normalize_and_scale(img_3), ps4.normalize_and_scale(img_4), ps4.normalize_and_scale(img_b)])

    output = cv2.vconcat([output1, output2])

    return output

def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.

    output = insert_images(shift_0, shift_r10, 35 )


    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-1.png"),
                ps4.normalize_and_scale(output))

##    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-2.png"),
##                ps4.normalize_and_scale(shift_20))
##
##    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-3.png"),
##                ps4.normalize_and_scale(shift_40))
##
##    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-4.png"),
##                ps4.normalize_and_scale(shift_60))
##
##    cv2.imwrite(os.path.join(output_dir, "ps4-5-a-5.png"),
##                ps4.normalize_and_scale(shift_80))






def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    mc01 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                      'mc01.png'), 0) / 255.
    
    mc02 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                        'mc02.png'), 0) / 255.

    mc03 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                        'mc03.png'), 0) / 255.

    output1 = insert_images(mc01, mc02, 30 )

    output2 = insert_images(mc02, mc03, 30 )

    output3 = insert_images(mc01, mc03, 30 )

    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-1.png"),
                ps4.normalize_and_scale(output1))

    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-2.png"),
                ps4.normalize_and_scale(output2))

    cv2.imwrite(os.path.join(output_dir, "ps4-5-b-3.png"),
                ps4.normalize_and_scale(output3))

    

 #   raise NotImplementedError


def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    k_size = 350
    levels = 3
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    sigma = 0

    video = os.path.join(VID_DIR, 'ps4-my-video.mp4')
    output_video = os.path.join(VID_DIR, 'ps4-6-final.mp4')




    image_gen = video_frame_generator(video)

    img1 = image_gen.__next__()

    h = img1.shape[0]

    w = img1.shape[1]

    video_out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))

    output_counter = 1

    while img1 is not None:
        
        img2 = image_gen.__next__()

        if img2 is None:
            break

        gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
        gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

        u, v = ps4.optic_flow_lk(gray_img2, gray_img1, k_size, 'uniform', sigma)
        
 #       u, v = ps4.hierarchical_lk(gray_img2, gray_img1, levels, k_size, 'uniform', 0, interpolation, border_mode)

        u[np.where(np.isnan(u))] = 0.
        v[np.where(np.isnan(v))] = 0.

        u_v = quiver(u, v, scale=5, stride=10)


#        print('u_v',u_v.size)

 #       print('img1',img1.size)

        print(output_counter)


        output = cv2.addWeighted(img1, 0.7, u_v, 0.3, 0.0)

        if output_counter == 1 or output_counter ==2 :
            cv2.imwrite(os.path.join(output_dir, "ps4-6-a-{}.png".format(output_counter)), output)

        output_counter += 1

        video_out.write(output)

        img1 = img2

    
    video_out.release()        

        

    #raise NotImplementedError

def video_frame_generator(filename):

    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break
        
    video.release()
    yield None



if __name__ == '__main__':
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_6()
