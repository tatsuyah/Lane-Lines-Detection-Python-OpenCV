
# Lane Line Detection using Python and OpenCV

## Camera Calibration


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
%matplotlib inline
```


```python
def cal_undistort(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist, mtx, dist

def collect_callibration_points():
    objpoints = []
    imgpoints = []

    images = glob.glob('./camera_cal/calibration*.jpg')
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1, 2)

    for fname in images:
        img = mpimg.imread(fname)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    return imgpoints, objpoints

def compare_images(image1, image2, image1_exp="Image 1", image2_exp="Image 2"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.set_title(image1_exp, fontsize=50)
    ax2.imshow(image2)
    ax2.set_title(image2_exp, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


imgpoints, objpoints = collect_callibration_points()
img = mpimg.imread('./camera_cal/calibration3.jpg')
undistorted, mtx, dist_coefficients = cal_undistort(img, objpoints, imgpoints)
#compare_images(img, undistorted, "Original Image", "Undistorted Image")

image_path = './test_images/straight_lines1.jpg'
image = mpimg.imread(image_path)
image, mtx, dist_coefficients = cal_undistort(image, objpoints, imgpoints)
```

## Gradient Thresholds


```python
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    isX = True if orient == 'x' else False
    sobel = cv2.Sobel(gray, cv2.CV_64F, isX, not isX)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    return dir_binary

def apply_thresholds(image, ksize=3):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined

combined = apply_thresholds(image)
compare_images(image, combined, "Original Image", "Gradient Thresholds")
```


![png](output/output_6_0.png)


## Color Threshold


```python
def apply_color_threshold(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    return s_binary


s_binary = apply_color_threshold(image)
compare_images(image, s_binary, "Original Image", "Color Threshold")
```


![png](output/output_8_0.png)


## Combine Color and Gradient


```python
def combine_threshold(s_binary, combined):
    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1)] = 1

    return combined_binary


combined_binary = combine_threshold(s_binary, combined)
compare_images(image, combined_binary, "Original Image", "Gradient and Color Threshold")
```


![png](output/output_10_0.png)


## Perspective Transform


```python
def warp(img):
    img_size = (img.shape[1], img.shape[0])

    src = np.float32(
        [[685, 450],
          [1090, 710],
          [220, 710],
          [595, 450]])

    dst = np.float32(
        [[900, 0],
          [900, 710],
          [250, 710],
          [250, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return binary_warped, Minv

def compare_plotted_images(image1, image2, image1_exp="Image 1", image2_exp="Image 2"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1)
    ax1.plot([685, 1090], [450, 710], color='r', linewidth="5")
    ax1.plot([1090, 220], [710, 710], color='r', linewidth="5")
    ax1.plot([220, 595], [710, 450], color='r', linewidth="5")
    ax1.plot([595, 685], [450, 450], color='r', linewidth="5")
    ax1.set_title(image1_exp, fontsize=50)
    ax2.imshow(image2)
    ax2.plot([900, 900], [0, 710], color='r', linewidth="5")
    ax2.plot([900, 250], [710, 710], color='r', linewidth="5")
    ax2.plot([250, 250], [710, 0], color='r', linewidth="5")
    ax2.plot([250, 900], [0, 0], color='r', linewidth="5")
    ax2.set_title(image2_exp, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


warped, Minv = warp(image)
compare_plotted_images(image, warped, "Original Image", "Warped Image")
```


![png](output/output_12_0.png)


# Finding the Lines

## Histogram


```python
def get_histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    return histogram


binary_warped, Minv = warp(combined_binary)
histogram = get_histogram(binary_warped)
plt.plot(histogram)
```




    [<matplotlib.lines.Line2D at 0x10b551898>]




![png](output/output_15_1.png)


## Sliding Window


```python
def slide_window(binary_warped, histogram):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    return ploty, left_fit, right_fit

ploty, left_fit, right_fit = slide_window(binary_warped, histogram)
```


![png](output/output_17_0.png)


## Skipping Slinding Window


```python
def skip_sliding_window(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
    left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    ################################
    ## Visualization
    ################################

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin,
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = rightx
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = right_fitx
    ret['ploty'] = ploty

    return ret

draw_info = skip_sliding_window(binary_warped, left_fit, right_fit)
```


![png](output/output_19_0.png)


## Measuring Curvature


```python
def measure_curvature(ploty, lines_info):
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    leftx = lines_info['left_fitx']
    rightx = lines_info['right_fitx']

    leftx = leftx[::-1]  
    rightx = rightx[::-1]  

    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    print(left_curverad, 'm', right_curverad, 'm')

    return left_curverad, right_curverad

left_curverad, right_curverad = measure_curvature(ploty, draw_info)
```

    21233.5680677 m 1293.63478872 m


## Drawing


```python
def draw_lane_lines(original_image, warped_image, Minv, draw_info):
    leftx = draw_info['leftx']
    rightx = draw_info['rightx']
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    return result


result = draw_lane_lines(image, binary_warped, Minv, draw_info)
plt.imshow(result)
```




    <matplotlib.image.AxesImage at 0x10bf5e748>




![png](output/output_23_1.png)


## Defining image processing method


```python
global used_warped
global used_ret

def process_image(image):
    global used_warped
    global used_ret

    #Undistort image
    image, mtx, dist_coefficients = cal_undistort(image, objpoints, imgpoints)

    # Gradient thresholding
    gradient_combined = apply_thresholds(image)

    # Color thresholding
    s_binary = apply_color_threshold(image)

    # Combine Gradient and Color thresholding
    combined_binary = combine_threshold(s_binary, gradient_combined)

    # Transforming Perspective
    binary_warped, Minv = warp(combined_binary)

    # Getting Histogram
    histogram = get_histogram(binary_warped)

    # Sliding Window to detect lane lines
    ploty, left_fit, right_fit = slide_window(binary_warped, histogram)

    # Skipping Sliding Window
    ret = skip_sliding_window(binary_warped, left_fit, right_fit)

    # Measuring Curvature
    left_curverad, right_curverad = measure_curvature(ploty, ret)

     # Sanity check: whether the lines are roughly parallel and have similar curvature
    slope_left = ret['left_fitx'][0] - ret['left_fitx'][-1]
    slope_right = ret['right_fitx'][0] - ret['right_fitx'][-1]
    slope_diff = abs(slope_left - slope_right)
    slope_threshold = 150
    curve_diff = abs(left_curverad - right_curverad)
    curve_threshold = 10000

    if (slope_diff > slope_threshold or curve_diff > curve_threshold):
        binary_warped = used_warped
        ret = used_ret

    # Visualizing Lane Lines Info
    result = draw_lane_lines(image, binary_warped, Minv, ret)

    # Annotating curvature
    fontType = cv2.FONT_HERSHEY_SIMPLEX
    curvature_text = 'The radius of curvature = ' + str(round(left_curverad, 3)) + 'm'
    cv2.putText(result, curvature_text, (30, 60), fontType, 1.5, (255, 255, 255), 3)

    # Annotating deviation
    deviation_pixels = image.shape[1]/2 - abs(ret['right_fitx'][-1] - ret['left_fitx'][-1])
    xm_per_pix = 3.7/700
    deviation = deviation_pixels * xm_per_pix
    direction = "left" if deviation < 0 else "right"
    deviation_text = 'Vehicle is ' + str(round(abs(deviation), 3)) + 'm ' + direction + ' of center'
    cv2.putText(result, deviation_text, (30, 110), fontType, 1.5, (255, 255, 255), 3)

    used_warped = binary_warped
    used_ret = ret

    return result

#result_image = process_image(image)
#plt.imshow(result_image)
```

## Applying to Video


```python
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
from IPython.display import HTML
```


```python
output = 'result.mp4'
clip = VideoFileClip("project_video.mp4")
video_clip = clip.fl_image(process_image)
%time video_clip.write_videofile(output, audio=False)
```

    373.06845538 m 658.715212662 m
    curve_diff:  285.646757282
    [MoviePy] >>>> Building video result.mp4
    [MoviePy] Writing video result.mp4


      0%|          | 1/1261 [00:01<26:07,  1.24s/it]

    373.06845538 m 658.715212662 m
    curve_diff:  285.646757282


      0%|          | 2/1261 [00:02<25:45,  1.23s/it]

    387.008845952 m 546.553791349 m
    curve_diff:  159.544945397


      0%|          | 3/1261 [00:03<25:23,  1.21s/it]

    388.197985204 m 618.068241349 m
    curve_diff:  229.870256145


      0%|          | 4/1261 [00:04<25:26,  1.21s/it]

    377.507355847 m 514.374975832 m
    curve_diff:  136.867619984


      0%|          | 5/1261 [00:06<25:22,  1.21s/it]

    407.428063255 m 440.29903724 m
    curve_diff:  32.8709739848


      0%|          | 6/1261 [00:07<25:19,  1.21s/it]

    399.906952626 m 387.004225876 m
    curve_diff:  12.9027267504


      1%|          | 7/1261 [00:08<25:17,  1.21s/it]

    407.845074284 m 381.609908269 m
    curve_diff:  26.2351660145


      1%|          | 8/1261 [00:09<25:20,  1.21s/it]

    435.507512746 m 355.930659797 m
    curve_diff:  79.5768529482


      1%|          | 9/1261 [00:10<25:39,  1.23s/it]

    438.677321466 m 482.55000948 m
    curve_diff:  43.8726880141


      1%|          | 10/1261 [00:12<25:56,  1.24s/it]

    417.845687432 m 570.697849906 m
    curve_diff:  152.852162474


      1%|          | 11/1261 [00:13<25:38,  1.23s/it]

    419.625923321 m 848.175317354 m
    curve_diff:  428.549394033


      1%|          | 12/1261 [00:14<25:33,  1.23s/it]

    447.728327469 m 717.326766034 m
    curve_diff:  269.598438565


      1%|          | 13/1261 [00:15<25:31,  1.23s/it]

    434.872189917 m 772.147179646 m
    curve_diff:  337.274989729


      1%|          | 14/1261 [00:17<25:34,  1.23s/it]

    446.616242973 m 530.843906503 m
    curve_diff:  84.2276635299


      1%|          | 15/1261 [00:18<25:24,  1.22s/it]

    451.31197604 m 442.652697359 m
    curve_diff:  8.65927868133


      1%|▏         | 16/1261 [00:19<25:20,  1.22s/it]

    424.226247595 m 367.997678595 m
    curve_diff:  56.2285690002


      1%|▏         | 17/1261 [00:20<25:12,  1.22s/it]

    415.349955926 m 335.38952305 m
    curve_diff:  79.9604328759


      1%|▏         | 18/1261 [00:22<25:39,  1.24s/it]

    488.947235393 m 354.16257937 m
    curve_diff:  134.784656023


      2%|▏         | 19/1261 [00:23<25:25,  1.23s/it]

    490.630861581 m 334.129228515 m
    curve_diff:  156.501633066


      2%|▏         | 20/1261 [00:24<26:33,  1.28s/it]

    500.023690506 m 334.587148329 m
    curve_diff:  165.436542177


      2%|▏         | 21/1261 [00:26<27:24,  1.33s/it]

    505.004493951 m 308.11031444 m
    curve_diff:  196.894179511


      2%|▏         | 22/1261 [00:27<27:13,  1.32s/it]

    499.017747651 m 317.01641508 m
    curve_diff:  182.001332571


      2%|▏         | 23/1261 [00:28<26:38,  1.29s/it]

    518.993076167 m 371.953469909 m
    curve_diff:  147.039606258


      2%|▏         | 24/1261 [00:29<26:14,  1.27s/it]

    463.744013623 m 472.017438718 m
    curve_diff:  8.27342509503


      2%|▏         | 25/1261 [00:31<25:44,  1.25s/it]

    429.619696069 m 702.131364745 m
    curve_diff:  272.511668676


      2%|▏         | 26/1261 [00:32<26:29,  1.29s/it]

    443.551612833 m 854.29417474 m
    curve_diff:  410.742561907


      2%|▏         | 27/1261 [00:33<26:25,  1.29s/it]

    424.839790758 m 653.800676099 m
    curve_diff:  228.960885341


      2%|▏         | 28/1261 [00:34<26:13,  1.28s/it]

    437.967414569 m 609.652336008 m
    curve_diff:  171.684921439


      2%|▏         | 29/1261 [00:36<26:00,  1.27s/it]

    431.033843651 m 475.924132116 m
    curve_diff:  44.8902884641


      2%|▏         | 30/1261 [00:37<25:42,  1.25s/it]

    374.602458617 m 495.414569608 m
    curve_diff:  120.812110991


      2%|▏         | 31/1261 [00:38<25:08,  1.23s/it]

    356.911137484 m 296.902782246 m
    curve_diff:  60.0083552387


      3%|▎         | 32/1261 [00:39<24:52,  1.21s/it]

    327.007736692 m 284.234170258 m
    curve_diff:  42.7735664344


      3%|▎         | 33/1261 [00:40<25:03,  1.22s/it]

    323.324169307 m 265.991192337 m
    curve_diff:  57.33297697


      3%|▎         | 34/1261 [00:42<25:22,  1.24s/it]

    330.037692708 m 336.210166322 m
    curve_diff:  6.17247361367


      3%|▎         | 35/1261 [00:43<25:08,  1.23s/it]

    317.948658526 m 396.790075221 m
    curve_diff:  78.8414166948


      3%|▎         | 36/1261 [00:44<25:41,  1.26s/it]

    292.751784957 m 411.35893044 m
    curve_diff:  118.607145483


      3%|▎         | 37/1261 [00:46<25:55,  1.27s/it]

    287.909216281 m 381.837774869 m
    curve_diff:  93.9285585887


      3%|▎         | 38/1261 [00:47<25:22,  1.24s/it]

    290.280341827 m 281.932801435 m
    curve_diff:  8.3475403923


      3%|▎         | 39/1261 [00:48<24:59,  1.23s/it]

    279.024933929 m 268.408175723 m
    curve_diff:  10.6167582062


      3%|▎         | 40/1261 [00:49<24:39,  1.21s/it]

    285.388622555 m 314.442675574 m
    curve_diff:  29.0540530189


      3%|▎         | 41/1261 [00:50<24:24,  1.20s/it]

    299.395951769 m 268.470363057 m
    curve_diff:  30.9255887117


      3%|▎         | 42/1261 [00:52<24:23,  1.20s/it]

    321.939011116 m 277.766112299 m
    curve_diff:  44.1728988169


      3%|▎         | 43/1261 [00:53<27:55,  1.38s/it]

    335.973850892 m 276.176573908 m
    curve_diff:  59.7972769837


      3%|▎         | 44/1261 [00:55<27:51,  1.37s/it]

    357.896917094 m 273.601353995 m
    curve_diff:  84.295563099


      4%|▎         | 45/1261 [00:56<27:20,  1.35s/it]

    352.141653063 m 294.82801589 m
    curve_diff:  57.3136371737


      4%|▎         | 46/1261 [00:57<26:34,  1.31s/it]

    386.6315585 m 294.415005134 m
    curve_diff:  92.2165533653


      4%|▎         | 47/1261 [00:58<25:55,  1.28s/it]

    386.715437166 m 348.005200934 m
    curve_diff:  38.7102362312


      4%|▍         | 48/1261 [01:00<25:27,  1.26s/it]

    397.975988157 m 423.502352286 m
    curve_diff:  25.5263641289


      4%|▍         | 49/1261 [01:01<25:41,  1.27s/it]

    440.201705688 m 437.728641233 m
    curve_diff:  2.47306445444


      4%|▍         | 50/1261 [01:02<25:32,  1.27s/it]

    437.112245408 m 371.742314899 m
    curve_diff:  65.3699305096


      4%|▍         | 51/1261 [01:03<25:39,  1.27s/it]

    455.454322403 m 489.968289307 m
    curve_diff:  34.5139669037


      4%|▍         | 52/1261 [01:05<25:45,  1.28s/it]

    441.364909779 m 445.554124868 m
    curve_diff:  4.18921508913


      4%|▍         | 53/1261 [01:06<25:41,  1.28s/it]

    456.571018392 m 412.707291216 m
    curve_diff:  43.8637271759


      4%|▍         | 54/1261 [01:07<25:37,  1.27s/it]

    476.379650292 m 394.875295257 m
    curve_diff:  81.5043550341


      4%|▍         | 55/1261 [01:09<27:25,  1.36s/it]

    459.96913827 m 347.357099914 m
    curve_diff:  112.612038355


      4%|▍         | 56/1261 [01:10<26:53,  1.34s/it]

    498.071033145 m 346.471057715 m
    curve_diff:  151.59997543


      5%|▍         | 57/1261 [01:11<26:49,  1.34s/it]

    479.739514141 m 356.034832809 m
    curve_diff:  123.704681332


      5%|▍         | 58/1261 [01:13<25:56,  1.29s/it]

    464.119428495 m 404.671257638 m
    curve_diff:  59.4481708575


      5%|▍         | 59/1261 [01:14<25:48,  1.29s/it]

    402.584159219 m 574.881246101 m
    curve_diff:  172.297086882


      5%|▍         | 60/1261 [01:15<26:00,  1.30s/it]

    405.161719252 m 326.086123909 m
    curve_diff:  79.0755953434


      5%|▍         | 61/1261 [01:17<25:55,  1.30s/it]

    443.264793015 m 344.380237373 m
    curve_diff:  98.8845556419


      5%|▍         | 62/1261 [01:18<25:53,  1.30s/it]

    419.790001184 m 312.557711113 m
    curve_diff:  107.232290071


      5%|▍         | 63/1261 [01:19<25:28,  1.28s/it]

    418.433981619 m 313.939709323 m
    curve_diff:  104.494272295


      5%|▌         | 64/1261 [01:20<25:12,  1.26s/it]

    408.212861014 m 295.140403898 m
    curve_diff:  113.072457115


      5%|▌         | 65/1261 [01:22<24:56,  1.25s/it]

    421.680418801 m 304.68353807 m
    curve_diff:  116.996880731


      5%|▌         | 66/1261 [01:23<24:45,  1.24s/it]

    390.274314493 m 356.176553919 m
    curve_diff:  34.0977605746


      5%|▌         | 67/1261 [01:24<25:08,  1.26s/it]

    414.190466858 m 380.35055792 m
    curve_diff:  33.8399089382


      5%|▌         | 68/1261 [01:25<25:03,  1.26s/it]

    438.684057404 m 430.34524758 m
    curve_diff:  8.3388098246


      5%|▌         | 69/1261 [01:27<24:54,  1.25s/it]

    409.447702596 m 380.994624663 m
    curve_diff:  28.453077933


      6%|▌         | 70/1261 [01:28<25:55,  1.31s/it]

    466.539479885 m 398.127454795 m
    curve_diff:  68.4120250903


      6%|▌         | 71/1261 [01:29<25:33,  1.29s/it]

    458.687664084 m 455.59649153 m
    curve_diff:  3.09117255436


      6%|▌         | 72/1261 [01:30<25:20,  1.28s/it]

    445.651346029 m 505.932895383 m
    curve_diff:  60.2815493538


      6%|▌         | 73/1261 [01:32<25:08,  1.27s/it]

    460.807767898 m 390.986573897 m
    curve_diff:  69.8211940013


      6%|▌         | 74/1261 [01:33<24:54,  1.26s/it]

    475.70357583 m 358.545573976 m
    curve_diff:  117.158001854


      6%|▌         | 75/1261 [01:34<24:58,  1.26s/it]

    512.450454904 m 325.909574746 m
    curve_diff:  186.540880157


      6%|▌         | 76/1261 [01:36<25:01,  1.27s/it]

    485.327097168 m 354.533957153 m
    curve_diff:  130.793140015


      6%|▌         | 77/1261 [01:37<24:47,  1.26s/it]

    528.382402463 m 375.813579723 m
    curve_diff:  152.568822741


      6%|▌         | 78/1261 [01:38<24:39,  1.25s/it]

    582.45058827 m 433.161576794 m
    curve_diff:  149.289011476


      6%|▋         | 79/1261 [01:39<24:30,  1.24s/it]

    571.277820718 m 400.852285366 m
    curve_diff:  170.425535352


      6%|▋         | 80/1261 [01:40<24:26,  1.24s/it]

    555.321564869 m 370.917399872 m
    curve_diff:  184.404164997


      6%|▋         | 81/1261 [01:42<24:22,  1.24s/it]

    527.936062324 m 339.980907886 m
    curve_diff:  187.955154438


      7%|▋         | 82/1261 [01:43<24:26,  1.24s/it]

    536.810305384 m 340.559034059 m
    curve_diff:  196.251271325


      7%|▋         | 83/1261 [01:44<24:24,  1.24s/it]

    516.891860731 m 311.942035378 m
    curve_diff:  204.949825353


      7%|▋         | 84/1261 [01:46<25:27,  1.30s/it]

    528.043360135 m 333.702055497 m
    curve_diff:  194.341304639


      7%|▋         | 85/1261 [01:47<25:07,  1.28s/it]

    495.522735027 m 320.950816286 m
    curve_diff:  174.571918742


      7%|▋         | 86/1261 [01:48<24:48,  1.27s/it]

    426.794518486 m 432.165428952 m
    curve_diff:  5.37091046672


      7%|▋         | 87/1261 [01:49<24:30,  1.25s/it]

    396.633431299 m 466.483797652 m
    curve_diff:  69.8503663527


      7%|▋         | 88/1261 [01:51<24:23,  1.25s/it]

    368.027193216 m 441.344752642 m
    curve_diff:  73.3175594256


      7%|▋         | 89/1261 [01:52<24:19,  1.25s/it]

    369.510606916 m 420.049714435 m
    curve_diff:  50.5391075186


      7%|▋         | 90/1261 [01:53<24:12,  1.24s/it]

    334.18796029 m 306.896571207 m
    curve_diff:  27.2913890837


      7%|▋         | 91/1261 [01:54<24:11,  1.24s/it]

    354.204267856 m 363.001591973 m
    curve_diff:  8.79732411629


      7%|▋         | 92/1261 [01:56<24:31,  1.26s/it]

    370.20831086 m 350.573356013 m
    curve_diff:  19.6349548477


      7%|▋         | 93/1261 [01:57<24:37,  1.26s/it]

    391.138498337 m 355.01087295 m
    curve_diff:  36.1276253876


      7%|▋         | 94/1261 [01:58<24:31,  1.26s/it]

    372.945879757 m 400.493005036 m
    curve_diff:  27.5471252797


      8%|▊         | 95/1261 [01:59<24:28,  1.26s/it]

    392.078261514 m 382.345137183 m
    curve_diff:  9.7331243306


      8%|▊         | 96/1261 [02:01<24:36,  1.27s/it]

    348.647042351 m 317.752271852 m
    curve_diff:  30.8947704988


      8%|▊         | 97/1261 [02:02<24:23,  1.26s/it]

    345.443771465 m 433.635351624 m
    curve_diff:  88.191580159


      8%|▊         | 98/1261 [02:03<24:19,  1.25s/it]

    358.739248865 m 383.101681066 m
    curve_diff:  24.3624322005


      8%|▊         | 99/1261 [02:04<24:09,  1.25s/it]

    342.313017027 m 367.307575985 m
    curve_diff:  24.9945589583


      8%|▊         | 100/1261 [02:06<24:12,  1.25s/it]

    344.392868182 m 438.623646162 m
    curve_diff:  94.2307779794


      8%|▊         | 101/1261 [02:07<24:22,  1.26s/it]

    344.496919869 m 365.421063614 m
    curve_diff:  20.924143745


      8%|▊         | 102/1261 [02:08<24:09,  1.25s/it]

    345.098482172 m 413.710539804 m
    curve_diff:  68.6120576327


      8%|▊         | 103/1261 [02:09<24:06,  1.25s/it]

    340.995834097 m 290.483294911 m
    curve_diff:  50.512539186


      8%|▊         | 104/1261 [02:11<24:04,  1.25s/it]

    343.591938523 m 291.566486505 m
    curve_diff:  52.0254520185


      8%|▊         | 105/1261 [02:12<24:03,  1.25s/it]

    361.283231015 m 324.84146028 m
    curve_diff:  36.4417707356


      8%|▊         | 106/1261 [02:13<23:54,  1.24s/it]

    369.160160709 m 298.005410962 m
    curve_diff:  71.1547497473


      8%|▊         | 107/1261 [02:14<23:56,  1.24s/it]

    363.187173552 m 286.485056996 m
    curve_diff:  76.7021165557


      9%|▊         | 108/1261 [02:16<24:07,  1.26s/it]

    345.192375629 m 271.046211404 m
    curve_diff:  74.1461642255


      9%|▊         | 109/1261 [02:17<24:46,  1.29s/it]

    370.4408392 m 262.039394598 m
    curve_diff:  108.401444602


      9%|▊         | 110/1261 [02:18<24:26,  1.27s/it]

    348.450664589 m 340.524906663 m
    curve_diff:  7.92575792606


      9%|▉         | 111/1261 [02:19<24:13,  1.26s/it]

    329.236083621 m 487.87829746 m
    curve_diff:  158.642213839


      9%|▉         | 112/1261 [02:21<24:03,  1.26s/it]

    363.77328542 m 432.327404691 m
    curve_diff:  68.554119271


      9%|▉         | 113/1261 [02:22<23:54,  1.25s/it]

    331.383539763 m 340.439410273 m
    curve_diff:  9.05587050951


      9%|▉         | 114/1261 [02:23<24:12,  1.27s/it]

    323.104307663 m 360.991309637 m
    curve_diff:  37.8870019744


      9%|▉         | 115/1261 [02:25<25:33,  1.34s/it]

    307.392169717 m 300.762157396 m
    curve_diff:  6.63001232085


      9%|▉         | 116/1261 [02:26<25:20,  1.33s/it]

    307.641289345 m 281.765654882 m
    curve_diff:  25.8756344627


      9%|▉         | 117/1261 [02:28<29:10,  1.53s/it]

    326.159215129 m 289.953514119 m
    curve_diff:  36.2057010094


      9%|▉         | 118/1261 [02:30<28:54,  1.52s/it]

    327.248537835 m 269.727295914 m
    curve_diff:  57.5212419215


      9%|▉         | 119/1261 [02:31<27:27,  1.44s/it]

    334.172460479 m 295.797855545 m
    curve_diff:  38.3746049341


     10%|▉         | 120/1261 [02:32<26:23,  1.39s/it]

    329.886445711 m 335.88082542 m
    curve_diff:  5.99437970877


     10%|▉         | 121/1261 [02:33<25:30,  1.34s/it]

    330.458117351 m 370.127611485 m
    curve_diff:  39.6694941343


     10%|▉         | 122/1261 [02:35<24:54,  1.31s/it]

    338.933270844 m 421.15162301 m
    curve_diff:  82.218352166


     10%|▉         | 123/1261 [02:36<24:25,  1.29s/it]

    316.278506799 m 371.26054462 m
    curve_diff:  54.9820378215


     10%|▉         | 124/1261 [02:37<24:12,  1.28s/it]

    314.947544523 m 289.779764788 m
    curve_diff:  25.1677797355


     10%|▉         | 125/1261 [02:38<24:29,  1.29s/it]

    313.272444624 m 285.199894095 m
    curve_diff:  28.0725505289


     10%|▉         | 126/1261 [02:40<24:09,  1.28s/it]

    339.49157376 m 321.600753426 m
    curve_diff:  17.8908203337


     10%|█         | 127/1261 [02:41<23:53,  1.26s/it]

    330.056416299 m 299.07788192 m
    curve_diff:  30.9785343789


     10%|█         | 128/1261 [02:42<23:44,  1.26s/it]

    338.606164802 m 275.296498413 m
    curve_diff:  63.3096663892


     10%|█         | 129/1261 [02:43<23:40,  1.25s/it]

    338.023456616 m 322.184785964 m
    curve_diff:  15.8386706523


     10%|█         | 130/1261 [02:45<23:52,  1.27s/it]

    327.770420074 m 338.719145385 m
    curve_diff:  10.9487253114


     10%|█         | 131/1261 [02:46<23:40,  1.26s/it]

    376.606855011 m 368.347980349 m
    curve_diff:  8.2588746624


     10%|█         | 132/1261 [02:47<24:01,  1.28s/it]

    399.005666114 m 369.459323253 m
    curve_diff:  29.5463428611


     11%|█         | 133/1261 [02:48<24:03,  1.28s/it]

    358.493951746 m 398.254558494 m
    curve_diff:  39.7606067473


     11%|█         | 134/1261 [02:50<23:48,  1.27s/it]

    358.205011214 m 426.529350553 m
    curve_diff:  68.3243393393


     11%|█         | 135/1261 [02:51<23:35,  1.26s/it]

    388.006345958 m 480.419797426 m
    curve_diff:  92.4134514679


     11%|█         | 136/1261 [02:52<23:23,  1.25s/it]

    383.122075604 m 459.202512858 m
    curve_diff:  76.0804372542


     11%|█         | 137/1261 [02:53<23:15,  1.24s/it]

    381.572353469 m 501.812708195 m
    curve_diff:  120.240354727


     11%|█         | 138/1261 [02:55<23:10,  1.24s/it]

    411.466708821 m 546.782569562 m
    curve_diff:  135.315860741


     11%|█         | 139/1261 [02:56<23:14,  1.24s/it]

    446.808616194 m 450.798986539 m
    curve_diff:  3.9903703456


     11%|█         | 140/1261 [02:57<23:14,  1.24s/it]

    461.283360556 m 574.587228048 m
    curve_diff:  113.303867492


     11%|█         | 141/1261 [02:58<23:32,  1.26s/it]

    510.909017237 m 430.159627647 m
    curve_diff:  80.7493895902


     11%|█▏        | 142/1261 [03:00<23:44,  1.27s/it]

    528.881771102 m 417.147195939 m
    curve_diff:  111.734575163


     11%|█▏        | 143/1261 [03:01<23:33,  1.26s/it]

    470.585283788 m 348.374112314 m
    curve_diff:  122.211171474


     11%|█▏        | 144/1261 [03:02<23:20,  1.25s/it]

    453.523821567 m 368.697028016 m
    curve_diff:  84.8267935514


     11%|█▏        | 145/1261 [03:03<23:18,  1.25s/it]

    455.5280505 m 362.308019873 m
    curve_diff:  93.2200306276


     12%|█▏        | 146/1261 [03:05<23:23,  1.26s/it]

    421.1842105 m 509.460840973 m
    curve_diff:  88.2766304728


     12%|█▏        | 147/1261 [03:06<23:13,  1.25s/it]

    414.590408276 m 563.843963189 m
    curve_diff:  149.253554913


     12%|█▏        | 148/1261 [03:07<23:10,  1.25s/it]

    390.890477129 m 368.194154947 m
    curve_diff:  22.6963221823


     12%|█▏        | 149/1261 [03:08<23:06,  1.25s/it]

    402.79478706 m 369.477792378 m
    curve_diff:  33.3169946813


     12%|█▏        | 150/1261 [03:10<23:35,  1.27s/it]

    394.105762704 m 374.54885042 m
    curve_diff:  19.556912284


     12%|█▏        | 151/1261 [03:11<23:25,  1.27s/it]

    408.804705929 m 525.588884963 m
    curve_diff:  116.784179034


     12%|█▏        | 152/1261 [03:12<23:09,  1.25s/it]

    419.834404947 m 376.235008356 m
    curve_diff:  43.5993965912


     12%|█▏        | 153/1261 [03:14<23:10,  1.25s/it]

    405.497725797 m 329.380873581 m
    curve_diff:  76.1168522155


     12%|█▏        | 154/1261 [03:15<22:55,  1.24s/it]

    396.86136379 m 375.155291222 m
    curve_diff:  21.7060725678


     12%|█▏        | 155/1261 [03:16<22:51,  1.24s/it]

    421.981614234 m 327.926274045 m
    curve_diff:  94.0553401886


     12%|█▏        | 156/1261 [03:17<23:29,  1.28s/it]

    391.017775881 m 340.046457744 m
    curve_diff:  50.971318137


     12%|█▏        | 157/1261 [03:19<23:14,  1.26s/it]

    398.881447906 m 369.31464346 m
    curve_diff:  29.5668044455


     13%|█▎        | 158/1261 [03:20<23:25,  1.27s/it]

    368.953816217 m 463.816751153 m
    curve_diff:  94.8629349357


     13%|█▎        | 159/1261 [03:21<23:15,  1.27s/it]

    392.685918855 m 394.644966007 m
    curve_diff:  1.95904715152


     13%|█▎        | 160/1261 [03:22<23:07,  1.26s/it]

    371.126540382 m 361.233451218 m
    curve_diff:  9.89308916361


     13%|█▎        | 161/1261 [03:24<22:53,  1.25s/it]

    388.30372573 m 481.752786279 m
    curve_diff:  93.4490605489


     13%|█▎        | 162/1261 [03:25<22:42,  1.24s/it]

    401.053307298 m 399.480720145 m
    curve_diff:  1.57258715233


     13%|█▎        | 163/1261 [03:26<22:39,  1.24s/it]

    405.657077932 m 397.24348296 m
    curve_diff:  8.41359497194


     13%|█▎        | 164/1261 [03:27<22:45,  1.24s/it]

    416.11494661 m 341.193396019 m
    curve_diff:  74.9215505917


     13%|█▎        | 165/1261 [03:28<22:39,  1.24s/it]

    376.932384899 m 243.549808884 m
    curve_diff:  133.382576015


     13%|█▎        | 166/1261 [03:30<22:34,  1.24s/it]

    404.006919065 m 238.761441053 m
    curve_diff:  165.245478013


     13%|█▎        | 167/1261 [03:31<23:12,  1.27s/it]

    350.211520678 m 271.601321633 m
    curve_diff:  78.6101990454


     13%|█▎        | 168/1261 [03:32<22:59,  1.26s/it]

    395.938800263 m 277.840033585 m
    curve_diff:  118.098766678


     13%|█▎        | 169/1261 [03:34<22:52,  1.26s/it]

    441.925828307 m 262.568955536 m
    curve_diff:  179.356872771


     13%|█▎        | 170/1261 [03:35<22:40,  1.25s/it]

    458.355470672 m 263.09899035 m
    curve_diff:  195.256480322


     14%|█▎        | 171/1261 [03:36<22:30,  1.24s/it]

    435.678406963 m 262.318849876 m
    curve_diff:  173.359557086


     14%|█▎        | 172/1261 [03:37<22:28,  1.24s/it]

    426.340512739 m 257.267364752 m
    curve_diff:  169.073147987


     14%|█▎        | 173/1261 [03:38<22:25,  1.24s/it]

    380.359614486 m 270.608910308 m
    curve_diff:  109.750704177


     14%|█▍        | 174/1261 [03:40<22:25,  1.24s/it]

    326.984760282 m 238.290000458 m
    curve_diff:  88.6947598245


     14%|█▍        | 175/1261 [03:41<22:43,  1.26s/it]

    309.534734091 m 240.226027022 m
    curve_diff:  69.3087070686


     14%|█▍        | 176/1261 [03:42<22:40,  1.25s/it]

    298.905198457 m 203.048843349 m
    curve_diff:  95.8563551072


     14%|█▍        | 177/1261 [03:43<22:30,  1.25s/it]

    343.186458814 m 246.856619404 m
    curve_diff:  96.3298394092


     14%|█▍        | 178/1261 [03:45<22:23,  1.24s/it]

    335.958888973 m 219.424276001 m
    curve_diff:  116.534612971


     14%|█▍        | 179/1261 [03:46<22:22,  1.24s/it]

    342.733635129 m 229.810079528 m
    curve_diff:  112.923555601


     14%|█▍        | 180/1261 [03:47<22:23,  1.24s/it]

    350.654384964 m 245.755586435 m
    curve_diff:  104.898798529


     14%|█▍        | 181/1261 [03:48<22:33,  1.25s/it]

    327.2221537 m 290.316674754 m
    curve_diff:  36.9054789458


     14%|█▍        | 182/1261 [03:50<22:27,  1.25s/it]

    354.540781818 m 280.31253784 m
    curve_diff:  74.2282439789


     15%|█▍        | 183/1261 [03:51<22:29,  1.25s/it]

    325.761752833 m 347.145545299 m
    curve_diff:  21.3837924666


     15%|█▍        | 184/1261 [03:52<22:52,  1.27s/it]

    286.847415476 m 291.479467344 m
    curve_diff:  4.63205186776


     15%|█▍        | 185/1261 [03:54<22:41,  1.27s/it]

    266.924910372 m 250.177271105 m
    curve_diff:  16.7476392669


     15%|█▍        | 186/1261 [03:55<22:29,  1.26s/it]

    239.676275774 m 228.635271758 m
    curve_diff:  11.0410040157


     15%|█▍        | 187/1261 [03:56<22:19,  1.25s/it]

    243.091408267 m 256.063257281 m
    curve_diff:  12.9718490135


     15%|█▍        | 188/1261 [03:57<22:24,  1.25s/it]

    241.73461285 m 256.99430149 m
    curve_diff:  15.2596886401


     15%|█▍        | 189/1261 [03:59<22:16,  1.25s/it]

    251.985450445 m 275.677206639 m
    curve_diff:  23.6917561944


     15%|█▌        | 190/1261 [04:00<22:41,  1.27s/it]

    250.149663539 m 267.824721992 m
    curve_diff:  17.6750584524


     15%|█▌        | 191/1261 [04:01<22:24,  1.26s/it]

    246.692565368 m 239.357035962 m
    curve_diff:  7.33552940653


     15%|█▌        | 192/1261 [04:02<22:45,  1.28s/it]

    263.201827632 m 260.868651187 m
    curve_diff:  2.33317644568


     15%|█▌        | 193/1261 [04:04<22:32,  1.27s/it]

    252.511318298 m 255.58907245 m
    curve_diff:  3.07775415143


     15%|█▌        | 194/1261 [04:05<22:32,  1.27s/it]

    263.602045905 m 258.776187384 m
    curve_diff:  4.82585852062


     15%|█▌        | 195/1261 [04:06<22:29,  1.27s/it]

    265.635147589 m 260.348601635 m
    curve_diff:  5.28654595447


     16%|█▌        | 196/1261 [04:07<22:23,  1.26s/it]

    318.60608392 m 399.040475654 m
    curve_diff:  80.434391734


     16%|█▌        | 197/1261 [04:09<22:21,  1.26s/it]

    382.025802184 m 434.619777625 m
    curve_diff:  52.593975441


     16%|█▌        | 198/1261 [04:10<22:23,  1.26s/it]

    421.423132575 m 377.728557835 m
    curve_diff:  43.6945747402


     16%|█▌        | 199/1261 [04:11<22:16,  1.26s/it]

    433.774188725 m 337.048064279 m
    curve_diff:  96.7261244461


     16%|█▌        | 200/1261 [04:13<22:35,  1.28s/it]

    415.225042103 m 468.607780946 m
    curve_diff:  53.3827388429


     16%|█▌        | 201/1261 [04:14<22:21,  1.27s/it]

    430.102463749 m 499.395987372 m
    curve_diff:  69.2935236227


     16%|█▌        | 202/1261 [04:15<22:12,  1.26s/it]

    402.581894466 m 447.942714924 m
    curve_diff:  45.3608204576


     16%|█▌        | 203/1261 [04:16<22:09,  1.26s/it]

    395.156688572 m 445.013394519 m
    curve_diff:  49.8567059476


     16%|█▌        | 204/1261 [04:17<22:00,  1.25s/it]

    403.225883149 m 378.421316988 m
    curve_diff:  24.804566161


     16%|█▋        | 205/1261 [04:19<21:56,  1.25s/it]

    387.440129829 m 354.606561532 m
    curve_diff:  32.8335682966


     16%|█▋        | 206/1261 [04:20<21:52,  1.24s/it]

    374.225682376 m 309.057510078 m
    curve_diff:  65.1681722982


     16%|█▋        | 207/1261 [04:21<22:09,  1.26s/it]

    330.77249403 m 487.07456785 m
    curve_diff:  156.302073821


     16%|█▋        | 208/1261 [04:23<22:10,  1.26s/it]

    320.760223045 m 521.373076598 m
    curve_diff:  200.612853553


     17%|█▋        | 209/1261 [04:24<22:56,  1.31s/it]

    329.641117566 m 535.235040698 m
    curve_diff:  205.593923131


     17%|█▋        | 210/1261 [04:25<22:37,  1.29s/it]

    344.602703079 m 539.010650951 m
    curve_diff:  194.407947872


     17%|█▋        | 211/1261 [04:26<22:22,  1.28s/it]

    364.864496431 m 659.79838376 m
    curve_diff:  294.933887329


     17%|█▋        | 212/1261 [04:28<22:17,  1.28s/it]

    347.109227202 m 565.655246193 m
    curve_diff:  218.546018991


     17%|█▋        | 213/1261 [04:29<22:10,  1.27s/it]

    349.516452622 m 670.293871576 m
    curve_diff:  320.777418954


     17%|█▋        | 214/1261 [04:30<21:59,  1.26s/it]

    331.827968591 m 713.916191614 m
    curve_diff:  382.088223023


     17%|█▋        | 215/1261 [04:31<22:03,  1.27s/it]

    333.286114465 m 468.048548735 m
    curve_diff:  134.762434271


     17%|█▋        | 216/1261 [04:33<21:55,  1.26s/it]

    357.963772695 m 409.124316838 m
    curve_diff:  51.1605441428


     17%|█▋        | 217/1261 [04:34<22:20,  1.28s/it]

    386.340594788 m 362.40632492 m
    curve_diff:  23.9342698678


     17%|█▋        | 218/1261 [04:35<22:10,  1.28s/it]

    403.787701179 m 371.938602684 m
    curve_diff:  31.8490984952


     17%|█▋        | 219/1261 [04:37<22:03,  1.27s/it]

    353.606296258 m 435.903611648 m
    curve_diff:  82.2973153901


     17%|█▋        | 220/1261 [04:38<22:00,  1.27s/it]

    388.846570561 m 379.279853279 m
    curve_diff:  9.56671728183


     18%|█▊        | 221/1261 [04:39<21:52,  1.26s/it]

    380.885471037 m 425.943208419 m
    curve_diff:  45.057737382


     18%|█▊        | 222/1261 [04:40<21:53,  1.26s/it]

    412.971999908 m 485.547643704 m
    curve_diff:  72.5756437956


     18%|█▊        | 223/1261 [04:42<22:02,  1.27s/it]

    417.266249437 m 470.685021719 m
    curve_diff:  53.4187722817


     18%|█▊        | 224/1261 [04:43<22:10,  1.28s/it]

    402.055974921 m 383.88623966 m
    curve_diff:  18.1697352611


     18%|█▊        | 225/1261 [04:44<22:36,  1.31s/it]

    395.389818338 m 388.537934588 m
    curve_diff:  6.85188375038


     18%|█▊        | 226/1261 [04:46<22:18,  1.29s/it]

    385.004343604 m 307.647603898 m
    curve_diff:  77.356739706


     18%|█▊        | 227/1261 [04:47<22:06,  1.28s/it]

    412.132537877 m 295.558832817 m
    curve_diff:  116.57370506


     18%|█▊        | 228/1261 [04:48<21:59,  1.28s/it]

    417.159494901 m 302.996210312 m
    curve_diff:  114.163284589


     18%|█▊        | 229/1261 [04:49<21:48,  1.27s/it]

    469.073647848 m 303.805685151 m
    curve_diff:  165.267962697


     18%|█▊        | 230/1261 [04:51<21:44,  1.26s/it]

    487.317343871 m 278.275362699 m
    curve_diff:  209.041981172


     18%|█▊        | 231/1261 [04:52<21:40,  1.26s/it]

    432.906684279 m 301.59615029 m
    curve_diff:  131.310533988


     18%|█▊        | 232/1261 [04:53<21:57,  1.28s/it]

    444.724714973 m 290.98399376 m
    curve_diff:  153.740721214


     18%|█▊        | 233/1261 [04:55<22:09,  1.29s/it]

    433.411038237 m 333.076294948 m
    curve_diff:  100.334743289


     19%|█▊        | 234/1261 [04:56<22:14,  1.30s/it]

    435.94361664 m 341.322569672 m
    curve_diff:  94.6210469679


     19%|█▊        | 235/1261 [04:57<22:02,  1.29s/it]

    397.525469221 m 395.375332934 m
    curve_diff:  2.15013628667


     19%|█▊        | 236/1261 [04:58<21:56,  1.28s/it]

    356.102164651 m 250.364120011 m
    curve_diff:  105.738044641


     19%|█▉        | 237/1261 [05:00<21:43,  1.27s/it]

    406.586818985 m 283.533486698 m
    curve_diff:  123.053332287


     19%|█▉        | 238/1261 [05:01<21:35,  1.27s/it]

    389.578059595 m 243.732534 m
    curve_diff:  145.845525594


     19%|█▉        | 239/1261 [05:02<21:24,  1.26s/it]

    369.67221999 m 269.604757779 m
    curve_diff:  100.067462211


     19%|█▉        | 240/1261 [05:03<21:31,  1.26s/it]

    347.348069052 m 272.93890509 m
    curve_diff:  74.4091639623


     19%|█▉        | 241/1261 [05:05<21:24,  1.26s/it]

    370.720827472 m 297.892032025 m
    curve_diff:  72.8287954475


     19%|█▉        | 242/1261 [05:06<21:51,  1.29s/it]

    388.44027551 m 304.951133088 m
    curve_diff:  83.4891424213


     19%|█▉        | 243/1261 [05:07<21:35,  1.27s/it]

    355.651086069 m 326.213313816 m
    curve_diff:  29.4377722533


     19%|█▉        | 244/1261 [05:08<21:33,  1.27s/it]

    370.92706257 m 351.966298684 m
    curve_diff:  18.9607638857


     19%|█▉        | 245/1261 [05:10<21:27,  1.27s/it]

    349.133210013 m 236.015739854 m
    curve_diff:  113.117470158


     20%|█▉        | 246/1261 [05:11<21:26,  1.27s/it]

    336.455526043 m 228.274577154 m
    curve_diff:  108.180948889


     20%|█▉        | 247/1261 [05:12<21:16,  1.26s/it]

    346.068602523 m 263.930355083 m
    curve_diff:  82.1382474402


     20%|█▉        | 248/1261 [05:14<21:16,  1.26s/it]

    332.150241834 m 234.589446357 m
    curve_diff:  97.5607954771


     20%|█▉        | 249/1261 [05:15<21:10,  1.26s/it]

    337.31487775 m 255.760836424 m
    curve_diff:  81.554041326


     20%|█▉        | 250/1261 [05:16<21:43,  1.29s/it]

    341.644215758 m 275.068122063 m
    curve_diff:  66.5760936954


     20%|█▉        | 251/1261 [05:17<21:42,  1.29s/it]

    368.350729709 m 302.202269286 m
    curve_diff:  66.1484604231


     20%|█▉        | 252/1261 [05:19<21:31,  1.28s/it]

    363.437918682 m 307.851928173 m
    curve_diff:  55.5859905097


     20%|██        | 253/1261 [05:20<21:28,  1.28s/it]

    375.404346041 m 341.384708284 m
    curve_diff:  34.0196377569


     20%|██        | 254/1261 [05:21<21:44,  1.29s/it]

    407.411318446 m 365.893696487 m
    curve_diff:  41.5176219589


     20%|██        | 255/1261 [05:23<21:27,  1.28s/it]

    395.754565011 m 339.281859726 m
    curve_diff:  56.4727052851


     20%|██        | 256/1261 [05:24<21:14,  1.27s/it]

    385.677557865 m 369.825132316 m
    curve_diff:  15.8524255491


     20%|██        | 257/1261 [05:25<21:55,  1.31s/it]

    370.739113447 m 258.45421818 m
    curve_diff:  112.284895267


     20%|██        | 258/1261 [05:27<22:10,  1.33s/it]

    381.937934111 m 310.864386652 m
    curve_diff:  71.0735474591


     21%|██        | 259/1261 [05:28<21:53,  1.31s/it]

    404.57763736 m 262.320565104 m
    curve_diff:  142.257072256


     21%|██        | 260/1261 [05:29<21:42,  1.30s/it]

    397.739025731 m 240.651955581 m
    curve_diff:  157.08707015


     21%|██        | 261/1261 [05:30<21:27,  1.29s/it]

    407.772574625 m 282.271752216 m
    curve_diff:  125.500822409


     21%|██        | 262/1261 [05:32<21:17,  1.28s/it]

    403.433488553 m 279.539145713 m
    curve_diff:  123.894342839


     21%|██        | 263/1261 [05:33<21:06,  1.27s/it]

    420.342216281 m 300.860222589 m
    curve_diff:  119.481993692


     21%|██        | 264/1261 [05:34<21:10,  1.27s/it]

    429.384309357 m 298.08650118 m
    curve_diff:  131.297808177


     21%|██        | 265/1261 [05:35<21:06,  1.27s/it]

    424.120524857 m 341.04629723 m
    curve_diff:  83.074227627


     21%|██        | 266/1261 [05:37<21:22,  1.29s/it]

    462.874885648 m 352.813578332 m
    curve_diff:  110.061307317


     21%|██        | 267/1261 [05:38<21:12,  1.28s/it]

    434.821875113 m 429.710279633 m
    curve_diff:  5.11159547966


     21%|██▏       | 268/1261 [05:39<21:02,  1.27s/it]

    454.46886219 m 443.042782101 m
    curve_diff:  11.4260800895


     21%|██▏       | 269/1261 [05:41<20:55,  1.27s/it]

    478.798617607 m 436.7762916 m
    curve_diff:  42.0223260077


     21%|██▏       | 270/1261 [05:42<20:50,  1.26s/it]

    467.248300575 m 613.858592811 m
    curve_diff:  146.610292236


     21%|██▏       | 271/1261 [05:43<20:44,  1.26s/it]

    487.878217529 m 604.156404053 m
    curve_diff:  116.278186524


     22%|██▏       | 272/1261 [05:44<20:43,  1.26s/it]

    476.204077257 m 488.41203426 m
    curve_diff:  12.2079570032


     22%|██▏       | 273/1261 [05:46<20:39,  1.25s/it]

    490.422873785 m 342.60027397 m
    curve_diff:  147.822599815


     22%|██▏       | 274/1261 [05:47<20:42,  1.26s/it]

    471.153318315 m 388.49826975 m
    curve_diff:  82.6550485653


     22%|██▏       | 275/1261 [05:48<21:08,  1.29s/it]

    471.545110912 m 392.802617256 m
    curve_diff:  78.7424936565


     22%|██▏       | 276/1261 [05:49<20:56,  1.28s/it]

    515.116131373 m 406.171002668 m
    curve_diff:  108.945128706


     22%|██▏       | 277/1261 [05:51<20:49,  1.27s/it]

    588.335436762 m 420.92803442 m
    curve_diff:  167.407402342


     22%|██▏       | 278/1261 [05:52<20:39,  1.26s/it]

    601.928777653 m 441.119490449 m
    curve_diff:  160.809287204


     22%|██▏       | 279/1261 [05:53<20:47,  1.27s/it]

    589.718981727 m 498.253507865 m
    curve_diff:  91.4654738616


     22%|██▏       | 280/1261 [05:54<20:55,  1.28s/it]

    570.124686391 m 563.401896521 m
    curve_diff:  6.72278987033


     22%|██▏       | 281/1261 [05:56<21:08,  1.29s/it]

    629.621305693 m 494.864176495 m
    curve_diff:  134.757129198


     22%|██▏       | 282/1261 [05:58<25:41,  1.57s/it]

    611.455292547 m 403.569930477 m
    curve_diff:  207.88536207


     22%|██▏       | 283/1261 [06:00<25:51,  1.59s/it]

    649.437228968 m 488.189240234 m
    curve_diff:  161.247988733


     23%|██▎       | 284/1261 [06:01<24:10,  1.48s/it]

    637.441106219 m 441.072307959 m
    curve_diff:  196.36879826


     23%|██▎       | 285/1261 [06:02<23:06,  1.42s/it]

    696.094075202 m 489.322448777 m
    curve_diff:  206.771626425


     23%|██▎       | 286/1261 [06:04<23:09,  1.42s/it]

    718.82383343 m 456.116799715 m
    curve_diff:  262.707033715


     23%|██▎       | 287/1261 [06:05<22:42,  1.40s/it]

    726.95904509 m 489.350250776 m
    curve_diff:  237.608794314


     23%|██▎       | 288/1261 [06:06<22:05,  1.36s/it]

    766.363171031 m 513.871620714 m
    curve_diff:  252.491550317


     23%|██▎       | 289/1261 [06:08<22:42,  1.40s/it]

    792.067874188 m 447.932144047 m
    curve_diff:  344.135730142


     23%|██▎       | 290/1261 [06:09<23:29,  1.45s/it]

    803.316873129 m 479.256254099 m
    curve_diff:  324.06061903


     23%|██▎       | 291/1261 [06:11<22:39,  1.40s/it]

    803.377999899 m 599.63862111 m
    curve_diff:  203.739378789


     23%|██▎       | 292/1261 [06:12<21:55,  1.36s/it]

    812.079351746 m 696.17061141 m
    curve_diff:  115.908740336


     23%|██▎       | 293/1261 [06:13<22:51,  1.42s/it]

    838.458533316 m 814.817439458 m
    curve_diff:  23.641093858


     23%|██▎       | 294/1261 [06:15<23:29,  1.46s/it]

    821.688755674 m 533.78433213 m
    curve_diff:  287.904423544


     23%|██▎       | 295/1261 [06:16<22:32,  1.40s/it]

    785.508376926 m 710.386444546 m
    curve_diff:  75.1219323808


     23%|██▎       | 296/1261 [06:17<21:43,  1.35s/it]

    833.959802852 m 603.702066725 m
    curve_diff:  230.257736127


     24%|██▎       | 297/1261 [06:19<21:10,  1.32s/it]

    875.046693532 m 755.781546812 m
    curve_diff:  119.26514672


     24%|██▎       | 298/1261 [06:20<21:10,  1.32s/it]

    909.535712586 m 762.665850267 m
    curve_diff:  146.86986232


     24%|██▎       | 299/1261 [06:21<20:32,  1.28s/it]

    962.27344469 m 724.580031767 m
    curve_diff:  237.693412923


     24%|██▍       | 300/1261 [06:22<20:08,  1.26s/it]

    1094.49609927 m 758.172405238 m
    curve_diff:  336.323694034


     24%|██▍       | 301/1261 [06:24<20:18,  1.27s/it]

    1260.08874085 m 728.065655743 m
    curve_diff:  532.023085103


     24%|██▍       | 302/1261 [06:25<20:17,  1.27s/it]

    1353.6904853 m 719.734583918 m
    curve_diff:  633.955901377


     24%|██▍       | 303/1261 [06:26<20:03,  1.26s/it]

    1359.03215816 m 1006.19782761 m
    curve_diff:  352.834330555


     24%|██▍       | 304/1261 [06:27<19:46,  1.24s/it]

    1279.22436324 m 1683.0676155 m
    curve_diff:  403.843252261


     24%|██▍       | 305/1261 [06:29<20:01,  1.26s/it]

    1620.5975489 m 1840.80895312 m
    curve_diff:  220.211404214


     24%|██▍       | 306/1261 [06:30<20:04,  1.26s/it]

    1729.53107949 m 5285.58194064 m
    curve_diff:  3556.05086114


     24%|██▍       | 307/1261 [06:31<20:01,  1.26s/it]

    2220.75567894 m 5642.76537075 m
    curve_diff:  3422.0096918


     24%|██▍       | 308/1261 [06:32<19:45,  1.24s/it]

    2098.07762974 m 1590.53167775 m
    curve_diff:  507.545951991


     25%|██▍       | 309/1261 [06:34<19:33,  1.23s/it]

    2474.7873606 m 2913.76156179 m
    curve_diff:  438.974201196


     25%|██▍       | 310/1261 [06:35<19:28,  1.23s/it]

    3786.94028754 m 1105.50373436 m
    curve_diff:  2681.43655318


     25%|██▍       | 311/1261 [06:36<19:38,  1.24s/it]

    4330.48620018 m 1374.12137492 m
    curve_diff:  2956.36482526


     25%|██▍       | 312/1261 [06:37<19:44,  1.25s/it]

    4380.16639771 m 1291.42329501 m
    curve_diff:  3088.74310271


     25%|██▍       | 313/1261 [06:39<21:15,  1.35s/it]

    4464.80134159 m 1360.22485838 m
    curve_diff:  3104.57648321


     25%|██▍       | 314/1261 [06:40<21:48,  1.38s/it]

    11121.2459094 m 1290.2719633 m
    curve_diff:  9830.97394614


     25%|██▍       | 315/1261 [06:42<21:30,  1.36s/it]

    34502.9566731 m 1096.36058068 m
    curve_diff:  33406.5960924


     25%|██▌       | 316/1261 [06:43<20:50,  1.32s/it]

    44996.8426476 m 1024.47633674 m
    curve_diff:  43972.3663109


     25%|██▌       | 317/1261 [06:44<20:13,  1.29s/it]

    18264.3030151 m 918.242478878 m
    curve_diff:  17346.0605362


     25%|██▌       | 318/1261 [06:45<19:46,  1.26s/it]

    20257.2773112 m 713.893085269 m
    curve_diff:  19543.3842259


     25%|██▌       | 319/1261 [06:47<19:34,  1.25s/it]

    12431.3760734 m 760.216236339 m
    curve_diff:  11671.159837


     25%|██▌       | 320/1261 [06:48<19:36,  1.25s/it]

    8679.26372299 m 1033.98298908 m
    curve_diff:  7645.28073391


     25%|██▌       | 321/1261 [06:49<19:24,  1.24s/it]

    5280.80788984 m 913.384283446 m
    curve_diff:  4367.42360639


     26%|██▌       | 322/1261 [06:50<19:20,  1.24s/it]

    9053.64180489 m 910.044752724 m
    curve_diff:  8143.59705217


     26%|██▌       | 323/1261 [06:52<19:46,  1.27s/it]

    5673.35693117 m 1175.57507592 m
    curve_diff:  4497.78185524


     26%|██▌       | 324/1261 [06:53<19:44,  1.26s/it]

    5991.51406651 m 1007.5597779 m
    curve_diff:  4983.95428861


     26%|██▌       | 325/1261 [06:54<19:36,  1.26s/it]

    7315.90623348 m 1400.08582428 m
    curve_diff:  5915.8204092


     26%|██▌       | 326/1261 [06:55<19:26,  1.25s/it]

    7055.69478097 m 1449.576162 m
    curve_diff:  5606.11861897


     26%|██▌       | 327/1261 [06:57<19:19,  1.24s/it]

    5406.77249309 m 1552.2021237 m
    curve_diff:  3854.57036939


     26%|██▌       | 328/1261 [06:58<19:14,  1.24s/it]

    11103.0449425 m 2784.23352896 m
    curve_diff:  8318.81141355


     26%|██▌       | 329/1261 [06:59<19:03,  1.23s/it]

    6640.98710781 m 2229.5981547 m
    curve_diff:  4411.38895311


     26%|██▌       | 330/1261 [07:00<18:56,  1.22s/it]

    7659.84242161 m 4749.7230474 m
    curve_diff:  2910.11937421


     26%|██▌       | 331/1261 [07:01<19:11,  1.24s/it]

    7090.85748776 m 2053.08877397 m
    curve_diff:  5037.76871379


     26%|██▋       | 332/1261 [07:03<20:17,  1.31s/it]

    8468.67863805 m 981.658962177 m
    curve_diff:  7487.01967588


     26%|██▋       | 333/1261 [07:05<23:20,  1.51s/it]

    466000.312824 m 1616.35379533 m
    curve_diff:  464383.959029


     26%|██▋       | 334/1261 [07:07<24:38,  1.59s/it]

    10818.185748 m 1979.87732624 m
    curve_diff:  8838.30842175


     27%|██▋       | 335/1261 [07:08<22:51,  1.48s/it]

    9855.30542443 m 2434.37306445 m
    curve_diff:  7420.93235997


     27%|██▋       | 336/1261 [07:09<21:43,  1.41s/it]

    9303.13597522 m 1393.19118737 m
    curve_diff:  7909.94478785


     27%|██▋       | 337/1261 [07:10<20:47,  1.35s/it]

    9537.46319609 m 1723.27380072 m
    curve_diff:  7814.18939538


     27%|██▋       | 338/1261 [07:12<20:08,  1.31s/it]

    7006.2013795 m 2723.68136506 m
    curve_diff:  4282.52001443


     27%|██▋       | 339/1261 [07:13<19:51,  1.29s/it]

    31365.5228746 m 3860.59445605 m
    curve_diff:  27504.9284186


     27%|██▋       | 340/1261 [07:14<19:22,  1.26s/it]

    14451.345265 m 2960.08982787 m
    curve_diff:  11491.2554372


     27%|██▋       | 341/1261 [07:15<19:17,  1.26s/it]

    77071.7300664 m 1063.12260078 m
    curve_diff:  76008.6074656


     27%|██▋       | 342/1261 [07:17<19:37,  1.28s/it]

    7740.22555251 m 802.530699151 m
    curve_diff:  6937.69485336


     27%|██▋       | 343/1261 [07:18<19:35,  1.28s/it]

    8256.01209053 m 1431.78571272 m
    curve_diff:  6824.22637781


     27%|██▋       | 344/1261 [07:19<19:56,  1.30s/it]

    5084.45009653 m 1404.12059753 m
    curve_diff:  3680.329499


     27%|██▋       | 345/1261 [07:21<19:50,  1.30s/it]

    4297.2024814 m 945.323095124 m
    curve_diff:  3351.87938628


     27%|██▋       | 346/1261 [07:22<19:49,  1.30s/it]

    4794.62359739 m 1556.05948702 m
    curve_diff:  3238.56411036


     28%|██▊       | 347/1261 [07:23<20:27,  1.34s/it]

    5881.49649078 m 1003.84222363 m
    curve_diff:  4877.65426715


     28%|██▊       | 348/1261 [07:25<19:59,  1.31s/it]

    5100.29036144 m 1142.33372582 m
    curve_diff:  3957.95663562


     28%|██▊       | 349/1261 [07:26<20:51,  1.37s/it]

    4176.95688439 m 1392.13119651 m
    curve_diff:  2784.82568787


     28%|██▊       | 350/1261 [07:27<20:32,  1.35s/it]

    10464.2719638 m 1336.40467208 m
    curve_diff:  9127.8672917


     28%|██▊       | 351/1261 [07:29<20:17,  1.34s/it]

    6157.98342947 m 1687.5031971 m
    curve_diff:  4470.48023237


     28%|██▊       | 352/1261 [07:30<19:56,  1.32s/it]

    8386.81430024 m 1882.83173867 m
    curve_diff:  6503.98256157


     28%|██▊       | 353/1261 [07:31<19:40,  1.30s/it]

    26014.8846765 m 1965.64890816 m
    curve_diff:  24049.2357683


     28%|██▊       | 354/1261 [07:32<19:24,  1.28s/it]

    76129.0393046 m 2069.55105118 m
    curve_diff:  74059.4882534


     28%|██▊       | 355/1261 [07:34<19:51,  1.31s/it]

    22079.4763146 m 2878.36352689 m
    curve_diff:  19201.1127877


     28%|██▊       | 356/1261 [07:35<19:46,  1.31s/it]

    35208.4994806 m 3488.96892083 m
    curve_diff:  31719.5305598


     28%|██▊       | 357/1261 [07:36<19:41,  1.31s/it]

    2235.46221824 m 1639.63766057 m
    curve_diff:  595.82455767


     28%|██▊       | 358/1261 [07:38<19:30,  1.30s/it]

    2109.90909086 m 1441.90944587 m
    curve_diff:  667.999644987


     28%|██▊       | 359/1261 [07:39<19:24,  1.29s/it]

    1503.50681743 m 1075.08948395 m
    curve_diff:  428.417333482


     29%|██▊       | 360/1261 [07:40<19:20,  1.29s/it]

    1287.48164669 m 1010.22946742 m
    curve_diff:  277.252179268


     29%|██▊       | 361/1261 [07:42<19:18,  1.29s/it]

    1211.04069749 m 1755.92988242 m
    curve_diff:  544.889184932


     29%|██▊       | 362/1261 [07:43<19:20,  1.29s/it]

    1057.34832829 m 2999.25734552 m
    curve_diff:  1941.90901723


     29%|██▉       | 363/1261 [07:44<19:39,  1.31s/it]

    1061.2402397 m 3451.88723266 m
    curve_diff:  2390.64699297


     29%|██▉       | 364/1261 [07:45<19:27,  1.30s/it]

    1056.67769864 m 2206.06131705 m
    curve_diff:  1149.38361841


     29%|██▉       | 365/1261 [07:47<19:33,  1.31s/it]

    1084.54437429 m 1758.33903758 m
    curve_diff:  673.794663287


     29%|██▉       | 366/1261 [07:48<19:26,  1.30s/it]

    1251.20861212 m 1245.67692867 m
    curve_diff:  5.531683446


     29%|██▉       | 367/1261 [07:49<19:27,  1.31s/it]

    1195.69921629 m 1526.58402509 m
    curve_diff:  330.884808795


     29%|██▉       | 368/1261 [07:51<19:23,  1.30s/it]

    1171.63381631 m 1133.10694661 m
    curve_diff:  38.5268697029


     29%|██▉       | 369/1261 [07:52<19:18,  1.30s/it]

    1416.20098786 m 1235.67468674 m
    curve_diff:  180.526301117


     29%|██▉       | 370/1261 [07:53<19:00,  1.28s/it]

    1588.11886157 m 1472.34466757 m
    curve_diff:  115.774193999


     29%|██▉       | 371/1261 [07:55<19:14,  1.30s/it]

    2109.71323563 m 1627.83264857 m
    curve_diff:  481.880587053


     30%|██▉       | 372/1261 [07:56<19:16,  1.30s/it]

    2225.39262098 m 1673.55357475 m
    curve_diff:  551.83904623


     30%|██▉       | 373/1261 [07:57<19:11,  1.30s/it]

    3189.73033518 m 1784.20126543 m
    curve_diff:  1405.52906975


     30%|██▉       | 374/1261 [07:58<19:06,  1.29s/it]

    4143.38683713 m 7471.95573593 m
    curve_diff:  3328.56889881


     30%|██▉       | 375/1261 [08:00<19:08,  1.30s/it]

    11463.4387672 m 11422.2091887 m
    curve_diff:  41.2295784734


     30%|██▉       | 376/1261 [08:01<19:02,  1.29s/it]

    5528.76809074 m 4752.72735193 m
    curve_diff:  776.04073881


     30%|██▉       | 377/1261 [08:02<18:59,  1.29s/it]

    4475.60571651 m 4457.24265993 m
    curve_diff:  18.3630565802


     30%|██▉       | 378/1261 [08:04<18:51,  1.28s/it]

    3196.34893196 m 2272.78382343 m
    curve_diff:  923.565108526


     30%|███       | 379/1261 [08:05<19:07,  1.30s/it]

    2703.29735646 m 4156.12790173 m
    curve_diff:  1452.83054527


     30%|███       | 380/1261 [08:06<19:17,  1.31s/it]

    2273.81253402 m 10309.9388133 m
    curve_diff:  8036.12627925


     30%|███       | 381/1261 [08:08<19:33,  1.33s/it]

    1988.17885034 m 8734.87686438 m
    curve_diff:  6746.69801403


     30%|███       | 382/1261 [08:09<19:37,  1.34s/it]

    2126.48317222 m 2783.47600499 m
    curve_diff:  656.99283277


     30%|███       | 383/1261 [08:10<19:37,  1.34s/it]

    2277.56101154 m 2387.98324848 m
    curve_diff:  110.422236944


     30%|███       | 384/1261 [08:12<19:19,  1.32s/it]

    2094.96700664 m 8999.97077937 m
    curve_diff:  6905.00377273


     31%|███       | 385/1261 [08:13<19:07,  1.31s/it]

    1904.93603144 m 2470.83514062 m
    curve_diff:  565.899109181


     31%|███       | 386/1261 [08:14<19:00,  1.30s/it]

    2179.42549396 m 2550.82338806 m
    curve_diff:  371.397894101


     31%|███       | 387/1261 [08:16<19:26,  1.33s/it]

    2380.35622087 m 1769.76444877 m
    curve_diff:  610.591772104


     31%|███       | 388/1261 [08:17<19:10,  1.32s/it]

    2641.82738327 m 2399.37100495 m
    curve_diff:  242.456378322


     31%|███       | 389/1261 [08:18<19:13,  1.32s/it]

    2496.60805952 m 2008.67048266 m
    curve_diff:  487.937576858


     31%|███       | 390/1261 [08:20<19:20,  1.33s/it]

    2295.58527654 m 3151.91110786 m
    curve_diff:  856.325831322


     31%|███       | 391/1261 [08:21<19:28,  1.34s/it]

    2482.34448187 m 11246.3098362 m
    curve_diff:  8763.96535432


     31%|███       | 392/1261 [08:22<19:35,  1.35s/it]

    3269.29407739 m 3400.6239158 m
    curve_diff:  131.329838413


     31%|███       | 393/1261 [08:24<22:36,  1.56s/it]

    3430.56326335 m 1260.06584159 m
    curve_diff:  2170.49742177


     31%|███       | 394/1261 [08:26<23:28,  1.62s/it]

    3369.26429104 m 1771.79165099 m
    curve_diff:  1597.47264005


     31%|███▏      | 395/1261 [08:28<22:41,  1.57s/it]

    4897.66281643 m 2464.49971951 m
    curve_diff:  2433.16309691


     31%|███▏      | 396/1261 [08:29<22:41,  1.57s/it]

    7661.44307567 m 3708.66743039 m
    curve_diff:  3952.77564529


     31%|███▏      | 397/1261 [08:31<22:27,  1.56s/it]

    17529.6992796 m 4311.50464874 m
    curve_diff:  13218.1946308


     32%|███▏      | 398/1261 [08:32<22:12,  1.54s/it]

    20926.7208691 m 5191.3092342 m
    curve_diff:  15735.4116349


     32%|███▏      | 399/1261 [08:34<22:13,  1.55s/it]

    56108.8100707 m 15838.5138608 m
    curve_diff:  40270.2962099


     32%|███▏      | 400/1261 [08:35<22:29,  1.57s/it]

    11253.0285589 m 3242.63329081 m
    curve_diff:  8010.39526807


     32%|███▏      | 401/1261 [08:38<26:22,  1.84s/it]

    143834.382709 m 1412.69825809 m
    curve_diff:  142421.68445


     32%|███▏      | 402/1261 [08:40<26:35,  1.86s/it]

    21991.4988235 m 1283.0580824 m
    curve_diff:  20708.4407411


     32%|███▏      | 403/1261 [08:42<27:05,  1.89s/it]

    11749.7668386 m 3664.59956635 m
    curve_diff:  8085.16727221


     32%|███▏      | 404/1261 [08:43<26:29,  1.86s/it]

    20660.7750758 m 10200.3227531 m
    curve_diff:  10460.4523227


     32%|███▏      | 405/1261 [08:45<25:47,  1.81s/it]

    24816.5369065 m 6115.62197408 m
    curve_diff:  18700.9149324


     32%|███▏      | 406/1261 [08:47<26:44,  1.88s/it]

    53226.5539265 m 2292.45956377 m
    curve_diff:  50934.0943627


     32%|███▏      | 407/1261 [08:49<24:59,  1.76s/it]

    3169542.81993 m 1355.34628377 m
    curve_diff:  3168187.47365


     32%|███▏      | 408/1261 [08:51<25:27,  1.79s/it]

    120664.323106 m 4140.14028572 m
    curve_diff:  116524.18282


     32%|███▏      | 409/1261 [08:52<24:41,  1.74s/it]

    18379.4969185 m 8047.07109171 m
    curve_diff:  10332.4258268


     33%|███▎      | 410/1261 [08:54<23:35,  1.66s/it]

    13841.3994255 m 4018.7377466 m
    curve_diff:  9822.66167885


     33%|███▎      | 411/1261 [08:55<22:58,  1.62s/it]

    5326.44274901 m 58640.0733954 m
    curve_diff:  53313.6306464


     33%|███▎      | 412/1261 [08:57<23:10,  1.64s/it]

    5480.92293275 m 3314.17434679 m
    curve_diff:  2166.74858596


     33%|███▎      | 413/1261 [08:59<24:20,  1.72s/it]

    4301.09159516 m 1350.80696028 m
    curve_diff:  2950.28463488


     33%|███▎      | 414/1261 [09:00<23:25,  1.66s/it]

    3052.93618095 m 1476.57091771 m
    curve_diff:  1576.36526324


     33%|███▎      | 415/1261 [09:02<22:48,  1.62s/it]

    2559.4715763 m 188601.496737 m
    curve_diff:  186042.025161


     33%|███▎      | 416/1261 [09:03<22:21,  1.59s/it]

    2773.65848616 m 7236.94377524 m
    curve_diff:  4463.28528907


     33%|███▎      | 417/1261 [09:05<21:38,  1.54s/it]

    2408.66052727 m 1220.29790318 m
    curve_diff:  1188.36262409


     33%|███▎      | 418/1261 [09:06<21:11,  1.51s/it]

    2677.2698164 m 1935.72232456 m
    curve_diff:  741.547491845


     33%|███▎      | 419/1261 [09:08<20:20,  1.45s/it]

    2967.17742462 m 2082.38175003 m
    curve_diff:  884.795674591


     33%|███▎      | 420/1261 [09:09<20:55,  1.49s/it]

    1998.88446568 m 6444.03552344 m
    curve_diff:  4445.15105776


     33%|███▎      | 421/1261 [09:10<20:17,  1.45s/it]

    2011.71281573 m 13138.9024471 m
    curve_diff:  11127.1896314


     33%|███▎      | 422/1261 [09:12<20:11,  1.44s/it]

    4352.58153599 m 2532.22241958 m
    curve_diff:  1820.35911641


     34%|███▎      | 423/1261 [09:13<20:10,  1.44s/it]

    3640.0986693 m 2198.37301877 m
    curve_diff:  1441.72565053


     34%|███▎      | 424/1261 [09:15<20:32,  1.47s/it]

    3942.15710485 m 1714.22248701 m
    curve_diff:  2227.93461784


     34%|███▎      | 425/1261 [09:16<20:28,  1.47s/it]

    4281.66063138 m 1304.94957926 m
    curve_diff:  2976.71105212


     34%|███▍      | 426/1261 [09:18<20:30,  1.47s/it]

    4105.68812604 m 1400.58553764 m
    curve_diff:  2705.10258839


     34%|███▍      | 427/1261 [09:19<20:58,  1.51s/it]

    12097.6195595 m 3797.91880636 m
    curve_diff:  8299.70075313


     34%|███▍      | 428/1261 [09:21<20:29,  1.48s/it]

    35445.7981394 m 3768.55377684 m
    curve_diff:  31677.2443625


     34%|███▍      | 429/1261 [09:22<20:35,  1.49s/it]

    11157.4006867 m 3426.75479954 m
    curve_diff:  7730.6458872


     34%|███▍      | 430/1261 [09:24<20:18,  1.47s/it]

    6261.22385493 m 4935.27098917 m
    curve_diff:  1325.95286576


     34%|███▍      | 431/1261 [09:25<20:27,  1.48s/it]

    5565.86247493 m 7218.47182885 m
    curve_diff:  1652.60935392


     34%|███▍      | 432/1261 [09:27<20:04,  1.45s/it]

    3664.07454648 m 2894.02640424 m
    curve_diff:  770.048142242


     34%|███▍      | 433/1261 [09:28<20:18,  1.47s/it]

    3692.15268714 m 2671.36891508 m
    curve_diff:  1020.78377206


     34%|███▍      | 434/1261 [09:30<20:46,  1.51s/it]

    2805.52491245 m 5210.20817387 m
    curve_diff:  2404.68326142


     34%|███▍      | 435/1261 [09:31<20:52,  1.52s/it]

    2182.6218062 m 1702.95845006 m
    curve_diff:  479.663356136


     35%|███▍      | 436/1261 [09:33<21:13,  1.54s/it]

    1994.69022194 m 1052.23875425 m
    curve_diff:  942.451467685


     35%|███▍      | 437/1261 [09:34<20:56,  1.53s/it]

    1780.75363661 m 2529.95193724 m
    curve_diff:  749.198300627


     35%|███▍      | 438/1261 [09:36<20:54,  1.52s/it]

    2156.47462333 m 1460.8816024 m
    curve_diff:  695.593020927


     35%|███▍      | 439/1261 [09:37<20:08,  1.47s/it]

    1855.2386005 m 2045.63218163 m
    curve_diff:  190.393581129


     35%|███▍      | 440/1261 [09:39<20:22,  1.49s/it]

    1903.30164266 m 1653.8616101 m
    curve_diff:  249.440032557


     35%|███▍      | 441/1261 [09:40<20:12,  1.48s/it]

    1802.03824481 m 2077.31235271 m
    curve_diff:  275.2741079


     35%|███▌      | 442/1261 [09:42<20:19,  1.49s/it]

    2393.23111577 m 2130.21075615 m
    curve_diff:  263.020359619


     35%|███▌      | 443/1261 [09:43<20:18,  1.49s/it]

    1967.72995663 m 9512.73233112 m
    curve_diff:  7545.00237449


     35%|███▌      | 444/1261 [09:45<20:00,  1.47s/it]

    2171.97050055 m 2248.98183151 m
    curve_diff:  77.0113309603


     35%|███▌      | 445/1261 [09:46<19:47,  1.46s/it]

    2244.05462689 m 3091.09773251 m
    curve_diff:  847.043105625


     35%|███▌      | 446/1261 [09:47<19:33,  1.44s/it]

    1951.01951495 m 1604.95122385 m
    curve_diff:  346.068291098


     35%|███▌      | 447/1261 [09:49<20:00,  1.47s/it]

    2663.4618852 m 1580.85317037 m
    curve_diff:  1082.60871483


     36%|███▌      | 448/1261 [09:51<19:56,  1.47s/it]

    3156.08593789 m 1919.57706324 m
    curve_diff:  1236.50887465


     36%|███▌      | 449/1261 [09:52<20:13,  1.49s/it]

    5621.48042615 m 8415.64397089 m
    curve_diff:  2794.16354474


     36%|███▌      | 450/1261 [09:54<21:40,  1.60s/it]

    8158.15428866 m 16825.2238809 m
    curve_diff:  8667.06959224


     36%|███▌      | 451/1261 [09:55<20:56,  1.55s/it]

    74587.7621694 m 12220.8280182 m
    curve_diff:  62366.9341512


     36%|███▌      | 452/1261 [09:57<20:57,  1.55s/it]

    14543.1045335 m 286305.112525 m
    curve_diff:  271762.007991


     36%|███▌      | 453/1261 [09:58<20:47,  1.54s/it]

    7936.76851004 m 7794.0390017 m
    curve_diff:  142.729508348


     36%|███▌      | 454/1261 [10:00<20:57,  1.56s/it]

    3776.05356671 m 9696.71337799 m
    curve_diff:  5920.65981128


     36%|███▌      | 455/1261 [10:02<21:35,  1.61s/it]

    2812.13133059 m 68904.8350629 m
    curve_diff:  66092.7037323


     36%|███▌      | 456/1261 [10:03<21:21,  1.59s/it]

    2276.3957942 m 3768.66653947 m
    curve_diff:  1492.27074527


     36%|███▌      | 457/1261 [10:05<20:23,  1.52s/it]

    1916.03116664 m 3252.86583577 m
    curve_diff:  1336.83466913


     36%|███▋      | 458/1261 [10:06<19:40,  1.47s/it]

    2438.50160652 m 3105.80204782 m
    curve_diff:  667.300441296


     36%|███▋      | 459/1261 [10:08<20:01,  1.50s/it]

    2096.80998429 m 6421.28437404 m
    curve_diff:  4324.47438975


     36%|███▋      | 460/1261 [10:09<19:11,  1.44s/it]

    2441.6941611 m 3916.26573981 m
    curve_diff:  1474.57157871


     37%|███▋      | 461/1261 [10:10<19:10,  1.44s/it]

    2167.77040655 m 1461.31249003 m
    curve_diff:  706.457916527


     37%|███▋      | 462/1261 [10:12<20:53,  1.57s/it]

    2032.72841433 m 2468.98967014 m
    curve_diff:  436.261255802


     37%|███▋      | 463/1261 [10:14<21:48,  1.64s/it]

    2197.09874884 m 3393.29547419 m
    curve_diff:  1196.19672535


     37%|███▋      | 464/1261 [10:15<21:19,  1.61s/it]

    2383.33501978 m 1280.15975879 m
    curve_diff:  1103.17526099


     37%|███▋      | 465/1261 [10:17<21:52,  1.65s/it]

    2884.71715085 m 1459.53103146 m
    curve_diff:  1425.18611939


     37%|███▋      | 466/1261 [10:19<20:56,  1.58s/it]

    2804.48819277 m 2085.65360948 m
    curve_diff:  718.834583293


     37%|███▋      | 467/1261 [10:20<20:12,  1.53s/it]

    2707.96718195 m 2946.44644565 m
    curve_diff:  238.479263696


     37%|███▋      | 468/1261 [10:21<19:25,  1.47s/it]

    3269.27562447 m 2714.70418527 m
    curve_diff:  554.571439208


     37%|███▋      | 469/1261 [10:23<18:47,  1.42s/it]

    4908.10654718 m 46270.5968512 m
    curve_diff:  41362.490304


     37%|███▋      | 470/1261 [10:25<23:16,  1.77s/it]

    6724.07829266 m 3515.13184567 m
    curve_diff:  3208.94644699


     37%|███▋      | 471/1261 [10:27<22:12,  1.69s/it]

    8124.29829654 m 4335.43731308 m
    curve_diff:  3788.86098346


     37%|███▋      | 472/1261 [10:28<21:18,  1.62s/it]

    13013.8118214 m 4065.2708216 m
    curve_diff:  8948.54099982


     38%|███▊      | 473/1261 [10:30<20:23,  1.55s/it]

    25103.1561818 m 6005.51778985 m
    curve_diff:  19097.6383919


     38%|███▊      | 474/1261 [10:31<19:55,  1.52s/it]

    21241.4433664 m 2566.95902248 m
    curve_diff:  18674.484344


     38%|███▊      | 475/1261 [10:32<19:08,  1.46s/it]

    12745.4779194 m 12182.2819223 m
    curve_diff:  563.195997114


     38%|███▊      | 476/1261 [10:34<18:47,  1.44s/it]

    14239.6084889 m 4680.02153878 m
    curve_diff:  9559.58695014


     38%|███▊      | 477/1261 [10:35<19:05,  1.46s/it]

    8040.73020038 m 8136.20559321 m
    curve_diff:  95.4753928305


     38%|███▊      | 478/1261 [10:37<18:42,  1.43s/it]

    5725.00960385 m 3099.38784544 m
    curve_diff:  2625.6217584


     38%|███▊      | 479/1261 [10:38<18:32,  1.42s/it]

    7797.04463693 m 3325.46399353 m
    curve_diff:  4471.5806434


     38%|███▊      | 480/1261 [10:40<18:35,  1.43s/it]

    3902.64865021 m 33548.8256803 m
    curve_diff:  29646.1770301


     38%|███▊      | 481/1261 [10:41<18:17,  1.41s/it]

    3799.25302688 m 18888.8722549 m
    curve_diff:  15089.619228


     38%|███▊      | 482/1261 [10:42<17:57,  1.38s/it]

    3869.58271852 m 16132.863918 m
    curve_diff:  12263.2811995


     38%|███▊      | 483/1261 [10:44<17:58,  1.39s/it]

    4413.52989554 m 53030.1201656 m
    curve_diff:  48616.5902701


     38%|███▊      | 484/1261 [10:45<18:32,  1.43s/it]

    6415.66882706 m 2047.76749602 m
    curve_diff:  4367.90133104


     38%|███▊      | 485/1261 [10:47<18:15,  1.41s/it]

    5654.17676496 m 3534.04634069 m
    curve_diff:  2120.13042427


     39%|███▊      | 486/1261 [10:48<18:11,  1.41s/it]

    7434.78347844 m 969.565975893 m
    curve_diff:  6465.21750255


     39%|███▊      | 487/1261 [10:49<17:55,  1.39s/it]

    15595.5394668 m 1241.48868879 m
    curve_diff:  14354.050778


     39%|███▊      | 488/1261 [10:51<17:40,  1.37s/it]

    14023.4979352 m 27075.3582042 m
    curve_diff:  13051.860269


     39%|███▉      | 489/1261 [10:52<17:36,  1.37s/it]

    30325.301698 m 17286.0397361 m
    curve_diff:  13039.2619619


     39%|███▉      | 490/1261 [10:53<17:21,  1.35s/it]

    19039.4461459 m 4774.87200285 m
    curve_diff:  14264.574143


     39%|███▉      | 491/1261 [10:55<17:02,  1.33s/it]

    54876.5290264 m 5473.78628536 m
    curve_diff:  49402.742741


     39%|███▉      | 492/1261 [10:56<16:58,  1.32s/it]

    35732.636143 m 3195.58715482 m
    curve_diff:  32537.0489882


     39%|███▉      | 493/1261 [10:57<16:38,  1.30s/it]

    29269.3030768 m 3856.62197223 m
    curve_diff:  25412.6811045


     39%|███▉      | 494/1261 [10:58<16:21,  1.28s/it]

    64275.2201809 m 9841.87370812 m
    curve_diff:  54433.3464728


     39%|███▉      | 495/1261 [11:00<16:14,  1.27s/it]

    25138.7488483 m 5772.17249769 m
    curve_diff:  19366.5763506


     39%|███▉      | 496/1261 [11:01<16:08,  1.27s/it]

    16046.3872847 m 1535.12452499 m
    curve_diff:  14511.2627597


     39%|███▉      | 497/1261 [11:02<16:03,  1.26s/it]

    7730.06132578 m 1037.7545656 m
    curve_diff:  6692.30676017


     39%|███▉      | 498/1261 [11:03<15:53,  1.25s/it]

    4662.34285919 m 2166.43414702 m
    curve_diff:  2495.90871216


     40%|███▉      | 499/1261 [11:05<15:44,  1.24s/it]

    4401.93391978 m 2587.49339915 m
    curve_diff:  1814.44052063


     40%|███▉      | 500/1261 [11:06<15:47,  1.25s/it]

    4775.17768855 m 2526.2653544 m
    curve_diff:  2248.91233415


     40%|███▉      | 501/1261 [11:07<15:59,  1.26s/it]

    4959.26190617 m 4222.31363052 m
    curve_diff:  736.948275655


     40%|███▉      | 502/1261 [11:08<15:50,  1.25s/it]

    5218.17670126 m 3266.90607823 m
    curve_diff:  1951.27062303


     40%|███▉      | 503/1261 [11:10<15:45,  1.25s/it]

    7826.88155863 m 2499.27374129 m
    curve_diff:  5327.60781733


     40%|███▉      | 504/1261 [11:11<15:39,  1.24s/it]

    2391.62705138 m 5002.69968623 m
    curve_diff:  2611.07263485


     40%|████      | 505/1261 [11:12<15:38,  1.24s/it]

    3950.56401959 m 11781.9511495 m
    curve_diff:  7831.38712994


     40%|████      | 506/1261 [11:13<15:35,  1.24s/it]

    5176.37707226 m 35835.4142355 m
    curve_diff:  30659.0371632


     40%|████      | 507/1261 [11:15<15:45,  1.25s/it]

    3754.13872908 m 25718.6753012 m
    curve_diff:  21964.5365721


     40%|████      | 508/1261 [11:16<15:35,  1.24s/it]

    5838.94674112 m 5754.3584458 m
    curve_diff:  84.5882953162


     40%|████      | 509/1261 [11:17<15:43,  1.25s/it]

    90118.0485592 m 1575.88517708 m
    curve_diff:  88542.1633821


     40%|████      | 510/1261 [11:18<15:36,  1.25s/it]

    33989.6878602 m 875.730705028 m
    curve_diff:  33113.9571551


     41%|████      | 511/1261 [11:19<15:30,  1.24s/it]

    6020.51192479 m 1090.47576798 m
    curve_diff:  4930.03615681


     41%|████      | 512/1261 [11:21<15:26,  1.24s/it]

    3011.20229186 m 1265.73047074 m
    curve_diff:  1745.47182113


     41%|████      | 513/1261 [11:22<15:26,  1.24s/it]

    12210.9897949 m 12357.9229444 m
    curve_diff:  146.933149423


     41%|████      | 514/1261 [11:23<15:23,  1.24s/it]

    56120.1284149 m 16389.1203972 m
    curve_diff:  39731.0080177


     41%|████      | 515/1261 [11:24<15:22,  1.24s/it]

    2711.52507585 m 6476.53191739 m
    curve_diff:  3765.00684154


     41%|████      | 516/1261 [11:26<15:20,  1.24s/it]

    2361.46043075 m 18289.3374494 m
    curve_diff:  15927.8770186


     41%|████      | 517/1261 [11:27<15:29,  1.25s/it]

    808.138714851 m 14919.5930492 m
    curve_diff:  14111.4543343


     41%|████      | 518/1261 [11:28<15:43,  1.27s/it]

    769.662248498 m 1998.30084313 m
    curve_diff:  1228.63859463


     41%|████      | 519/1261 [11:29<15:36,  1.26s/it]

    911.133255495 m 3258.35695806 m
    curve_diff:  2347.22370257


     41%|████      | 520/1261 [11:31<15:27,  1.25s/it]

    2476.1377604 m 15560.150561 m
    curve_diff:  13084.0128006


     41%|████▏     | 521/1261 [11:32<15:24,  1.25s/it]

    647.500272449 m 2715.28861515 m
    curve_diff:  2067.7883427


     41%|████▏     | 522/1261 [11:33<15:21,  1.25s/it]

    632.286431015 m 12323.1219584 m
    curve_diff:  11690.8355274


     41%|████▏     | 523/1261 [11:34<15:21,  1.25s/it]

    710.208394607 m 5249.13120086 m
    curve_diff:  4538.92280625


     42%|████▏     | 524/1261 [11:36<15:19,  1.25s/it]

    1739.22554491 m 426.46732036 m
    curve_diff:  1312.75822455


     42%|████▏     | 525/1261 [11:37<15:16,  1.25s/it]

    756.727626213 m 419.749910569 m
    curve_diff:  336.977715644


     42%|████▏     | 526/1261 [11:38<15:39,  1.28s/it]

    2248.22407608 m 280.561209096 m
    curve_diff:  1967.66286698


     42%|████▏     | 527/1261 [11:40<15:30,  1.27s/it]

    1918.60325489 m 326.975473378 m
    curve_diff:  1591.62778151


     42%|████▏     | 528/1261 [11:41<15:20,  1.26s/it]

    1953.80252238 m 364.169021637 m
    curve_diff:  1589.63350074


     42%|████▏     | 529/1261 [11:42<15:21,  1.26s/it]

    1836.44501558 m 505.308336605 m
    curve_diff:  1331.13667898


     42%|████▏     | 530/1261 [11:43<15:13,  1.25s/it]

    344.523273522 m 29396.4354592 m
    curve_diff:  29051.9121857


     42%|████▏     | 531/1261 [11:44<15:03,  1.24s/it]

    464.848551432 m 21352.4640462 m
    curve_diff:  20887.6154948


     42%|████▏     | 532/1261 [11:46<14:57,  1.23s/it]

    471.100294812 m 1521.38987684 m
    curve_diff:  1050.28958202


     42%|████▏     | 533/1261 [11:47<14:52,  1.23s/it]

    195.309624563 m 969.628217152 m
    curve_diff:  774.318592589


     42%|████▏     | 534/1261 [11:48<15:03,  1.24s/it]

    249.642783086 m 1444.93917123 m
    curve_diff:  1195.29638814


     42%|████▏     | 535/1261 [11:49<15:05,  1.25s/it]

    420.513166497 m 2097.83351947 m
    curve_diff:  1677.32035297


     43%|████▎     | 536/1261 [11:51<14:58,  1.24s/it]

    219.882872705 m 463.675291859 m
    curve_diff:  243.792419154


     43%|████▎     | 537/1261 [11:52<14:50,  1.23s/it]

    357.821337042 m 1059.68375245 m
    curve_diff:  701.862415406


     43%|████▎     | 538/1261 [11:53<14:43,  1.22s/it]

    263.413323904 m 731.281221009 m
    curve_diff:  467.867897104


     43%|████▎     | 539/1261 [11:54<14:43,  1.22s/it]

    243.39970458 m 539.964357256 m
    curve_diff:  296.564652676


     43%|████▎     | 540/1261 [11:56<14:39,  1.22s/it]

    809.039492875 m 3285.63216888 m
    curve_diff:  2476.59267601


     43%|████▎     | 541/1261 [11:57<14:37,  1.22s/it]

    551.070375327 m 3691.52568034 m
    curve_diff:  3140.45530501


     43%|████▎     | 542/1261 [11:58<14:34,  1.22s/it]

    212.37431647 m 2690.83191983 m
    curve_diff:  2478.45760336


     43%|████▎     | 543/1261 [11:59<14:46,  1.24s/it]

    261.658117551 m 2807.80475711 m
    curve_diff:  2546.14663956


     43%|████▎     | 544/1261 [12:00<14:39,  1.23s/it]

    559.154480749 m 2597.98956766 m
    curve_diff:  2038.83508692


     43%|████▎     | 545/1261 [12:02<14:35,  1.22s/it]

    1502.11130065 m 879.660712375 m
    curve_diff:  622.450588272


     43%|████▎     | 546/1261 [12:03<14:31,  1.22s/it]

    367.127315906 m 547.854248554 m
    curve_diff:  180.726932648


     43%|████▎     | 547/1261 [12:04<14:24,  1.21s/it]

    867.41386355 m 678.549254322 m
    curve_diff:  188.864609228


     43%|████▎     | 548/1261 [12:05<14:21,  1.21s/it]

    2157.26208227 m 660.71008219 m
    curve_diff:  1496.55200008


     44%|████▎     | 549/1261 [12:06<14:19,  1.21s/it]

    961.401803275 m 633.94848799 m
    curve_diff:  327.453315284


     44%|████▎     | 550/1261 [12:08<14:16,  1.20s/it]

    571.462819713 m 954.296711431 m
    curve_diff:  382.833891719


     44%|████▎     | 551/1261 [12:09<14:14,  1.20s/it]

    1567.64831839 m 695.69267112 m
    curve_diff:  871.955647271


     44%|████▍     | 552/1261 [12:10<14:44,  1.25s/it]

    972.797947215 m 770.388437633 m
    curve_diff:  202.409509582


     44%|████▍     | 553/1261 [12:11<14:34,  1.24s/it]

    7379.6044683 m 785.111306119 m
    curve_diff:  6594.49316218


     44%|████▍     | 554/1261 [12:13<14:29,  1.23s/it]

    4155.52785009 m 942.718756295 m
    curve_diff:  3212.8090938


     44%|████▍     | 555/1261 [12:14<14:19,  1.22s/it]

    466.393062677 m 1300.25670634 m
    curve_diff:  833.863643664


     44%|████▍     | 556/1261 [12:15<14:16,  1.22s/it]

    187.559275832 m 941.814256692 m
    curve_diff:  754.254980859


     44%|████▍     | 557/1261 [12:16<14:17,  1.22s/it]

    570.928098278 m 1016.30082263 m
    curve_diff:  445.372724357


     44%|████▍     | 558/1261 [12:17<14:14,  1.22s/it]

    413.535210545 m 1197.14076795 m
    curve_diff:  783.605557404


     44%|████▍     | 559/1261 [12:19<14:09,  1.21s/it]

    2296.18162386 m 1017.1706861 m
    curve_diff:  1279.01093776


     44%|████▍     | 560/1261 [12:20<14:22,  1.23s/it]

    2536.18691765 m 670.906594901 m
    curve_diff:  1865.28032274


     44%|████▍     | 561/1261 [12:21<14:23,  1.23s/it]

    1238.98469305 m 728.312188779 m
    curve_diff:  510.672504269


     45%|████▍     | 562/1261 [12:22<14:23,  1.24s/it]

    4323.91144985 m 780.391843637 m
    curve_diff:  3543.51960621


     45%|████▍     | 563/1261 [12:24<14:48,  1.27s/it]

    586.737697579 m 2625.66417306 m
    curve_diff:  2038.92647548


     45%|████▍     | 564/1261 [12:25<14:51,  1.28s/it]

    452.83046089 m 815.114499996 m
    curve_diff:  362.284039106


     45%|████▍     | 565/1261 [12:26<14:42,  1.27s/it]

    1037.77689548 m 624.445364829 m
    curve_diff:  413.331530654


     45%|████▍     | 566/1261 [12:28<14:31,  1.25s/it]

    201.030286657 m 1724.13133543 m
    curve_diff:  1523.10104878


     45%|████▍     | 567/1261 [12:29<14:19,  1.24s/it]

    609.475989957 m 1347.73389449 m
    curve_diff:  738.257904532


     45%|████▌     | 568/1261 [12:30<14:12,  1.23s/it]

    307.34990923 m 1896.15511425 m
    curve_diff:  1588.80520502


     45%|████▌     | 569/1261 [12:31<14:21,  1.24s/it]

    132.970042249 m 1240.84665601 m
    curve_diff:  1107.87661376


     45%|████▌     | 570/1261 [12:32<14:15,  1.24s/it]

    974.476688053 m 1509.54567931 m
    curve_diff:  535.068991254


     45%|████▌     | 571/1261 [12:34<14:07,  1.23s/it]

    234.128914458 m 2217.80535206 m
    curve_diff:  1983.67643761


     45%|████▌     | 572/1261 [12:35<14:03,  1.22s/it]

    254.361386724 m 1218.34700656 m
    curve_diff:  963.985619835


     45%|████▌     | 573/1261 [12:36<13:58,  1.22s/it]

    648.272157669 m 2563.8743613 m
    curve_diff:  1915.60220363


     46%|████▌     | 574/1261 [12:37<13:56,  1.22s/it]

    355.872884634 m 1938.3633005 m
    curve_diff:  1582.49041587


     46%|████▌     | 575/1261 [12:38<13:53,  1.22s/it]

    662.937318408 m 1662.21315604 m
    curve_diff:  999.275837635


     46%|████▌     | 576/1261 [12:40<13:51,  1.21s/it]

    474.359952598 m 957.441725774 m
    curve_diff:  483.081773176


     46%|████▌     | 577/1261 [12:41<13:59,  1.23s/it]

    3381.83544064 m 6649.54616399 m
    curve_diff:  3267.71072336


     46%|████▌     | 578/1261 [12:42<14:14,  1.25s/it]

    531.393080567 m 957.983768809 m
    curve_diff:  426.590688242


     46%|████▌     | 579/1261 [12:43<14:01,  1.23s/it]

    401.90352015 m 665.688342209 m
    curve_diff:  263.784822059


     46%|████▌     | 580/1261 [12:45<13:57,  1.23s/it]

    341.128887048 m 781.527355821 m
    curve_diff:  440.398468772


     46%|████▌     | 581/1261 [12:46<13:47,  1.22s/it]

    227.42920102 m 843.228731038 m
    curve_diff:  615.799530018


     46%|████▌     | 582/1261 [12:47<13:51,  1.22s/it]

    3451.18613746 m 829.023998099 m
    curve_diff:  2622.16213937


     46%|████▌     | 583/1261 [12:48<13:46,  1.22s/it]

    1610.69007213 m 808.731168146 m
    curve_diff:  801.958903986


     46%|████▋     | 584/1261 [12:50<13:45,  1.22s/it]

    601.404762601 m 657.450895644 m
    curve_diff:  56.0461330432


     46%|████▋     | 585/1261 [12:51<13:48,  1.23s/it]

    323.439960998 m 452.481497843 m
    curve_diff:  129.041536845


     46%|████▋     | 586/1261 [12:52<14:02,  1.25s/it]

    2382.01363784 m 2321.91565551 m
    curve_diff:  60.0979823347


     47%|████▋     | 587/1261 [12:53<13:58,  1.24s/it]

    1322.40261355 m 31856.9208177 m
    curve_diff:  30534.5182041


     47%|████▋     | 588/1261 [12:55<13:51,  1.24s/it]

    600.70942789 m 10896.264862 m
    curve_diff:  10295.5554341


     47%|████▋     | 589/1261 [12:56<13:46,  1.23s/it]

    106.184222993 m 4790.82624081 m
    curve_diff:  4684.64201782


     47%|████▋     | 590/1261 [12:57<13:43,  1.23s/it]

    1641.44554942 m 1932.76612756 m
    curve_diff:  291.320578141


     47%|████▋     | 591/1261 [12:58<13:39,  1.22s/it]

    618.389085815 m 1127.73775379 m
    curve_diff:  509.348667975


     47%|████▋     | 592/1261 [12:59<13:37,  1.22s/it]

    136.141507878 m 1875.00311754 m
    curve_diff:  1738.86160966


     47%|████▋     | 593/1261 [13:01<13:33,  1.22s/it]

    227.908759461 m 1308.01375469 m
    curve_diff:  1080.10499523


     47%|████▋     | 594/1261 [13:02<13:37,  1.23s/it]

    372.189716857 m 1077.05930073 m
    curve_diff:  704.869583874


     47%|████▋     | 595/1261 [13:03<13:49,  1.25s/it]

    401.764442448 m 23790.2907769 m
    curve_diff:  23388.5263344


     47%|████▋     | 596/1261 [13:04<13:47,  1.24s/it]

    212.817694438 m 11409.2001241 m
    curve_diff:  11196.3824297


     47%|████▋     | 597/1261 [13:06<13:43,  1.24s/it]

    1766.4290814 m 3047.16917088 m
    curve_diff:  1280.74008947


     47%|████▋     | 598/1261 [13:07<13:37,  1.23s/it]

    920.854242316 m 1138.16104988 m
    curve_diff:  217.306807563


     48%|████▊     | 599/1261 [13:08<13:33,  1.23s/it]

    1316.52853606 m 13837.7271729 m
    curve_diff:  12521.1986369


     48%|████▊     | 600/1261 [13:09<13:33,  1.23s/it]

    2241.99833998 m 580.924122912 m
    curve_diff:  1661.07421707


     48%|████▊     | 601/1261 [13:11<13:31,  1.23s/it]

    1029.68441105 m 434.880257559 m
    curve_diff:  594.80415349


     48%|████▊     | 602/1261 [13:12<13:33,  1.23s/it]

    4714.47882586 m 349.330789062 m
    curve_diff:  4365.1480368


     48%|████▊     | 603/1261 [13:13<13:42,  1.25s/it]

    544.225677678 m 319.46806868 m
    curve_diff:  224.757608998


     48%|████▊     | 604/1261 [13:14<13:39,  1.25s/it]

    683.198545271 m 263.293188481 m
    curve_diff:  419.905356789


     48%|████▊     | 605/1261 [13:16<13:35,  1.24s/it]

    656.860620942 m 285.394774957 m
    curve_diff:  371.465845984


     48%|████▊     | 606/1261 [13:17<13:35,  1.24s/it]

    467.155854392 m 317.152947811 m
    curve_diff:  150.002906581


     48%|████▊     | 607/1261 [13:18<13:32,  1.24s/it]

    351.568310148 m 285.972619023 m
    curve_diff:  65.5956911251


     48%|████▊     | 608/1261 [13:19<13:30,  1.24s/it]

    384.814377242 m 301.304408236 m
    curve_diff:  83.5099690064


     48%|████▊     | 609/1261 [13:20<13:28,  1.24s/it]

    375.018591285 m 308.711790033 m
    curve_diff:  66.3068012511


     48%|████▊     | 610/1261 [13:22<13:28,  1.24s/it]

    338.895002716 m 346.180388144 m
    curve_diff:  7.28538542866


     48%|████▊     | 611/1261 [13:23<13:30,  1.25s/it]

    348.096538437 m 353.55524514 m
    curve_diff:  5.45870670353


     49%|████▊     | 612/1261 [13:24<13:50,  1.28s/it]

    347.265313511 m 374.258827664 m
    curve_diff:  26.9935141532


     49%|████▊     | 613/1261 [13:26<13:43,  1.27s/it]

    344.751971713 m 379.345593972 m
    curve_diff:  34.5936222592


     49%|████▊     | 614/1261 [13:27<13:38,  1.26s/it]

    358.789611402 m 435.343405491 m
    curve_diff:  76.5537940884


     49%|████▉     | 615/1261 [13:28<13:34,  1.26s/it]

    353.06717934 m 409.54206793 m
    curve_diff:  56.4748885901


     49%|████▉     | 616/1261 [13:29<13:33,  1.26s/it]

    350.456485651 m 535.508352863 m
    curve_diff:  185.051867212


     49%|████▉     | 617/1261 [13:31<13:27,  1.25s/it]

    319.850734369 m 669.451819166 m
    curve_diff:  349.601084798


     49%|████▉     | 618/1261 [13:32<13:26,  1.25s/it]

    305.349176665 m 519.615251945 m
    curve_diff:  214.266075279


     49%|████▉     | 619/1261 [13:33<13:24,  1.25s/it]

    286.570563041 m 494.346148453 m
    curve_diff:  207.775585413


     49%|████▉     | 620/1261 [13:34<13:36,  1.27s/it]

    323.194612821 m 585.945801971 m
    curve_diff:  262.75118915


     49%|████▉     | 621/1261 [13:36<13:28,  1.26s/it]

    314.50898556 m 517.365404926 m
    curve_diff:  202.856419366


     49%|████▉     | 622/1261 [13:37<13:27,  1.26s/it]

    320.597352484 m 652.620710278 m
    curve_diff:  332.023357794


     49%|████▉     | 623/1261 [13:38<13:25,  1.26s/it]

    324.955850333 m 560.335094419 m
    curve_diff:  235.379244087


     49%|████▉     | 624/1261 [13:39<13:21,  1.26s/it]

    354.112052676 m 588.326030235 m
    curve_diff:  234.213977558


     50%|████▉     | 625/1261 [13:41<13:17,  1.25s/it]

    313.832528695 m 362.821433849 m
    curve_diff:  48.9889051539


     50%|████▉     | 626/1261 [13:42<13:16,  1.25s/it]

    335.502236529 m 386.182737845 m
    curve_diff:  50.6805013154


     50%|████▉     | 627/1261 [13:43<13:10,  1.25s/it]

    352.570396941 m 381.525033909 m
    curve_diff:  28.9546369683


     50%|████▉     | 628/1261 [13:44<13:20,  1.26s/it]

    360.573520587 m 320.563888059 m
    curve_diff:  40.009632528


     50%|████▉     | 629/1261 [13:46<13:14,  1.26s/it]

    353.632273817 m 346.638774351 m
    curve_diff:  6.99349946572


     50%|████▉     | 630/1261 [13:47<13:08,  1.25s/it]

    422.088116241 m 401.455830243 m
    curve_diff:  20.6322859985


     50%|█████     | 631/1261 [13:48<13:05,  1.25s/it]

    443.625420096 m 568.68065961 m
    curve_diff:  125.055239515


     50%|█████     | 632/1261 [13:49<13:05,  1.25s/it]

    531.979190773 m 694.327080975 m
    curve_diff:  162.347890202


     50%|█████     | 633/1261 [13:51<13:05,  1.25s/it]

    559.348562927 m 633.718362248 m
    curve_diff:  74.3697993217


     50%|█████     | 634/1261 [13:52<13:02,  1.25s/it]

    518.148702427 m 465.942572441 m
    curve_diff:  52.2061299857


     50%|█████     | 635/1261 [13:53<13:02,  1.25s/it]

    537.242699201 m 404.446818775 m
    curve_diff:  132.795880426


     50%|█████     | 636/1261 [13:54<13:02,  1.25s/it]

    506.568354787 m 349.360054067 m
    curve_diff:  157.20830072


     51%|█████     | 637/1261 [13:56<13:13,  1.27s/it]

    365.001696428 m 289.722729731 m
    curve_diff:  75.2789666971


     51%|█████     | 638/1261 [13:57<13:07,  1.26s/it]

    412.697856776 m 305.89309275 m
    curve_diff:  106.804764026


     51%|█████     | 639/1261 [13:58<13:03,  1.26s/it]

    424.836227738 m 344.950871479 m
    curve_diff:  79.8853562585


     51%|█████     | 640/1261 [13:59<12:58,  1.25s/it]

    569.597581899 m 388.433487011 m
    curve_diff:  181.164094888


     51%|█████     | 641/1261 [14:01<12:56,  1.25s/it]

    652.393610265 m 633.765226737 m
    curve_diff:  18.6283835279


     51%|█████     | 642/1261 [14:02<12:58,  1.26s/it]

    789.182882025 m 600.029400376 m
    curve_diff:  189.153481649


     51%|█████     | 643/1261 [14:03<12:55,  1.26s/it]

    716.855683658 m 597.396201551 m
    curve_diff:  119.459482107


     51%|█████     | 644/1261 [14:05<12:51,  1.25s/it]

    582.929836488 m 416.088080052 m
    curve_diff:  166.841756437


     51%|█████     | 645/1261 [14:06<13:00,  1.27s/it]

    540.162368677 m 432.581671908 m
    curve_diff:  107.58069677


     51%|█████     | 646/1261 [14:07<12:54,  1.26s/it]

    462.886675104 m 465.318808658 m
    curve_diff:  2.43213355409


     51%|█████▏    | 647/1261 [14:08<12:49,  1.25s/it]

    472.642287154 m 467.466799797 m
    curve_diff:  5.17548735705


     51%|█████▏    | 648/1261 [14:10<12:46,  1.25s/it]

    465.65575094 m 463.450834734 m
    curve_diff:  2.20491620599


     51%|█████▏    | 649/1261 [14:11<12:45,  1.25s/it]

    436.76895187 m 520.166524562 m
    curve_diff:  83.3975726924


     52%|█████▏    | 650/1261 [14:12<12:47,  1.26s/it]

    452.094290762 m 529.201193884 m
    curve_diff:  77.1069031219


     52%|█████▏    | 651/1261 [14:13<12:48,  1.26s/it]

    445.99827218 m 515.663441 m
    curve_diff:  69.66516882


     52%|█████▏    | 652/1261 [14:15<12:45,  1.26s/it]

    436.498801759 m 499.052452797 m
    curve_diff:  62.5536510379


     52%|█████▏    | 653/1261 [14:16<12:53,  1.27s/it]

    389.197900688 m 442.724700622 m
    curve_diff:  53.5267999337


     52%|█████▏    | 654/1261 [14:17<12:49,  1.27s/it]

    375.056026608 m 467.477107999 m
    curve_diff:  92.4210813903


     52%|█████▏    | 655/1261 [14:18<12:45,  1.26s/it]

    372.808530718 m 499.663608347 m
    curve_diff:  126.855077629


     52%|█████▏    | 656/1261 [14:20<12:39,  1.26s/it]

    383.891193664 m 473.993565368 m
    curve_diff:  90.1023717033


     52%|█████▏    | 657/1261 [14:21<12:32,  1.25s/it]

    379.020855743 m 558.016510288 m
    curve_diff:  178.995654545


     52%|█████▏    | 658/1261 [14:22<12:31,  1.25s/it]

    381.418451193 m 552.36437454 m
    curve_diff:  170.945923347


     52%|█████▏    | 659/1261 [14:23<12:45,  1.27s/it]

    396.492324917 m 569.602389308 m
    curve_diff:  173.110064391


     52%|█████▏    | 660/1261 [14:25<13:02,  1.30s/it]

    410.16260438 m 599.859550477 m
    curve_diff:  189.696946096


     52%|█████▏    | 661/1261 [14:26<13:00,  1.30s/it]

    433.743523872 m 594.995682496 m
    curve_diff:  161.252158624


     52%|█████▏    | 662/1261 [14:27<13:08,  1.32s/it]

    461.28720546 m 671.842588354 m
    curve_diff:  210.555382894


     53%|█████▎    | 663/1261 [14:29<12:51,  1.29s/it]

    482.729412635 m 711.984010431 m
    curve_diff:  229.254597796


     53%|█████▎    | 664/1261 [14:30<12:40,  1.27s/it]

    480.020673601 m 698.322660546 m
    curve_diff:  218.301986945


     53%|█████▎    | 665/1261 [14:31<12:31,  1.26s/it]

    469.782152691 m 584.996814251 m
    curve_diff:  115.214661559


     53%|█████▎    | 666/1261 [14:32<12:31,  1.26s/it]

    470.538810493 m 451.288439735 m
    curve_diff:  19.2503707581


     53%|█████▎    | 667/1261 [14:34<12:25,  1.26s/it]

    505.653923147 m 404.123596847 m
    curve_diff:  101.5303263


     53%|█████▎    | 668/1261 [14:35<12:25,  1.26s/it]

    540.49698048 m 456.812564408 m
    curve_diff:  83.6844160716


     53%|█████▎    | 669/1261 [14:36<12:19,  1.25s/it]

    578.620988397 m 484.268737657 m
    curve_diff:  94.3522507401


     53%|█████▎    | 670/1261 [14:37<12:26,  1.26s/it]

    576.335131413 m 526.266940093 m
    curve_diff:  50.0681913201


     53%|█████▎    | 671/1261 [14:39<12:20,  1.26s/it]

    626.364530418 m 526.483219012 m
    curve_diff:  99.8813114063


     53%|█████▎    | 672/1261 [14:40<12:17,  1.25s/it]

    617.601993638 m 535.504022625 m
    curve_diff:  82.0979710124


     53%|█████▎    | 673/1261 [14:41<12:13,  1.25s/it]

    656.735620587 m 507.073733536 m
    curve_diff:  149.661887051


     53%|█████▎    | 674/1261 [14:42<12:11,  1.25s/it]

    606.277755757 m 553.587945867 m
    curve_diff:  52.6898098898


     54%|█████▎    | 675/1261 [14:44<12:11,  1.25s/it]

    685.370161878 m 509.46735384 m
    curve_diff:  175.902808037


     54%|█████▎    | 676/1261 [14:45<12:07,  1.24s/it]

    619.992630309 m 515.512763016 m
    curve_diff:  104.479867292


     54%|█████▎    | 677/1261 [14:46<12:03,  1.24s/it]

    550.7748168 m 519.615789407 m
    curve_diff:  31.1590273929


     54%|█████▍    | 678/1261 [14:47<12:12,  1.26s/it]

    594.92710129 m 522.870009896 m
    curve_diff:  72.0570913939


     54%|█████▍    | 679/1261 [14:49<12:15,  1.26s/it]

    524.612030598 m 478.607782241 m
    curve_diff:  46.0042483578


     54%|█████▍    | 680/1261 [14:50<12:09,  1.26s/it]

    507.133243789 m 515.355608959 m
    curve_diff:  8.22236517078


     54%|█████▍    | 681/1261 [14:51<12:05,  1.25s/it]

    468.711420603 m 471.058258038 m
    curve_diff:  2.34683743555


     54%|█████▍    | 682/1261 [14:52<12:04,  1.25s/it]

    463.055993399 m 504.793920015 m
    curve_diff:  41.7379266163


     54%|█████▍    | 683/1261 [14:54<12:04,  1.25s/it]

    498.358232785 m 513.362339818 m
    curve_diff:  15.0041070332


     54%|█████▍    | 684/1261 [14:55<11:58,  1.25s/it]

    470.409923058 m 593.097788204 m
    curve_diff:  122.687865146


     54%|█████▍    | 685/1261 [14:56<11:55,  1.24s/it]

    455.50432699 m 596.92009552 m
    curve_diff:  141.415768531


     54%|█████▍    | 686/1261 [14:57<11:59,  1.25s/it]

    443.569902979 m 555.696884684 m
    curve_diff:  112.126981704


     54%|█████▍    | 687/1261 [14:59<12:08,  1.27s/it]

    447.990505368 m 524.519153521 m
    curve_diff:  76.5286481533


     55%|█████▍    | 688/1261 [15:00<12:01,  1.26s/it]

    445.243982123 m 501.658880702 m
    curve_diff:  56.4148985792


     55%|█████▍    | 689/1261 [15:01<11:56,  1.25s/it]

    449.662783643 m 579.358336089 m
    curve_diff:  129.695552446


     55%|█████▍    | 690/1261 [15:02<11:53,  1.25s/it]

    403.112040198 m 507.671974293 m
    curve_diff:  104.559934096


     55%|█████▍    | 691/1261 [15:04<11:52,  1.25s/it]

    388.739767392 m 467.752930641 m
    curve_diff:  79.0131632491


     55%|█████▍    | 692/1261 [15:05<11:47,  1.24s/it]

    376.205521283 m 477.770953644 m
    curve_diff:  101.56543236


     55%|█████▍    | 693/1261 [15:06<11:49,  1.25s/it]

    374.782806549 m 454.723823198 m
    curve_diff:  79.9410166499


     55%|█████▌    | 694/1261 [15:07<11:48,  1.25s/it]

    361.073800254 m 473.226181615 m
    curve_diff:  112.152381362


     55%|█████▌    | 695/1261 [15:09<12:01,  1.28s/it]

    360.364400226 m 508.244322337 m
    curve_diff:  147.879922111


     55%|█████▌    | 696/1261 [15:10<11:55,  1.27s/it]

    364.264444218 m 492.737226444 m
    curve_diff:  128.472782226


     55%|█████▌    | 697/1261 [15:11<11:50,  1.26s/it]

    384.870237407 m 461.916016824 m
    curve_diff:  77.045779417


     55%|█████▌    | 698/1261 [15:13<11:45,  1.25s/it]

    363.879519424 m 459.110799223 m
    curve_diff:  95.2312797994


     55%|█████▌    | 699/1261 [15:14<11:48,  1.26s/it]

    384.641500266 m 419.551679308 m
    curve_diff:  34.9101790425


     56%|█████▌    | 700/1261 [15:15<11:42,  1.25s/it]

    377.018887886 m 378.135528738 m
    curve_diff:  1.11664085182


     56%|█████▌    | 701/1261 [15:16<11:43,  1.26s/it]

    382.992084039 m 366.035210073 m
    curve_diff:  16.9568739665


     56%|█████▌    | 702/1261 [15:18<11:41,  1.25s/it]

    394.680041725 m 385.666837116 m
    curve_diff:  9.01320460889


     56%|█████▌    | 703/1261 [15:19<11:43,  1.26s/it]

    392.659535441 m 389.304007129 m
    curve_diff:  3.35552831236


     56%|█████▌    | 704/1261 [15:20<11:58,  1.29s/it]

    393.429347755 m 382.86087794 m
    curve_diff:  10.5684698154


     56%|█████▌    | 705/1261 [15:21<11:52,  1.28s/it]

    402.181058323 m 349.020385689 m
    curve_diff:  53.1606726342


     56%|█████▌    | 706/1261 [15:23<11:47,  1.27s/it]

    440.371623305 m 370.426955272 m
    curve_diff:  69.944668033


     56%|█████▌    | 707/1261 [15:24<11:39,  1.26s/it]

    429.120171318 m 378.465962583 m
    curve_diff:  50.6542087349


     56%|█████▌    | 708/1261 [15:25<11:52,  1.29s/it]

    432.302315119 m 380.977274363 m
    curve_diff:  51.3250407552


     56%|█████▌    | 709/1261 [15:27<11:46,  1.28s/it]

    432.845882751 m 395.019083538 m
    curve_diff:  37.8267992132


     56%|█████▋    | 710/1261 [15:28<11:39,  1.27s/it]

    425.939073283 m 379.916344504 m
    curve_diff:  46.0227287785


     56%|█████▋    | 711/1261 [15:29<11:34,  1.26s/it]

    443.794231231 m 396.850280796 m
    curve_diff:  46.9439504352


     56%|█████▋    | 712/1261 [15:30<11:50,  1.29s/it]

    482.679773232 m 461.196879883 m
    curve_diff:  21.4828933498


     57%|█████▋    | 713/1261 [15:32<11:40,  1.28s/it]

    492.896558319 m 423.61279518 m
    curve_diff:  69.2837631386


     57%|█████▋    | 714/1261 [15:33<11:36,  1.27s/it]

    436.601839077 m 377.413253855 m
    curve_diff:  59.188585222


     57%|█████▋    | 715/1261 [15:34<12:12,  1.34s/it]

    381.229069617 m 366.445759054 m
    curve_diff:  14.7833105638


     57%|█████▋    | 716/1261 [15:36<11:56,  1.31s/it]

    389.177114023 m 402.345255972 m
    curve_diff:  13.1681419491


     57%|█████▋    | 717/1261 [15:37<11:38,  1.28s/it]

    421.172402518 m 410.944299144 m
    curve_diff:  10.2281033741


     57%|█████▋    | 718/1261 [15:38<11:31,  1.27s/it]

    398.880573649 m 414.399300224 m
    curve_diff:  15.5187265758


     57%|█████▋    | 719/1261 [15:39<11:25,  1.27s/it]

    425.174983557 m 446.533752821 m
    curve_diff:  21.3587692643


     57%|█████▋    | 720/1261 [15:41<11:34,  1.28s/it]

    434.580145076 m 427.853035598 m
    curve_diff:  6.72710947749


     57%|█████▋    | 721/1261 [15:42<11:25,  1.27s/it]

    422.461214561 m 474.30794439 m
    curve_diff:  51.8467298285


     57%|█████▋    | 722/1261 [15:43<11:21,  1.26s/it]

    433.600414567 m 510.064824442 m
    curve_diff:  76.4644098755


     57%|█████▋    | 723/1261 [15:44<11:18,  1.26s/it]

    446.933709674 m 493.776383609 m
    curve_diff:  46.8426739344


     57%|█████▋    | 724/1261 [15:46<11:16,  1.26s/it]

    440.285608998 m 528.848850743 m
    curve_diff:  88.5632417452


     57%|█████▋    | 725/1261 [15:47<11:15,  1.26s/it]

    480.059423306 m 574.237916207 m
    curve_diff:  94.1784929015


     58%|█████▊    | 726/1261 [15:48<11:14,  1.26s/it]

    436.737433874 m 563.102963493 m
    curve_diff:  126.36552962


     58%|█████▊    | 727/1261 [15:49<11:12,  1.26s/it]

    426.674049108 m 601.987838545 m
    curve_diff:  175.313789437


     58%|█████▊    | 728/1261 [15:51<11:21,  1.28s/it]

    411.05453041 m 414.020216232 m
    curve_diff:  2.96568582211


     58%|█████▊    | 729/1261 [15:52<11:18,  1.28s/it]

    404.940086973 m 449.993042836 m
    curve_diff:  45.0529558629


     58%|█████▊    | 730/1261 [15:53<11:12,  1.27s/it]

    408.007111541 m 421.557737038 m
    curve_diff:  13.550625497


     58%|█████▊    | 731/1261 [15:55<11:07,  1.26s/it]

    409.256859507 m 430.952837415 m
    curve_diff:  21.6959779079


     58%|█████▊    | 732/1261 [15:56<11:06,  1.26s/it]

    424.909479106 m 432.967024271 m
    curve_diff:  8.05754516565


     58%|█████▊    | 733/1261 [15:57<11:07,  1.26s/it]

    409.63675224 m 386.875039688 m
    curve_diff:  22.7617125511


     58%|█████▊    | 734/1261 [15:58<11:07,  1.27s/it]

    431.273442013 m 410.71928021 m
    curve_diff:  20.5541618027


     58%|█████▊    | 735/1261 [16:00<11:05,  1.27s/it]

    421.196068239 m 411.168008004 m
    curve_diff:  10.0280602347


     58%|█████▊    | 736/1261 [16:01<11:05,  1.27s/it]

    448.775273634 m 426.865660429 m
    curve_diff:  21.909613205


     58%|█████▊    | 737/1261 [16:02<11:13,  1.29s/it]

    469.026967107 m 421.613712633 m
    curve_diff:  47.4132544735


     59%|█████▊    | 738/1261 [16:03<11:10,  1.28s/it]

    412.755910798 m 375.311869211 m
    curve_diff:  37.4440415873


     59%|█████▊    | 739/1261 [16:05<11:04,  1.27s/it]

    431.567130005 m 377.335677092 m
    curve_diff:  54.2314529133


     59%|█████▊    | 740/1261 [16:06<11:01,  1.27s/it]

    419.859831927 m 345.611575938 m
    curve_diff:  74.2482559898


     59%|█████▉    | 741/1261 [16:07<10:57,  1.26s/it]

    433.773971968 m 349.549596764 m
    curve_diff:  84.2243752042


     59%|█████▉    | 742/1261 [16:08<10:53,  1.26s/it]

    447.833742173 m 355.8300194 m
    curve_diff:  92.0037227734


     59%|█████▉    | 743/1261 [16:10<10:53,  1.26s/it]

    450.349795393 m 361.697463415 m
    curve_diff:  88.6523319775


     59%|█████▉    | 744/1261 [16:11<10:49,  1.26s/it]

    435.795158417 m 377.723855212 m
    curve_diff:  58.0713032054


     59%|█████▉    | 745/1261 [16:12<11:01,  1.28s/it]

    461.247835352 m 388.465225856 m
    curve_diff:  72.7826094961


     59%|█████▉    | 746/1261 [16:14<10:55,  1.27s/it]

    493.422055171 m 411.500397959 m
    curve_diff:  81.9216572116


     59%|█████▉    | 747/1261 [16:15<10:51,  1.27s/it]

    485.425005629 m 410.697650251 m
    curve_diff:  74.7273553779


     59%|█████▉    | 748/1261 [16:16<10:48,  1.26s/it]

    516.776519122 m 450.055034553 m
    curve_diff:  66.7214845686


     59%|█████▉    | 749/1261 [16:17<10:49,  1.27s/it]

    532.995547982 m 476.267798109 m
    curve_diff:  56.7277498735


     59%|█████▉    | 750/1261 [16:19<10:47,  1.27s/it]

    495.587463617 m 449.891337849 m
    curve_diff:  45.6961257687


     60%|█████▉    | 751/1261 [16:20<10:45,  1.26s/it]

    469.055182066 m 459.556266729 m
    curve_diff:  9.49891533733


     60%|█████▉    | 752/1261 [16:21<10:41,  1.26s/it]

    459.260790932 m 488.4039787 m
    curve_diff:  29.1431877679


     60%|█████▉    | 753/1261 [16:23<10:53,  1.29s/it]

    433.153524795 m 477.617186275 m
    curve_diff:  44.4636614805


     60%|█████▉    | 754/1261 [16:24<11:11,  1.32s/it]

    412.859097242 m 467.9264374 m
    curve_diff:  55.0673401578


     60%|█████▉    | 755/1261 [16:25<11:27,  1.36s/it]

    377.163916142 m 480.053891383 m
    curve_diff:  102.88997524


     60%|█████▉    | 756/1261 [16:27<11:11,  1.33s/it]

    411.766026545 m 481.61113738 m
    curve_diff:  69.8451108354


     60%|██████    | 757/1261 [16:28<10:56,  1.30s/it]

    414.841148013 m 512.086223068 m
    curve_diff:  97.2450750553


     60%|██████    | 758/1261 [16:29<10:49,  1.29s/it]

    393.52347499 m 614.024840024 m
    curve_diff:  220.501365034


     60%|██████    | 759/1261 [16:30<10:45,  1.29s/it]

    404.439157735 m 535.429373035 m
    curve_diff:  130.9902153


     60%|██████    | 760/1261 [16:32<10:42,  1.28s/it]

    397.461161291 m 573.836989837 m
    curve_diff:  176.375828546


     60%|██████    | 761/1261 [16:33<10:52,  1.30s/it]

    381.440666437 m 638.926783104 m
    curve_diff:  257.486116667


     60%|██████    | 762/1261 [16:34<10:46,  1.30s/it]

    422.402696613 m 766.496556435 m
    curve_diff:  344.093859822


     61%|██████    | 763/1261 [16:36<10:40,  1.29s/it]

    417.410496295 m 745.98822604 m
    curve_diff:  328.577729745


     61%|██████    | 764/1261 [16:37<10:36,  1.28s/it]

    394.958845724 m 621.977979639 m
    curve_diff:  227.019133916


     61%|██████    | 765/1261 [16:38<10:32,  1.28s/it]

    385.434504972 m 546.67925382 m
    curve_diff:  161.244748848


     61%|██████    | 766/1261 [16:39<10:29,  1.27s/it]

    375.692321578 m 573.959907165 m
    curve_diff:  198.267585587


     61%|██████    | 767/1261 [16:41<10:25,  1.27s/it]

    385.17786938 m 538.408244063 m
    curve_diff:  153.230374683


     61%|██████    | 768/1261 [16:42<10:23,  1.26s/it]

    398.655900199 m 525.651805738 m
    curve_diff:  126.995905539


     61%|██████    | 769/1261 [16:43<10:31,  1.28s/it]

    388.63013872 m 563.360647456 m
    curve_diff:  174.730508736


     61%|██████    | 770/1261 [16:45<10:34,  1.29s/it]

    380.004042739 m 521.912879804 m
    curve_diff:  141.908837065


     61%|██████    | 771/1261 [16:46<10:29,  1.29s/it]

    397.471117676 m 567.241807414 m
    curve_diff:  169.770689738


     61%|██████    | 772/1261 [16:47<10:25,  1.28s/it]

    423.805364011 m 600.009899988 m
    curve_diff:  176.204535977


     61%|██████▏   | 773/1261 [16:48<10:21,  1.27s/it]

    437.827854729 m 664.898517055 m
    curve_diff:  227.070662327


     61%|██████▏   | 774/1261 [16:50<10:16,  1.27s/it]

    458.768671965 m 588.372575988 m
    curve_diff:  129.603904022


     61%|██████▏   | 775/1261 [16:51<10:14,  1.26s/it]

    461.664475952 m 600.238339692 m
    curve_diff:  138.573863739


     62%|██████▏   | 776/1261 [16:52<10:12,  1.26s/it]

    431.617483061 m 455.837178703 m
    curve_diff:  24.2196956417


     62%|██████▏   | 777/1261 [16:53<10:11,  1.26s/it]

    436.492435526 m 416.183292915 m
    curve_diff:  20.3091426109


     62%|██████▏   | 778/1261 [16:55<10:16,  1.28s/it]

    441.331741928 m 469.133382836 m
    curve_diff:  27.8016409083


     62%|██████▏   | 779/1261 [16:56<10:13,  1.27s/it]

    474.546900436 m 493.038019468 m
    curve_diff:  18.4911190321


     62%|██████▏   | 780/1261 [16:57<10:11,  1.27s/it]

    500.444804232 m 529.702362833 m
    curve_diff:  29.2575586009


     62%|██████▏   | 781/1261 [16:58<10:09,  1.27s/it]

    512.486141484 m 515.855066486 m
    curve_diff:  3.36892500186


     62%|██████▏   | 782/1261 [17:00<10:04,  1.26s/it]

    573.536852835 m 514.37167846 m
    curve_diff:  59.1651743748


     62%|██████▏   | 783/1261 [17:01<10:05,  1.27s/it]

    611.864533895 m 506.875589081 m
    curve_diff:  104.988944814


     62%|██████▏   | 784/1261 [17:02<10:04,  1.27s/it]

    663.853521262 m 583.777685919 m
    curve_diff:  80.0758353434


     62%|██████▏   | 785/1261 [17:03<10:01,  1.26s/it]

    681.082519526 m 653.196367095 m
    curve_diff:  27.8861524312


     62%|██████▏   | 786/1261 [17:05<10:11,  1.29s/it]

    706.228925447 m 696.763885254 m
    curve_diff:  9.46504019354


     62%|██████▏   | 787/1261 [17:06<10:07,  1.28s/it]

    714.594312357 m 774.720330888 m
    curve_diff:  60.126018531


     62%|██████▏   | 788/1261 [17:07<10:04,  1.28s/it]

    701.956502021 m 760.613876749 m
    curve_diff:  58.6573747287


     63%|██████▎   | 789/1261 [17:09<10:00,  1.27s/it]

    696.800447262 m 695.994603627 m
    curve_diff:  0.8058436345


     63%|██████▎   | 790/1261 [17:10<09:56,  1.27s/it]

    728.64550865 m 661.053573645 m
    curve_diff:  67.5919350055


     63%|██████▎   | 791/1261 [17:11<09:56,  1.27s/it]

    740.388989246 m 655.45313172 m
    curve_diff:  84.9358575253


     63%|██████▎   | 792/1261 [17:12<09:54,  1.27s/it]

    818.073098024 m 816.275044256 m
    curve_diff:  1.79805376842


     63%|██████▎   | 793/1261 [17:14<09:53,  1.27s/it]

    798.886009634 m 950.948676566 m
    curve_diff:  152.062666933


     63%|██████▎   | 794/1261 [17:15<09:59,  1.28s/it]

    804.817519333 m 1336.99551507 m
    curve_diff:  532.177995739


     63%|██████▎   | 795/1261 [17:16<09:56,  1.28s/it]

    720.145344752 m 1084.01755352 m
    curve_diff:  363.872208771


     63%|██████▎   | 796/1261 [17:18<09:53,  1.28s/it]

    632.89802545 m 834.864779557 m
    curve_diff:  201.966754107


     63%|██████▎   | 797/1261 [17:19<09:48,  1.27s/it]

    653.375106906 m 874.838933147 m
    curve_diff:  221.463826241


     63%|██████▎   | 798/1261 [17:20<09:44,  1.26s/it]

    666.524855567 m 739.217483796 m
    curve_diff:  72.6926282293


     63%|██████▎   | 799/1261 [17:21<09:45,  1.27s/it]

    611.9329129 m 713.619848455 m
    curve_diff:  101.686935555


     63%|██████▎   | 800/1261 [17:23<09:42,  1.26s/it]

    628.775885419 m 926.621610603 m
    curve_diff:  297.845725185


     64%|██████▎   | 801/1261 [17:24<09:41,  1.26s/it]

    627.123407733 m 844.069371354 m
    curve_diff:  216.945963621


     64%|██████▎   | 802/1261 [17:25<09:41,  1.27s/it]

    654.971869297 m 677.221026369 m
    curve_diff:  22.2491570722


     64%|██████▎   | 803/1261 [17:26<09:49,  1.29s/it]

    597.070824363 m 917.663787304 m
    curve_diff:  320.592962941


     64%|██████▍   | 804/1261 [17:28<09:44,  1.28s/it]

    505.692352716 m 698.357905421 m
    curve_diff:  192.665552705


     64%|██████▍   | 805/1261 [17:29<09:40,  1.27s/it]

    483.862125627 m 642.241896884 m
    curve_diff:  158.379771257


     64%|██████▍   | 806/1261 [17:30<09:35,  1.26s/it]

    478.803620385 m 607.021709839 m
    curve_diff:  128.218089454


     64%|██████▍   | 807/1261 [17:31<09:34,  1.27s/it]

    471.464851141 m 600.969011677 m
    curve_diff:  129.504160535


     64%|██████▍   | 808/1261 [17:33<09:32,  1.26s/it]

    440.109586398 m 526.1749267 m
    curve_diff:  86.0653403013


     64%|██████▍   | 809/1261 [17:34<09:31,  1.26s/it]

    463.079325336 m 515.40738307 m
    curve_diff:  52.3280577345


     64%|██████▍   | 810/1261 [17:35<09:30,  1.27s/it]

    449.153012204 m 530.974221997 m
    curve_diff:  81.8212097936


     64%|██████▍   | 811/1261 [17:37<09:37,  1.28s/it]

    465.601127077 m 483.736223498 m
    curve_diff:  18.1350964208


     64%|██████▍   | 812/1261 [17:38<09:31,  1.27s/it]

    471.862359263 m 477.248362686 m
    curve_diff:  5.38600342276


     64%|██████▍   | 813/1261 [17:39<09:26,  1.26s/it]

    502.543429916 m 427.146301287 m
    curve_diff:  75.3971286286


     65%|██████▍   | 814/1261 [17:40<09:24,  1.26s/it]

    535.214478168 m 448.222889679 m
    curve_diff:  86.9915884895


     65%|██████▍   | 815/1261 [17:42<09:23,  1.26s/it]

    538.505163731 m 407.000156379 m
    curve_diff:  131.505007352


     65%|██████▍   | 816/1261 [17:43<09:21,  1.26s/it]

    521.878489158 m 382.917216619 m
    curve_diff:  138.961272539


     65%|██████▍   | 817/1261 [17:44<09:19,  1.26s/it]

    480.085638616 m 355.260801162 m
    curve_diff:  124.824837454


     65%|██████▍   | 818/1261 [17:45<09:20,  1.27s/it]

    469.602728238 m 381.024249731 m
    curve_diff:  88.5784785066


     65%|██████▍   | 819/1261 [17:47<09:26,  1.28s/it]

    544.528679887 m 397.158850063 m
    curve_diff:  147.369829824


     65%|██████▌   | 820/1261 [17:48<09:22,  1.28s/it]

    562.413711729 m 464.156486507 m
    curve_diff:  98.2572252213


     65%|██████▌   | 821/1261 [17:49<09:19,  1.27s/it]

    610.649175342 m 418.406006513 m
    curve_diff:  192.243168829


     65%|██████▌   | 822/1261 [17:51<09:15,  1.26s/it]

    597.462962265 m 481.480236598 m
    curve_diff:  115.982725668


     65%|██████▌   | 823/1261 [17:52<09:14,  1.27s/it]

    621.615321451 m 489.50026273 m
    curve_diff:  132.115058721


     65%|██████▌   | 824/1261 [17:53<09:13,  1.27s/it]

    606.602295093 m 514.293738332 m
    curve_diff:  92.3085567612


     65%|██████▌   | 825/1261 [17:54<09:12,  1.27s/it]

    528.517169933 m 439.407692199 m
    curve_diff:  89.1094777333


     66%|██████▌   | 826/1261 [17:56<09:09,  1.26s/it]

    483.477550801 m 411.355756769 m
    curve_diff:  72.1217940316


     66%|██████▌   | 827/1261 [17:57<09:16,  1.28s/it]

    500.032487471 m 405.579251621 m
    curve_diff:  94.4532358491


     66%|██████▌   | 828/1261 [17:58<09:20,  1.29s/it]

    462.184795966 m 375.945994575 m
    curve_diff:  86.2388013909


     66%|██████▌   | 829/1261 [17:59<09:15,  1.29s/it]

    450.482586704 m 394.000899431 m
    curve_diff:  56.481687273


     66%|██████▌   | 830/1261 [18:01<09:12,  1.28s/it]

    453.433613539 m 396.484124175 m
    curve_diff:  56.9494893632


     66%|██████▌   | 831/1261 [18:02<09:07,  1.27s/it]

    439.971836044 m 415.0503894 m
    curve_diff:  24.9214466438


     66%|██████▌   | 832/1261 [18:03<09:06,  1.27s/it]

    444.507278621 m 431.516191326 m
    curve_diff:  12.9910872949


     66%|██████▌   | 833/1261 [18:05<09:05,  1.28s/it]

    446.432116279 m 451.591526155 m
    curve_diff:  5.15940987512


     66%|██████▌   | 834/1261 [18:06<09:01,  1.27s/it]

    431.012019704 m 465.095951088 m
    curve_diff:  34.083931384


     66%|██████▌   | 835/1261 [18:07<08:57,  1.26s/it]

    422.775634829 m 550.467559766 m
    curve_diff:  127.691924936


     66%|██████▋   | 836/1261 [18:08<09:05,  1.28s/it]

    428.001216004 m 620.914879067 m
    curve_diff:  192.913663063


     66%|██████▋   | 837/1261 [18:10<09:03,  1.28s/it]

    439.643566234 m 658.613840864 m
    curve_diff:  218.970274629


     66%|██████▋   | 838/1261 [18:11<08:59,  1.28s/it]

    404.711975754 m 548.427325181 m
    curve_diff:  143.715349426


     67%|██████▋   | 839/1261 [18:12<08:55,  1.27s/it]

    405.370745885 m 489.886470465 m
    curve_diff:  84.5157245794


     67%|██████▋   | 840/1261 [18:13<08:53,  1.27s/it]

    420.503933765 m 465.086964749 m
    curve_diff:  44.5830309833


     67%|██████▋   | 841/1261 [18:15<08:51,  1.27s/it]

    406.127739474 m 486.579082406 m
    curve_diff:  80.4513429316


     67%|██████▋   | 842/1261 [18:16<08:52,  1.27s/it]

    420.453441143 m 486.415103789 m
    curve_diff:  65.9616626462


     67%|██████▋   | 843/1261 [18:17<08:51,  1.27s/it]

    472.149591777 m 507.267702377 m
    curve_diff:  35.1181106002


     67%|██████▋   | 844/1261 [18:19<08:58,  1.29s/it]

    504.999116444 m 539.097520174 m
    curve_diff:  34.0984037306


     67%|██████▋   | 845/1261 [18:20<08:53,  1.28s/it]

    517.543745614 m 611.124297463 m
    curve_diff:  93.5805518492


     67%|██████▋   | 846/1261 [18:21<08:50,  1.28s/it]

    521.614541969 m 565.182820944 m
    curve_diff:  43.5682789748


     67%|██████▋   | 847/1261 [18:22<08:47,  1.27s/it]

    557.138161431 m 850.878965119 m
    curve_diff:  293.740803688


     67%|██████▋   | 848/1261 [18:24<08:58,  1.30s/it]

    590.93219734 m 883.703688045 m
    curve_diff:  292.771490705


     67%|██████▋   | 849/1261 [18:25<09:07,  1.33s/it]

    651.273605816 m 855.084315168 m
    curve_diff:  203.810709351


     67%|██████▋   | 850/1261 [18:26<09:04,  1.32s/it]

    684.118462891 m 839.156395225 m
    curve_diff:  155.037932334


     67%|██████▋   | 851/1261 [18:28<08:58,  1.31s/it]

    667.634066284 m 694.462783684 m
    curve_diff:  26.8287174005


     68%|██████▊   | 852/1261 [18:29<09:02,  1.33s/it]

    729.762516293 m 639.267216549 m
    curve_diff:  90.4952997438


     68%|██████▊   | 853/1261 [18:30<08:56,  1.31s/it]

    714.028539657 m 548.745160076 m
    curve_diff:  165.283379581


     68%|██████▊   | 854/1261 [18:32<08:48,  1.30s/it]

    728.111447529 m 538.567875647 m
    curve_diff:  189.543571881


     68%|██████▊   | 855/1261 [18:33<08:45,  1.29s/it]

    759.819486213 m 573.227370609 m
    curve_diff:  186.592115603


     68%|██████▊   | 856/1261 [18:34<08:44,  1.30s/it]

    855.327474983 m 564.055532505 m
    curve_diff:  291.271942478


     68%|██████▊   | 857/1261 [18:36<08:41,  1.29s/it]

    825.710910699 m 573.997744923 m
    curve_diff:  251.713165776


     68%|██████▊   | 858/1261 [18:37<08:40,  1.29s/it]

    787.345620498 m 492.00213832 m
    curve_diff:  295.343482179


     68%|██████▊   | 859/1261 [18:38<08:42,  1.30s/it]

    776.954582565 m 590.703687365 m
    curve_diff:  186.2508952


     68%|██████▊   | 860/1261 [18:40<08:49,  1.32s/it]

    797.902026847 m 672.364390335 m
    curve_diff:  125.537636512


     68%|██████▊   | 861/1261 [18:41<08:48,  1.32s/it]

    736.576176351 m 568.229404825 m
    curve_diff:  168.346771526


     68%|██████▊   | 862/1261 [18:42<08:41,  1.31s/it]

    746.600534042 m 541.15905481 m
    curve_diff:  205.441479233


     68%|██████▊   | 863/1261 [18:43<08:37,  1.30s/it]

    762.46609216 m 512.109557456 m
    curve_diff:  250.356534704


     69%|██████▊   | 864/1261 [18:45<08:33,  1.29s/it]

    756.348423331 m 531.307913379 m
    curve_diff:  225.040509952


     69%|██████▊   | 865/1261 [18:46<08:30,  1.29s/it]

    732.99720384 m 531.238168044 m
    curve_diff:  201.759035796


     69%|██████▊   | 866/1261 [18:47<08:29,  1.29s/it]

    742.647834205 m 453.500030577 m
    curve_diff:  289.147803628


     69%|██████▉   | 867/1261 [18:49<08:26,  1.29s/it]

    801.653941011 m 505.289730652 m
    curve_diff:  296.364210359


     69%|██████▉   | 868/1261 [18:50<08:32,  1.30s/it]

    742.289845417 m 518.854866148 m
    curve_diff:  223.434979269


     69%|██████▉   | 869/1261 [18:51<08:28,  1.30s/it]

    661.355016755 m 618.958101768 m
    curve_diff:  42.3969149863


     69%|██████▉   | 870/1261 [18:52<08:26,  1.29s/it]

    678.171467492 m 662.510484789 m
    curve_diff:  15.6609827029


     69%|██████▉   | 871/1261 [18:54<08:21,  1.29s/it]

    613.053046444 m 506.678259186 m
    curve_diff:  106.374787257


     69%|██████▉   | 872/1261 [18:55<08:19,  1.28s/it]

    632.864483096 m 563.220842059 m
    curve_diff:  69.6436410369


     69%|██████▉   | 873/1261 [18:56<08:22,  1.30s/it]

    616.296172008 m 1039.6333579 m
    curve_diff:  423.337185888


     69%|██████▉   | 874/1261 [18:58<08:17,  1.29s/it]

    633.584588347 m 1176.50346359 m
    curve_diff:  542.918875244


     69%|██████▉   | 875/1261 [18:59<08:13,  1.28s/it]

    612.55316473 m 1459.16861607 m
    curve_diff:  846.615451338


     69%|██████▉   | 876/1261 [19:00<08:22,  1.31s/it]

    594.528964861 m 865.802166634 m
    curve_diff:  271.273201773


     70%|██████▉   | 877/1261 [19:01<08:19,  1.30s/it]

    600.357021701 m 900.555872536 m
    curve_diff:  300.198850835


     70%|██████▉   | 878/1261 [19:03<08:15,  1.29s/it]

    627.014372534 m 773.091421163 m
    curve_diff:  146.077048629


     70%|██████▉   | 879/1261 [19:04<08:13,  1.29s/it]

    562.421099177 m 623.232809442 m
    curve_diff:  60.8117102654


     70%|██████▉   | 880/1261 [19:05<08:12,  1.29s/it]

    623.342763681 m 641.88517368 m
    curve_diff:  18.5424099991


     70%|██████▉   | 881/1261 [19:07<08:11,  1.29s/it]

    663.851384656 m 763.495240426 m
    curve_diff:  99.6438557699


     70%|██████▉   | 882/1261 [19:08<08:08,  1.29s/it]

    695.832456864 m 941.773969911 m
    curve_diff:  245.941513047


     70%|███████   | 883/1261 [19:09<08:08,  1.29s/it]

    727.502661193 m 739.502495182 m
    curve_diff:  11.9998339886


     70%|███████   | 884/1261 [19:11<08:17,  1.32s/it]

    742.778832109 m 928.122806697 m
    curve_diff:  185.343974587


     70%|███████   | 885/1261 [19:12<08:12,  1.31s/it]

    874.93664527 m 1314.15800734 m
    curve_diff:  439.221362072


     70%|███████   | 886/1261 [19:13<08:08,  1.30s/it]

    1029.07662327 m 846.299740717 m
    curve_diff:  182.776882557


     70%|███████   | 887/1261 [19:14<08:05,  1.30s/it]

    1086.53257556 m 870.346830664 m
    curve_diff:  216.185744893


     70%|███████   | 888/1261 [19:16<08:04,  1.30s/it]

    1076.08025789 m 639.848627836 m
    curve_diff:  436.231630057


     70%|███████   | 889/1261 [19:17<08:04,  1.30s/it]

    1158.78347261 m 549.506378249 m
    curve_diff:  609.277094358


     71%|███████   | 890/1261 [19:18<08:03,  1.30s/it]

    1232.27301427 m 670.062974526 m
    curve_diff:  562.210039747


     71%|███████   | 891/1261 [19:20<08:00,  1.30s/it]

    1365.61151508 m 683.063297067 m
    curve_diff:  682.548218009


     71%|███████   | 892/1261 [19:21<08:03,  1.31s/it]

    1237.18276233 m 861.264431397 m
    curve_diff:  375.91833093


     71%|███████   | 893/1261 [19:22<08:03,  1.31s/it]

    1188.00106412 m 865.485791608 m
    curve_diff:  322.515272516


     71%|███████   | 894/1261 [19:24<07:57,  1.30s/it]

    980.847522889 m 842.540008849 m
    curve_diff:  138.30751404


     71%|███████   | 895/1261 [19:25<07:54,  1.30s/it]

    1031.46737667 m 871.828617474 m
    curve_diff:  159.638759198


     71%|███████   | 896/1261 [19:26<07:51,  1.29s/it]

    1054.1057446 m 718.569403076 m
    curve_diff:  335.536341529


     71%|███████   | 897/1261 [19:27<07:51,  1.30s/it]

    1069.37746842 m 688.543564713 m
    curve_diff:  380.833903702


     71%|███████   | 898/1261 [19:29<07:49,  1.29s/it]

    1000.41414233 m 655.893005441 m
    curve_diff:  344.521136894


     71%|███████▏  | 899/1261 [19:30<07:48,  1.29s/it]

    1242.01238623 m 590.413672687 m
    curve_diff:  651.598713548


     71%|███████▏  | 900/1261 [19:31<07:49,  1.30s/it]

    991.323359851 m 503.406554936 m
    curve_diff:  487.916804914


     71%|███████▏  | 901/1261 [19:33<07:55,  1.32s/it]

    875.842629348 m 448.390713008 m
    curve_diff:  427.451916341


     72%|███████▏  | 902/1261 [19:34<07:49,  1.31s/it]

    928.606454166 m 449.380159887 m
    curve_diff:  479.226294279


     72%|███████▏  | 903/1261 [19:35<07:45,  1.30s/it]

    892.809994148 m 494.590344973 m
    curve_diff:  398.219649175


     72%|███████▏  | 904/1261 [19:37<07:45,  1.30s/it]

    849.216883991 m 491.913478176 m
    curve_diff:  357.303405815


     72%|███████▏  | 905/1261 [19:38<07:43,  1.30s/it]

    755.20127001 m 460.32261024 m
    curve_diff:  294.87865977


     72%|███████▏  | 906/1261 [19:39<07:40,  1.30s/it]

    671.744402054 m 444.621011124 m
    curve_diff:  227.12339093


     72%|███████▏  | 907/1261 [19:40<07:37,  1.29s/it]

    647.960271929 m 508.331028598 m
    curve_diff:  139.62924333


     72%|███████▏  | 908/1261 [19:42<07:36,  1.29s/it]

    602.028545957 m 673.906541689 m
    curve_diff:  71.8779957328


     72%|███████▏  | 909/1261 [19:43<07:43,  1.32s/it]

    597.026521672 m 575.678788369 m
    curve_diff:  21.3477333021


     72%|███████▏  | 910/1261 [19:44<07:39,  1.31s/it]

    504.064036801 m 451.175818021 m
    curve_diff:  52.8882187805


     72%|███████▏  | 911/1261 [19:46<07:35,  1.30s/it]

    500.884192194 m 450.543804785 m
    curve_diff:  50.3403874093


     72%|███████▏  | 912/1261 [19:47<07:33,  1.30s/it]

    465.80524559 m 447.381516294 m
    curve_diff:  18.4237292958


     72%|███████▏  | 913/1261 [19:48<07:32,  1.30s/it]

    458.946348087 m 506.322825723 m
    curve_diff:  47.3764776354


     72%|███████▏  | 914/1261 [19:50<07:28,  1.29s/it]

    461.070785663 m 494.031880122 m
    curve_diff:  32.961094459


     73%|███████▎  | 915/1261 [19:51<07:28,  1.30s/it]

    487.944735369 m 565.715754257 m
    curve_diff:  77.7710188886


     73%|███████▎  | 916/1261 [19:52<07:26,  1.30s/it]

    442.968839023 m 527.16240316 m
    curve_diff:  84.1935641366


     73%|███████▎  | 917/1261 [19:54<07:34,  1.32s/it]

    443.985472422 m 641.79021685 m
    curve_diff:  197.804744428


     73%|███████▎  | 918/1261 [19:55<07:29,  1.31s/it]

    464.051636321 m 954.660927805 m
    curve_diff:  490.609291485


     73%|███████▎  | 919/1261 [19:56<07:25,  1.30s/it]

    485.055149169 m 1197.49953645 m
    curve_diff:  712.444387279


     73%|███████▎  | 920/1261 [19:57<07:23,  1.30s/it]

    507.602479151 m 948.370679738 m
    curve_diff:  440.768200587


     73%|███████▎  | 921/1261 [19:59<07:21,  1.30s/it]

    541.517283183 m 740.349187657 m
    curve_diff:  198.831904474


     73%|███████▎  | 922/1261 [20:00<07:21,  1.30s/it]

    568.200764723 m 625.141797707 m
    curve_diff:  56.9410329846


     73%|███████▎  | 923/1261 [20:01<07:18,  1.30s/it]

    603.37503241 m 570.998807097 m
    curve_diff:  32.376225313


     73%|███████▎  | 924/1261 [20:03<07:15,  1.29s/it]

    602.849094529 m 589.997741728 m
    curve_diff:  12.8513528005


     73%|███████▎  | 925/1261 [20:04<07:22,  1.32s/it]

    650.142911751 m 552.875064075 m
    curve_diff:  97.2678476765


     73%|███████▎  | 926/1261 [20:05<07:18,  1.31s/it]

    702.921886441 m 513.440801239 m
    curve_diff:  189.481085202


     74%|███████▎  | 927/1261 [20:07<07:14,  1.30s/it]

    710.881336011 m 538.533941122 m
    curve_diff:  172.347394889


     74%|███████▎  | 928/1261 [20:08<07:13,  1.30s/it]

    709.309365786 m 567.0157302 m
    curve_diff:  142.293635586


     74%|███████▎  | 929/1261 [20:09<07:12,  1.30s/it]

    760.052795964 m 525.824177492 m
    curve_diff:  234.228618472


     74%|███████▍  | 930/1261 [20:10<07:09,  1.30s/it]

    763.163617435 m 534.285478394 m
    curve_diff:  228.878139041


     74%|███████▍  | 931/1261 [20:12<07:06,  1.29s/it]

    717.96770545 m 476.84957325 m
    curve_diff:  241.1181322


     74%|███████▍  | 932/1261 [20:13<07:03,  1.29s/it]

    651.661015726 m 427.062164464 m
    curve_diff:  224.598851263


     74%|███████▍  | 933/1261 [20:14<07:10,  1.31s/it]

    599.234836995 m 422.15839797 m
    curve_diff:  177.076439024


     74%|███████▍  | 934/1261 [20:16<07:07,  1.31s/it]

    589.113463079 m 413.884361407 m
    curve_diff:  175.229101672


     74%|███████▍  | 935/1261 [20:17<07:04,  1.30s/it]

    555.474654435 m 367.440075117 m
    curve_diff:  188.034579317


     74%|███████▍  | 936/1261 [20:18<07:01,  1.30s/it]

    542.788487003 m 406.190818436 m
    curve_diff:  136.597668567


     74%|███████▍  | 937/1261 [20:20<06:58,  1.29s/it]

    556.441131083 m 404.271459572 m
    curve_diff:  152.169671511


     74%|███████▍  | 938/1261 [20:21<06:56,  1.29s/it]

    514.338720547 m 369.019424045 m
    curve_diff:  145.319296501


     74%|███████▍  | 939/1261 [20:22<06:54,  1.29s/it]

    520.160518231 m 406.699309933 m
    curve_diff:  113.461208298


     75%|███████▍  | 940/1261 [20:23<07:01,  1.31s/it]

    490.915829392 m 456.809430909 m
    curve_diff:  34.1063984831


     75%|███████▍  | 941/1261 [20:25<07:52,  1.48s/it]

    455.901578121 m 465.891086237 m
    curve_diff:  9.98950811653


     75%|███████▍  | 942/1261 [20:27<07:38,  1.44s/it]

    471.26104887 m 693.098362198 m
    curve_diff:  221.837313328


     75%|███████▍  | 943/1261 [20:28<07:22,  1.39s/it]

    456.020543101 m 534.246665224 m
    curve_diff:  78.2261221229


     75%|███████▍  | 944/1261 [20:29<07:11,  1.36s/it]

    515.663257659 m 677.00772533 m
    curve_diff:  161.344467671


     75%|███████▍  | 945/1261 [20:31<07:05,  1.35s/it]

    490.585618391 m 496.332497983 m
    curve_diff:  5.74687959164


     75%|███████▌  | 946/1261 [20:32<07:00,  1.33s/it]

    523.607023187 m 451.943957421 m
    curve_diff:  71.6630657658


     75%|███████▌  | 947/1261 [20:33<06:55,  1.32s/it]

    555.660764348 m 527.428351649 m
    curve_diff:  28.2324126998


     75%|███████▌  | 948/1261 [20:35<07:15,  1.39s/it]

    591.720005773 m 596.822350661 m
    curve_diff:  5.10234488822


     75%|███████▌  | 949/1261 [20:36<07:27,  1.43s/it]

    583.342805645 m 635.978538735 m
    curve_diff:  52.6357330896


     75%|███████▌  | 950/1261 [20:37<07:09,  1.38s/it]

    624.424521495 m 571.988113984 m
    curve_diff:  52.4364075111


     75%|███████▌  | 951/1261 [20:39<06:56,  1.34s/it]

    592.846996346 m 589.883262898 m
    curve_diff:  2.96373344805


     75%|███████▌  | 952/1261 [20:40<06:46,  1.32s/it]

    619.823428715 m 683.444789586 m
    curve_diff:  63.6213608715


     76%|███████▌  | 953/1261 [20:41<06:41,  1.30s/it]

    579.084780626 m 845.644026791 m
    curve_diff:  266.559246165


     76%|███████▌  | 954/1261 [20:43<06:35,  1.29s/it]

    592.079109222 m 1006.32595454 m
    curve_diff:  414.246845315


     76%|███████▌  | 955/1261 [20:44<06:31,  1.28s/it]

    623.411892341 m 1382.20007248 m
    curve_diff:  758.788180137


     76%|███████▌  | 956/1261 [20:45<06:29,  1.28s/it]

    696.553316685 m 887.127356174 m
    curve_diff:  190.574039489


     76%|███████▌  | 957/1261 [20:46<06:32,  1.29s/it]

    677.186583112 m 818.827664357 m
    curve_diff:  141.641081246


     76%|███████▌  | 958/1261 [20:48<06:25,  1.27s/it]

    621.925877279 m 642.321535166 m
    curve_diff:  20.3956578868


     76%|███████▌  | 959/1261 [20:49<06:42,  1.33s/it]

    665.326232126 m 641.578746219 m
    curve_diff:  23.7474859061


     76%|███████▌  | 960/1261 [20:52<09:01,  1.80s/it]

    719.833524097 m 691.703842662 m
    curve_diff:  28.1296814351


     76%|███████▌  | 961/1261 [20:54<09:28,  1.90s/it]

    759.620800893 m 720.611994644 m
    curve_diff:  39.0088062491


     76%|███████▋  | 962/1261 [20:57<10:49,  2.17s/it]

    922.658014861 m 761.609335066 m
    curve_diff:  161.048679794


     76%|███████▋  | 963/1261 [20:58<09:29,  1.91s/it]

    624.62003615 m 727.579282328 m
    curve_diff:  102.959246178


     76%|███████▋  | 964/1261 [21:00<09:01,  1.82s/it]

    600.553031549 m 702.072703627 m
    curve_diff:  101.519672078


     77%|███████▋  | 965/1261 [21:01<08:27,  1.71s/it]

    868.385412432 m 1144.44488506 m
    curve_diff:  276.059472624


     77%|███████▋  | 966/1261 [21:03<08:03,  1.64s/it]

    1484.33252667 m 820.55673353 m
    curve_diff:  663.775793145


     77%|███████▋  | 967/1261 [21:04<07:44,  1.58s/it]

    851.686289098 m 800.263159557 m
    curve_diff:  51.4231295418


     77%|███████▋  | 968/1261 [21:06<07:26,  1.52s/it]

    1678.02494048 m 6064.2022877 m
    curve_diff:  4386.17734722


     77%|███████▋  | 969/1261 [21:07<07:13,  1.49s/it]

    1985.07083258 m 2803.00455147 m
    curve_diff:  817.933718891


     77%|███████▋  | 970/1261 [21:08<06:59,  1.44s/it]

    545.043625677 m 26649.5796831 m
    curve_diff:  26104.5360574


     77%|███████▋  | 971/1261 [21:10<06:47,  1.41s/it]

    607.620834535 m 1009.97970399 m
    curve_diff:  402.358869457


     77%|███████▋  | 972/1261 [21:11<06:39,  1.38s/it]

    1370.49535019 m 3289.11578718 m
    curve_diff:  1918.62043698


     77%|███████▋  | 973/1261 [21:12<06:40,  1.39s/it]

    876.706088754 m 2230.79736918 m
    curve_diff:  1354.09128043


     77%|███████▋  | 974/1261 [21:14<06:39,  1.39s/it]

    592.239489428 m 1353.68861604 m
    curve_diff:  761.449126617


     77%|███████▋  | 975/1261 [21:15<06:35,  1.38s/it]

    4540.57488063 m 931.263805798 m
    curve_diff:  3609.31107483


     77%|███████▋  | 976/1261 [21:17<06:34,  1.39s/it]

    564.165912838 m 529.673910745 m
    curve_diff:  34.4920020927


     77%|███████▋  | 977/1261 [21:18<07:02,  1.49s/it]

    816.732091873 m 534.805229141 m
    curve_diff:  281.926862732


     78%|███████▊  | 978/1261 [21:20<06:47,  1.44s/it]

    5123.58487273 m 500.764219073 m
    curve_diff:  4622.82065366


     78%|███████▊  | 979/1261 [21:21<06:38,  1.41s/it]

    1799.73172967 m 1124.07113565 m
    curve_diff:  675.660594025


     78%|███████▊  | 980/1261 [21:22<06:39,  1.42s/it]

    284.721061064 m 398.174888986 m
    curve_diff:  113.453827923


     78%|███████▊  | 981/1261 [21:24<06:35,  1.41s/it]

    1300.72361751 m 293.266714427 m
    curve_diff:  1007.45690308


     78%|███████▊  | 982/1261 [21:25<06:27,  1.39s/it]

    404.345896407 m 331.716056943 m
    curve_diff:  72.6298394638


     78%|███████▊  | 983/1261 [21:27<06:29,  1.40s/it]

    11475.0167083 m 314.584883554 m
    curve_diff:  11160.4318247


     78%|███████▊  | 984/1261 [21:28<06:23,  1.38s/it]

    1716.5330237 m 397.523793383 m
    curve_diff:  1319.00923031


     78%|███████▊  | 985/1261 [21:29<06:23,  1.39s/it]

    1060.08132642 m 286.882480677 m
    curve_diff:  773.198845744


     78%|███████▊  | 986/1261 [21:31<06:16,  1.37s/it]

    700.628770614 m 290.634163404 m
    curve_diff:  409.99460721


     78%|███████▊  | 987/1261 [21:32<06:13,  1.36s/it]

    2673.94101243 m 92.1312089194 m
    curve_diff:  2581.80980351


     78%|███████▊  | 988/1261 [21:34<06:39,  1.46s/it]

    2144.21011434 m 222.6655986 m
    curve_diff:  1921.54451574


     78%|███████▊  | 989/1261 [21:35<06:30,  1.43s/it]

    1307.05159753 m 204.263250089 m
    curve_diff:  1102.78834744


     79%|███████▊  | 990/1261 [21:36<06:24,  1.42s/it]

    543.600651568 m 284.79902753 m
    curve_diff:  258.801624038


     79%|███████▊  | 991/1261 [21:38<06:39,  1.48s/it]

    1952.14997682 m 329.160487243 m
    curve_diff:  1622.98948958


     79%|███████▊  | 992/1261 [21:40<06:54,  1.54s/it]

    4320.95389836 m 509.6553326 m
    curve_diff:  3811.29856576


     79%|███████▊  | 993/1261 [21:41<06:46,  1.52s/it]

    1927.76513048 m 566.278546901 m
    curve_diff:  1361.48658358


     79%|███████▉  | 994/1261 [21:43<06:54,  1.55s/it]

    20497.6096307 m 468.410926744 m
    curve_diff:  20029.198704


     79%|███████▉  | 995/1261 [21:44<07:00,  1.58s/it]

    511.462365323 m 398.367183626 m
    curve_diff:  113.095181697


     79%|███████▉  | 996/1261 [21:46<06:42,  1.52s/it]

    328.95640626 m 395.393807817 m
    curve_diff:  66.4374015576


     79%|███████▉  | 997/1261 [21:47<06:24,  1.45s/it]

    172.934953997 m 442.646581166 m
    curve_diff:  269.711627169


     79%|███████▉  | 998/1261 [21:49<06:16,  1.43s/it]

    261.094067957 m 447.177680127 m
    curve_diff:  186.08361217


     79%|███████▉  | 999/1261 [21:50<06:11,  1.42s/it]

    269.609786389 m 441.829290765 m
    curve_diff:  172.219504376


     79%|███████▉  | 1000/1261 [21:51<06:07,  1.41s/it]

    285.692044438 m 361.207782144 m
    curve_diff:  75.5157377056


     79%|███████▉  | 1001/1261 [21:53<06:05,  1.41s/it]

    539.414562518 m 270.696501497 m
    curve_diff:  268.718061021


     79%|███████▉  | 1002/1261 [21:54<06:01,  1.40s/it]

    522.958938371 m 273.174652848 m
    curve_diff:  249.784285523


     80%|███████▉  | 1003/1261 [21:55<06:00,  1.40s/it]

    1293.39375447 m 318.742805605 m
    curve_diff:  974.650948863


     80%|███████▉  | 1004/1261 [21:57<05:57,  1.39s/it]

    1144.01224399 m 269.885322091 m
    curve_diff:  874.126921894


     80%|███████▉  | 1005/1261 [21:58<05:56,  1.39s/it]

    388.75739961 m 228.65236404 m
    curve_diff:  160.105035571


     80%|███████▉  | 1006/1261 [22:00<05:55,  1.40s/it]

    878.920662279 m 392.189376151 m
    curve_diff:  486.731286129


     80%|███████▉  | 1007/1261 [22:01<06:01,  1.42s/it]

    1319.79924981 m 456.369233606 m
    curve_diff:  863.430016206


     80%|███████▉  | 1008/1261 [22:02<05:56,  1.41s/it]

    319.699658946 m 322.577377011 m
    curve_diff:  2.87771806578


     80%|████████  | 1009/1261 [22:04<05:55,  1.41s/it]

    1353.25452169 m 349.329248234 m
    curve_diff:  1003.92527346


     80%|████████  | 1010/1261 [22:05<05:51,  1.40s/it]

    290.706044958 m 375.149512778 m
    curve_diff:  84.4434678201


     80%|████████  | 1011/1261 [22:07<05:49,  1.40s/it]

    368.981372414 m 361.892327661 m
    curve_diff:  7.08904475357


     80%|████████  | 1012/1261 [22:08<05:46,  1.39s/it]

    389.399641521 m 410.813035698 m
    curve_diff:  21.413394177


     80%|████████  | 1013/1261 [22:09<05:44,  1.39s/it]

    582.515691923 m 367.940381656 m
    curve_diff:  214.575310266


     80%|████████  | 1014/1261 [22:11<05:50,  1.42s/it]

    1424.30627084 m 367.905095329 m
    curve_diff:  1056.40117551


     80%|████████  | 1015/1261 [22:12<05:47,  1.41s/it]

    391.906682808 m 567.388094905 m
    curve_diff:  175.481412097


     81%|████████  | 1016/1261 [22:14<05:45,  1.41s/it]

    258.429414543 m 335.47463123 m
    curve_diff:  77.0452166876


     81%|████████  | 1017/1261 [22:15<05:42,  1.40s/it]

    158.973657881 m 863.358174379 m
    curve_diff:  704.384516498


     81%|████████  | 1018/1261 [22:17<05:44,  1.42s/it]

    237.295976376 m 740.425435354 m
    curve_diff:  503.129458977


     81%|████████  | 1019/1261 [22:18<05:42,  1.41s/it]

    137.140156206 m 851.338434655 m
    curve_diff:  714.19827845


     81%|████████  | 1020/1261 [22:19<05:44,  1.43s/it]

    300.905957165 m 1091.60507816 m
    curve_diff:  790.699120992


     81%|████████  | 1021/1261 [22:21<05:46,  1.44s/it]

    779.569376007 m 389.036192124 m
    curve_diff:  390.533183883


     81%|████████  | 1022/1261 [22:22<05:47,  1.45s/it]

    246.273287519 m 495.968586869 m
    curve_diff:  249.69529935


     81%|████████  | 1023/1261 [22:24<05:56,  1.50s/it]

    2573.87699857 m 554.558144498 m
    curve_diff:  2019.31885408


     81%|████████  | 1024/1261 [22:25<05:53,  1.49s/it]

    1720.46826506 m 580.654586687 m
    curve_diff:  1139.81367838


     81%|████████▏ | 1025/1261 [22:27<05:49,  1.48s/it]

    624.1291719 m 378.266842476 m
    curve_diff:  245.862329424


     81%|████████▏ | 1026/1261 [22:28<05:42,  1.46s/it]

    342.075471955 m 486.737324091 m
    curve_diff:  144.661852136


     81%|████████▏ | 1027/1261 [22:30<05:36,  1.44s/it]

    341.170756538 m 390.561050311 m
    curve_diff:  49.3902937725


     82%|████████▏ | 1028/1261 [22:31<05:35,  1.44s/it]

    615.536185123 m 388.346363176 m
    curve_diff:  227.189821946


     82%|████████▏ | 1029/1261 [22:33<05:42,  1.48s/it]

    436.243180737 m 240.874479044 m
    curve_diff:  195.368701693


     82%|████████▏ | 1030/1261 [22:34<05:37,  1.46s/it]

    531.691109826 m 236.886182311 m
    curve_diff:  294.804927515


     82%|████████▏ | 1031/1261 [22:36<05:36,  1.46s/it]

    436.886297842 m 239.801730675 m
    curve_diff:  197.084567167


     82%|████████▏ | 1032/1261 [22:37<05:35,  1.46s/it]

    466.382696806 m 227.97295259 m
    curve_diff:  238.409744216


     82%|████████▏ | 1033/1261 [22:39<05:34,  1.47s/it]

    304.709731477 m 223.688106761 m
    curve_diff:  81.0216247164


     82%|████████▏ | 1034/1261 [22:40<05:31,  1.46s/it]

    309.811090713 m 253.163359055 m
    curve_diff:  56.6477316572


     82%|████████▏ | 1035/1261 [22:41<05:27,  1.45s/it]

    233.092968291 m 225.050705914 m
    curve_diff:  8.04226237665


     82%|████████▏ | 1036/1261 [22:43<05:30,  1.47s/it]

    252.717621923 m 200.253719978 m
    curve_diff:  52.4639019451


     82%|████████▏ | 1037/1261 [22:44<05:26,  1.46s/it]

    298.149565215 m 193.459857802 m
    curve_diff:  104.689707414


     82%|████████▏ | 1038/1261 [22:46<05:24,  1.46s/it]

    280.949619084 m 79.6401434847 m
    curve_diff:  201.309475599


     82%|████████▏ | 1039/1261 [22:47<05:24,  1.46s/it]

    324.189692643 m 79.1269977224 m
    curve_diff:  245.062694921


     82%|████████▏ | 1040/1261 [22:49<05:21,  1.45s/it]

    276.123051528 m 256.383047932 m
    curve_diff:  19.7400035955


     83%|████████▎ | 1041/1261 [22:50<05:18,  1.45s/it]

    480.454109085 m 226.896529104 m
    curve_diff:  253.557579981


     83%|████████▎ | 1042/1261 [22:52<05:16,  1.44s/it]

    459.523777676 m 95.544382737 m
    curve_diff:  363.979394939


     83%|████████▎ | 1043/1261 [22:53<05:17,  1.45s/it]

    400.342810988 m 107.218830212 m
    curve_diff:  293.123980776


     83%|████████▎ | 1044/1261 [22:55<05:14,  1.45s/it]

    427.217630725 m 461.107027604 m
    curve_diff:  33.8893968789


     83%|████████▎ | 1045/1261 [22:56<05:11,  1.44s/it]

    342.687984489 m 87.1008001064 m
    curve_diff:  255.587184382


     83%|████████▎ | 1046/1261 [22:57<05:12,  1.46s/it]

    326.412873936 m 313.243052901 m
    curve_diff:  13.1698210351


     83%|████████▎ | 1047/1261 [22:59<05:08,  1.44s/it]

    334.777676522 m 366.274825977 m
    curve_diff:  31.4971494545


     83%|████████▎ | 1048/1261 [23:00<05:07,  1.44s/it]

    409.82432378 m 416.478289982 m
    curve_diff:  6.65396620192


     83%|████████▎ | 1049/1261 [23:02<05:05,  1.44s/it]

    500.639643315 m 468.177662129 m
    curve_diff:  32.4619811855


     83%|████████▎ | 1050/1261 [23:03<05:09,  1.47s/it]

    557.631904642 m 855.186748241 m
    curve_diff:  297.5548436


     83%|████████▎ | 1051/1261 [23:05<05:17,  1.51s/it]

    815.271400429 m 1386.14604164 m
    curve_diff:  570.87464121


     83%|████████▎ | 1052/1261 [23:06<05:10,  1.49s/it]

    1055.00262566 m 1479.24455461 m
    curve_diff:  424.241928948


     84%|████████▎ | 1053/1261 [23:08<05:05,  1.47s/it]

    761.329244735 m 1059.71184272 m
    curve_diff:  298.382597987


     84%|████████▎ | 1054/1261 [23:09<05:01,  1.46s/it]

    724.841227018 m 594.48227129 m
    curve_diff:  130.358955728


     84%|████████▎ | 1055/1261 [23:11<04:59,  1.46s/it]

    736.979771947 m 440.569600024 m
    curve_diff:  296.410171923


     84%|████████▎ | 1056/1261 [23:12<04:56,  1.45s/it]

    764.579415079 m 451.767154993 m
    curve_diff:  312.812260086


     84%|████████▍ | 1057/1261 [23:13<04:54,  1.44s/it]

    747.28680861 m 466.686658309 m
    curve_diff:  280.600150302


     84%|████████▍ | 1058/1261 [23:15<05:04,  1.50s/it]

    666.017518906 m 462.493730298 m
    curve_diff:  203.523788608


     84%|████████▍ | 1059/1261 [23:17<05:07,  1.52s/it]

    637.445517635 m 375.138862427 m
    curve_diff:  262.306655207


     84%|████████▍ | 1060/1261 [23:18<04:59,  1.49s/it]

    561.475130009 m 361.49556726 m
    curve_diff:  199.979562749


     84%|████████▍ | 1061/1261 [23:20<04:56,  1.48s/it]

    518.05505127 m 334.574482252 m
    curve_diff:  183.480569018


     84%|████████▍ | 1062/1261 [23:21<04:52,  1.47s/it]

    560.489463219 m 374.764977228 m
    curve_diff:  185.724485991


     84%|████████▍ | 1063/1261 [23:22<04:47,  1.45s/it]

    694.52793592 m 411.744084219 m
    curve_diff:  282.783851702


     84%|████████▍ | 1064/1261 [23:24<04:45,  1.45s/it]

    732.307128113 m 452.090171665 m
    curve_diff:  280.216956449


     84%|████████▍ | 1065/1261 [23:25<04:48,  1.47s/it]

    973.635496376 m 572.597692807 m
    curve_diff:  401.037803569


     85%|████████▍ | 1066/1261 [23:27<04:42,  1.45s/it]

    1115.05500088 m 624.624668743 m
    curve_diff:  490.430332133


     85%|████████▍ | 1067/1261 [23:28<04:37,  1.43s/it]

    1076.82889506 m 495.522280055 m
    curve_diff:  581.306615009


     85%|████████▍ | 1068/1261 [23:30<04:37,  1.44s/it]

    1019.65029722 m 516.083475299 m
    curve_diff:  503.56682192


     85%|████████▍ | 1069/1261 [23:31<04:36,  1.44s/it]

    973.097640424 m 532.941759577 m
    curve_diff:  440.155880847


     85%|████████▍ | 1070/1261 [23:33<04:36,  1.45s/it]

    919.82149143 m 90.63657536 m
    curve_diff:  829.18491607


     85%|████████▍ | 1071/1261 [23:34<04:28,  1.41s/it]

    947.867850046 m 77.6459421003 m
    curve_diff:  870.221907945


     85%|████████▌ | 1072/1261 [23:35<04:34,  1.45s/it]

    854.100987662 m 599.176617041 m
    curve_diff:  254.924370621


     85%|████████▌ | 1073/1261 [23:37<04:43,  1.51s/it]

    826.19203117 m 830.995606351 m
    curve_diff:  4.80357518129


     85%|████████▌ | 1074/1261 [23:39<04:40,  1.50s/it]

    819.167232799 m 814.583959185 m
    curve_diff:  4.58327361443


     85%|████████▌ | 1075/1261 [23:40<04:31,  1.46s/it]

    889.183390456 m 872.438764295 m
    curve_diff:  16.744626161


     85%|████████▌ | 1076/1261 [23:41<04:31,  1.47s/it]

    830.909713341 m 744.085942704 m
    curve_diff:  86.8237706367


     85%|████████▌ | 1077/1261 [23:43<04:27,  1.45s/it]

    668.063077478 m 499.746337475 m
    curve_diff:  168.316740003


     85%|████████▌ | 1078/1261 [23:44<04:22,  1.44s/it]

    668.034185349 m 464.429302194 m
    curve_diff:  203.604883156


     86%|████████▌ | 1079/1261 [23:46<04:19,  1.43s/it]

    702.993053904 m 508.916758323 m
    curve_diff:  194.076295581


     86%|████████▌ | 1080/1261 [23:47<04:21,  1.44s/it]

    663.731136872 m 626.720500497 m
    curve_diff:  37.0106363759


     86%|████████▌ | 1081/1261 [23:48<04:16,  1.42s/it]

    773.272013472 m 580.368253828 m
    curve_diff:  192.903759644


     86%|████████▌ | 1082/1261 [23:50<04:12,  1.41s/it]

    677.534434104 m 592.101689412 m
    curve_diff:  85.4327446915


     86%|████████▌ | 1083/1261 [23:51<04:09,  1.40s/it]

    675.035548364 m 485.290193091 m
    curve_diff:  189.745355273


     86%|████████▌ | 1084/1261 [23:53<04:08,  1.40s/it]

    705.90150691 m 499.344362883 m
    curve_diff:  206.557144027


     86%|████████▌ | 1085/1261 [23:54<04:07,  1.41s/it]

    787.899702512 m 442.621558758 m
    curve_diff:  345.278143753


     86%|████████▌ | 1086/1261 [23:55<04:08,  1.42s/it]

    772.044980505 m 463.643214407 m
    curve_diff:  308.401766098


     86%|████████▌ | 1087/1261 [23:57<04:09,  1.44s/it]

    861.598879175 m 647.874209462 m
    curve_diff:  213.724669713


     86%|████████▋ | 1088/1261 [23:58<04:07,  1.43s/it]

    1069.68466076 m 604.816603488 m
    curve_diff:  464.868057274


     86%|████████▋ | 1089/1261 [24:00<04:04,  1.42s/it]

    1098.47952042 m 569.911825785 m
    curve_diff:  528.567694634


     86%|████████▋ | 1090/1261 [24:01<04:01,  1.41s/it]

    1195.77735867 m 450.147243868 m
    curve_diff:  745.630114802


     87%|████████▋ | 1091/1261 [24:03<03:59,  1.41s/it]

    1314.17641405 m 484.83350796 m
    curve_diff:  829.342906093


     87%|████████▋ | 1092/1261 [24:04<03:57,  1.41s/it]

    1550.93036985 m 604.702284155 m
    curve_diff:  946.228085695


     87%|████████▋ | 1093/1261 [24:05<03:57,  1.41s/it]

    1255.21274834 m 706.049933603 m
    curve_diff:  549.162814732


     87%|████████▋ | 1094/1261 [24:07<03:56,  1.42s/it]

    1404.9555998 m 831.160460192 m
    curve_diff:  573.795139612


     87%|████████▋ | 1095/1261 [24:08<03:59,  1.44s/it]

    1316.9999633 m 815.595083608 m
    curve_diff:  501.404879692


     87%|████████▋ | 1096/1261 [24:10<03:57,  1.44s/it]

    1422.12370753 m 1990.21233787 m
    curve_diff:  568.088630339


     87%|████████▋ | 1097/1261 [24:11<03:53,  1.43s/it]

    1131.64260483 m 3621.70630462 m
    curve_diff:  2490.06369979


     87%|████████▋ | 1098/1261 [24:13<04:02,  1.49s/it]

    1057.77521529 m 1146.05122104 m
    curve_diff:  88.2760057502


     87%|████████▋ | 1099/1261 [24:14<03:56,  1.46s/it]

    1066.74237321 m 927.486447646 m
    curve_diff:  139.255925563


     87%|████████▋ | 1100/1261 [24:16<03:53,  1.45s/it]

    1014.17056158 m 729.095026565 m
    curve_diff:  285.075535017


     87%|████████▋ | 1101/1261 [24:17<03:50,  1.44s/it]

    981.388094028 m 571.469163379 m
    curve_diff:  409.918930649


     87%|████████▋ | 1102/1261 [24:19<03:54,  1.47s/it]

    1173.94266155 m 743.66088128 m
    curve_diff:  430.281780268


     87%|████████▋ | 1103/1261 [24:20<03:48,  1.45s/it]

    1049.15846926 m 590.535497013 m
    curve_diff:  458.622972245


     88%|████████▊ | 1104/1261 [24:21<03:45,  1.43s/it]

    1071.6766927 m 611.028573565 m
    curve_diff:  460.648119136


     88%|████████▊ | 1105/1261 [24:23<03:42,  1.43s/it]

    885.760500313 m 644.347509573 m
    curve_diff:  241.41299074


     88%|████████▊ | 1106/1261 [24:24<03:54,  1.51s/it]

    905.327829651 m 470.985162157 m
    curve_diff:  434.342667494


     88%|████████▊ | 1107/1261 [24:26<03:51,  1.50s/it]

    890.239609552 m 418.142897259 m
    curve_diff:  472.096712292


     88%|████████▊ | 1108/1261 [24:27<03:45,  1.48s/it]

    886.515750321 m 498.665856897 m
    curve_diff:  387.849893424


     88%|████████▊ | 1109/1261 [24:29<03:47,  1.49s/it]

    901.672645815 m 534.97141386 m
    curve_diff:  366.701231955


     88%|████████▊ | 1110/1261 [24:30<03:42,  1.48s/it]

    991.737914712 m 524.396813006 m
    curve_diff:  467.341101706


     88%|████████▊ | 1111/1261 [24:32<03:38,  1.45s/it]

    1183.81027047 m 382.384597572 m
    curve_diff:  801.425672901


     88%|████████▊ | 1112/1261 [24:33<03:33,  1.43s/it]

    1087.80320303 m 402.520283357 m
    curve_diff:  685.282919672


     88%|████████▊ | 1113/1261 [24:35<03:31,  1.43s/it]

    1277.78131535 m 368.143500042 m
    curve_diff:  909.637815303


     88%|████████▊ | 1114/1261 [24:36<03:28,  1.42s/it]

    1279.07635422 m 380.125519421 m
    curve_diff:  898.950834801


     88%|████████▊ | 1115/1261 [24:37<03:30,  1.44s/it]

    1349.599496 m 408.625644816 m
    curve_diff:  940.973851182


     89%|████████▊ | 1116/1261 [24:39<03:30,  1.45s/it]

    1031.02933766 m 428.429977424 m
    curve_diff:  602.599360237


     89%|████████▊ | 1117/1261 [24:40<03:33,  1.48s/it]

    1035.94829934 m 454.989387533 m
    curve_diff:  580.958911802


     89%|████████▊ | 1118/1261 [24:42<03:28,  1.46s/it]

    939.046991165 m 612.053774041 m
    curve_diff:  326.993217124


     89%|████████▊ | 1119/1261 [24:43<03:29,  1.47s/it]

    940.876020987 m 626.820630794 m
    curve_diff:  314.055390193


     89%|████████▉ | 1120/1261 [24:45<03:24,  1.45s/it]

    916.60414586 m 758.78747642 m
    curve_diff:  157.816669439


     89%|████████▉ | 1121/1261 [24:46<03:20,  1.44s/it]

    964.816403054 m 825.22681023 m
    curve_diff:  139.589592824


     89%|████████▉ | 1122/1261 [24:48<03:19,  1.43s/it]

    1059.83066499 m 972.574098656 m
    curve_diff:  87.2565663364


     89%|████████▉ | 1123/1261 [24:49<03:16,  1.42s/it]

    1025.79733984 m 881.072282058 m
    curve_diff:  144.725057782


     89%|████████▉ | 1124/1261 [24:51<03:21,  1.47s/it]

    787.510131014 m 754.586140767 m
    curve_diff:  32.923990247


     89%|████████▉ | 1125/1261 [24:52<03:17,  1.45s/it]

    780.089510039 m 759.066573349 m
    curve_diff:  21.02293669


     89%|████████▉ | 1126/1261 [24:53<03:14,  1.44s/it]

    695.935397473 m 581.221058768 m
    curve_diff:  114.714338704


     89%|████████▉ | 1127/1261 [24:55<03:11,  1.43s/it]

    672.398398657 m 540.116169604 m
    curve_diff:  132.282229053


     89%|████████▉ | 1128/1261 [24:56<03:09,  1.42s/it]

    711.665932053 m 528.524849702 m
    curve_diff:  183.141082351


     90%|████████▉ | 1129/1261 [24:58<03:07,  1.42s/it]

    666.096386968 m 693.81754314 m
    curve_diff:  27.7211561716


     90%|████████▉ | 1130/1261 [24:59<03:05,  1.42s/it]

    809.931659396 m 503.666574514 m
    curve_diff:  306.265084882


     90%|████████▉ | 1131/1261 [25:00<03:05,  1.43s/it]

    852.58170264 m 733.394476666 m
    curve_diff:  119.187225974


     90%|████████▉ | 1132/1261 [25:02<03:04,  1.43s/it]

    908.204754573 m 863.873289548 m
    curve_diff:  44.3314650249


     90%|████████▉ | 1133/1261 [25:03<03:02,  1.43s/it]

    1118.66178712 m 1009.78523832 m
    curve_diff:  108.876548799


     90%|████████▉ | 1134/1261 [25:05<02:58,  1.41s/it]

    1387.32147464 m 700.68326295 m
    curve_diff:  686.638211692


     90%|█████████ | 1135/1261 [25:06<02:56,  1.40s/it]

    1196.77062026 m 582.316583021 m
    curve_diff:  614.454037236


     90%|█████████ | 1136/1261 [25:08<02:55,  1.40s/it]

    1331.96260318 m 515.590946462 m
    curve_diff:  816.371656721


     90%|█████████ | 1137/1261 [25:09<02:54,  1.41s/it]

    1147.13560564 m 579.631206814 m
    curve_diff:  567.504398826


     90%|█████████ | 1138/1261 [25:10<02:56,  1.44s/it]

    1107.01089983 m 592.82782218 m
    curve_diff:  514.183077649


     90%|█████████ | 1139/1261 [25:12<02:56,  1.45s/it]

    1057.23852454 m 524.486883713 m
    curve_diff:  532.751640827


     90%|█████████ | 1140/1261 [25:13<02:57,  1.47s/it]

    1155.90380667 m 605.575856271 m
    curve_diff:  550.327950395


     90%|█████████ | 1141/1261 [25:15<02:58,  1.49s/it]

    1318.16636935 m 560.399051374 m
    curve_diff:  757.767317972


     91%|█████████ | 1142/1261 [25:17<02:59,  1.51s/it]

    1204.41349508 m 637.08950918 m
    curve_diff:  567.323985896


     91%|█████████ | 1143/1261 [25:18<02:59,  1.52s/it]

    1083.84264354 m 674.128632259 m
    curve_diff:  409.714011285


     91%|█████████ | 1144/1261 [25:20<02:55,  1.50s/it]

    1094.32559338 m 618.745189616 m
    curve_diff:  475.580403765


     91%|█████████ | 1145/1261 [25:21<02:50,  1.47s/it]

    1019.53373988 m 456.986110216 m
    curve_diff:  562.54762966


     91%|█████████ | 1146/1261 [25:22<02:52,  1.50s/it]

    1107.50536308 m 511.296356684 m
    curve_diff:  596.209006395


     91%|█████████ | 1147/1261 [25:24<02:53,  1.52s/it]

    980.475266402 m 427.59947879 m
    curve_diff:  552.875787611


     91%|█████████ | 1148/1261 [25:26<02:56,  1.56s/it]

    831.308025239 m 493.203515365 m
    curve_diff:  338.104509874


     91%|█████████ | 1149/1261 [25:27<02:50,  1.52s/it]

    878.397507136 m 514.387728311 m
    curve_diff:  364.009778824


     91%|█████████ | 1150/1261 [25:29<02:46,  1.50s/it]

    649.836917564 m 506.908736299 m
    curve_diff:  142.928181265


     91%|█████████▏| 1151/1261 [25:30<02:40,  1.46s/it]

    656.181118511 m 539.292768335 m
    curve_diff:  116.888350176


     91%|█████████▏| 1152/1261 [25:31<02:38,  1.45s/it]

    594.883370317 m 462.192448144 m
    curve_diff:  132.690922174


     91%|█████████▏| 1153/1261 [25:33<02:40,  1.48s/it]

    597.757347727 m 612.664906299 m
    curve_diff:  14.9075585718


     92%|█████████▏| 1154/1261 [25:34<02:37,  1.47s/it]

    617.110112851 m 542.284581045 m
    curve_diff:  74.8255318062


     92%|█████████▏| 1155/1261 [25:36<02:34,  1.46s/it]

    582.11956289 m 674.089021579 m
    curve_diff:  91.9694586887


     92%|█████████▏| 1156/1261 [25:37<02:31,  1.45s/it]

    559.121034206 m 672.510474954 m
    curve_diff:  113.389440748


     92%|█████████▏| 1157/1261 [25:39<02:30,  1.45s/it]

    575.727652954 m 774.395442335 m
    curve_diff:  198.667789381


     92%|█████████▏| 1158/1261 [25:40<02:29,  1.45s/it]

    543.541457034 m 524.062322358 m
    curve_diff:  19.4791346757


     92%|█████████▏| 1159/1261 [25:42<02:27,  1.45s/it]

    587.086066428 m 509.859203216 m
    curve_diff:  77.2268632126


     92%|█████████▏| 1160/1261 [25:43<02:28,  1.47s/it]

    561.457044421 m 416.487892475 m
    curve_diff:  144.969151947


     92%|█████████▏| 1161/1261 [25:45<02:25,  1.46s/it]

    523.144740363 m 439.485374169 m
    curve_diff:  83.6593661946


     92%|█████████▏| 1162/1261 [25:46<02:23,  1.45s/it]

    562.373834031 m 450.560645937 m
    curve_diff:  111.813188095


     92%|█████████▏| 1163/1261 [25:47<02:23,  1.47s/it]

    604.343006492 m 442.690923483 m
    curve_diff:  161.652083009


     92%|█████████▏| 1164/1261 [25:49<02:21,  1.46s/it]

    657.753284118 m 520.182184741 m
    curve_diff:  137.571099377


     92%|█████████▏| 1165/1261 [25:50<02:18,  1.45s/it]

    688.084212561 m 505.674293587 m
    curve_diff:  182.409918973


     92%|█████████▏| 1166/1261 [25:52<02:16,  1.44s/it]

    706.274959583 m 608.636815278 m
    curve_diff:  97.6381443045


     93%|█████████▎| 1167/1261 [25:53<02:16,  1.45s/it]

    944.778052634 m 738.073806897 m
    curve_diff:  206.704245738


     93%|█████████▎| 1168/1261 [25:55<02:14,  1.45s/it]

    1001.56542607 m 593.917835506 m
    curve_diff:  407.647590559


     93%|█████████▎| 1169/1261 [25:56<02:12,  1.45s/it]

    932.046403307 m 473.511642502 m
    curve_diff:  458.534760805


     93%|█████████▎| 1170/1261 [25:58<02:13,  1.47s/it]

    1011.25174263 m 481.636033994 m
    curve_diff:  529.615708639


     93%|█████████▎| 1171/1261 [25:59<02:11,  1.46s/it]

    1073.34727194 m 539.774411542 m
    curve_diff:  533.572860396


     93%|█████████▎| 1172/1261 [26:00<02:08,  1.45s/it]

    1110.21712542 m 573.412306942 m
    curve_diff:  536.804818476


     93%|█████████▎| 1173/1261 [26:02<02:06,  1.44s/it]

    1262.63762574 m 597.368278077 m
    curve_diff:  665.269347664


     93%|█████████▎| 1174/1261 [26:03<02:05,  1.44s/it]

    1098.57016004 m 651.763668172 m
    curve_diff:  446.806491864


     93%|█████████▎| 1175/1261 [26:05<02:05,  1.46s/it]

    1303.50277229 m 597.771388576 m
    curve_diff:  705.731383711


     93%|█████████▎| 1176/1261 [26:06<02:03,  1.45s/it]

    1339.19680427 m 925.91525913 m
    curve_diff:  413.281545144


     93%|█████████▎| 1177/1261 [26:08<02:02,  1.46s/it]

    1281.81684353 m 739.024090511 m
    curve_diff:  542.792753021


     93%|█████████▎| 1178/1261 [26:09<02:00,  1.45s/it]

    1461.74284053 m 767.031009878 m
    curve_diff:  694.711830652


     93%|█████████▎| 1179/1261 [26:11<01:59,  1.45s/it]

    1223.22569345 m 658.806466483 m
    curve_diff:  564.419226963


     94%|█████████▎| 1180/1261 [26:12<01:56,  1.44s/it]

    1081.53325144 m 566.773084808 m
    curve_diff:  514.76016663


     94%|█████████▎| 1181/1261 [26:13<01:54,  1.43s/it]

    1495.02627842 m 520.667856981 m
    curve_diff:  974.358421442


     94%|█████████▎| 1182/1261 [26:15<01:54,  1.45s/it]

    1072.46280174 m 472.148792367 m
    curve_diff:  600.314009376


     94%|█████████▍| 1183/1261 [26:16<01:53,  1.45s/it]

    958.28928528 m 504.091781525 m
    curve_diff:  454.197503755


     94%|█████████▍| 1184/1261 [26:18<01:51,  1.45s/it]

    823.753273013 m 534.428780187 m
    curve_diff:  289.324492826


     94%|█████████▍| 1185/1261 [26:19<01:49,  1.43s/it]

    654.814183906 m 468.082321484 m
    curve_diff:  186.731862422


     94%|█████████▍| 1186/1261 [26:21<01:47,  1.44s/it]

    698.332956651 m 513.496219028 m
    curve_diff:  184.836737623


     94%|█████████▍| 1187/1261 [26:22<01:46,  1.44s/it]

    644.305659325 m 824.400410153 m
    curve_diff:  180.094750828


     94%|█████████▍| 1188/1261 [26:24<01:51,  1.53s/it]

    715.018143154 m 794.994357722 m
    curve_diff:  79.9762145676


     94%|█████████▍| 1189/1261 [26:26<01:56,  1.61s/it]

    673.359654314 m 913.766733159 m
    curve_diff:  240.407078845


     94%|█████████▍| 1190/1261 [26:27<01:50,  1.56s/it]

    652.215621781 m 684.840623983 m
    curve_diff:  32.6250022015


     94%|█████████▍| 1191/1261 [26:29<01:45,  1.51s/it]

    661.275896664 m 528.614139379 m
    curve_diff:  132.661757285


     95%|█████████▍| 1192/1261 [26:30<01:42,  1.49s/it]

    651.654559779 m 478.529590828 m
    curve_diff:  173.124968951


     95%|█████████▍| 1193/1261 [26:31<01:39,  1.47s/it]

    590.701073852 m 465.72082713 m
    curve_diff:  124.980246723


     95%|█████████▍| 1194/1261 [26:33<01:38,  1.46s/it]

    605.190452744 m 475.382369191 m
    curve_diff:  129.808083553


     95%|█████████▍| 1195/1261 [26:34<01:36,  1.47s/it]

    572.753972464 m 442.537099959 m
    curve_diff:  130.216872505


     95%|█████████▍| 1196/1261 [26:36<01:36,  1.48s/it]

    543.908397858 m 441.995588191 m
    curve_diff:  101.912809667


     95%|█████████▍| 1197/1261 [26:37<01:33,  1.46s/it]

    509.58468068 m 469.928520404 m
    curve_diff:  39.6561602761


     95%|█████████▌| 1198/1261 [26:39<01:32,  1.47s/it]

    568.018560086 m 517.174142981 m
    curve_diff:  50.8444171055


     95%|█████████▌| 1199/1261 [26:40<01:30,  1.45s/it]

    653.954511311 m 563.978172152 m
    curve_diff:  89.9763391596


     95%|█████████▌| 1200/1261 [26:42<01:28,  1.45s/it]

    752.418055992 m 473.60017349 m
    curve_diff:  278.817882502


     95%|█████████▌| 1201/1261 [26:43<01:26,  1.45s/it]

    698.951092977 m 519.361057833 m
    curve_diff:  179.590035143


     95%|█████████▌| 1202/1261 [26:44<01:25,  1.45s/it]

    797.15164346 m 615.083167979 m
    curve_diff:  182.068475481


     95%|█████████▌| 1203/1261 [26:46<01:24,  1.46s/it]

    696.745866809 m 473.89339181 m
    curve_diff:  222.852474999


     95%|█████████▌| 1204/1261 [26:47<01:22,  1.46s/it]

    662.740706041 m 396.765497793 m
    curve_diff:  265.975208248


     96%|█████████▌| 1205/1261 [26:49<01:21,  1.45s/it]

    641.41543726 m 380.766819449 m
    curve_diff:  260.648617812


     96%|█████████▌| 1206/1261 [26:50<01:19,  1.44s/it]

    602.45521136 m 368.523007971 m
    curve_diff:  233.932203389


     96%|█████████▌| 1207/1261 [26:52<01:18,  1.45s/it]

    551.089160554 m 376.506654433 m
    curve_diff:  174.58250612


     96%|█████████▌| 1208/1261 [26:53<01:16,  1.44s/it]

    587.342167102 m 465.463608472 m
    curve_diff:  121.87855863


     96%|█████████▌| 1209/1261 [26:55<01:15,  1.44s/it]

    609.02454974 m 583.513610389 m
    curve_diff:  25.5109393509


     96%|█████████▌| 1210/1261 [26:56<01:14,  1.46s/it]

    664.449509387 m 614.428592328 m
    curve_diff:  50.020917059


     96%|█████████▌| 1211/1261 [26:58<01:13,  1.48s/it]

    677.520892445 m 522.378918887 m
    curve_diff:  155.141973558


     96%|█████████▌| 1212/1261 [26:59<01:11,  1.46s/it]

    717.932540512 m 536.576740375 m
    curve_diff:  181.355800137


     96%|█████████▌| 1213/1261 [27:00<01:09,  1.45s/it]

    763.99683561 m 552.3717211 m
    curve_diff:  211.62511451


     96%|█████████▋| 1214/1261 [27:02<01:07,  1.45s/it]

    857.749718277 m 500.503585551 m
    curve_diff:  357.246132727


     96%|█████████▋| 1215/1261 [27:03<01:06,  1.44s/it]

    904.176742883 m 508.1526339 m
    curve_diff:  396.024108983


     96%|█████████▋| 1216/1261 [27:05<01:04,  1.44s/it]

    865.438121434 m 477.803990579 m
    curve_diff:  387.634130855


     97%|█████████▋| 1217/1261 [27:06<01:03,  1.45s/it]

    1004.44503155 m 497.754725403 m
    curve_diff:  506.690306144


     97%|█████████▋| 1218/1261 [27:08<01:03,  1.48s/it]

    988.207024849 m 523.561753397 m
    curve_diff:  464.645271452


     97%|█████████▋| 1219/1261 [27:09<01:01,  1.46s/it]

    1043.38839173 m 549.274932456 m
    curve_diff:  494.113459277


     97%|█████████▋| 1220/1261 [27:11<00:59,  1.46s/it]

    910.451875003 m 606.862336343 m
    curve_diff:  303.58953866


     97%|█████████▋| 1221/1261 [27:12<00:58,  1.45s/it]

    1011.54500225 m 581.104289957 m
    curve_diff:  430.440712293


     97%|█████████▋| 1222/1261 [27:14<00:56,  1.46s/it]

    883.954349113 m 758.116524547 m
    curve_diff:  125.837824566


     97%|█████████▋| 1223/1261 [27:15<00:55,  1.46s/it]

    889.022816155 m 619.999204773 m
    curve_diff:  269.023611382


     97%|█████████▋| 1224/1261 [27:16<00:53,  1.46s/it]

    1010.60798286 m 524.362056451 m
    curve_diff:  486.245926409


     97%|█████████▋| 1225/1261 [27:18<00:53,  1.48s/it]

    1025.8274357 m 451.74188779 m
    curve_diff:  574.085547914


     97%|█████████▋| 1226/1261 [27:20<00:51,  1.48s/it]

    990.555121103 m 490.860524016 m
    curve_diff:  499.694597087


     97%|█████████▋| 1227/1261 [27:21<00:49,  1.47s/it]

    1260.54202046 m 530.818686787 m
    curve_diff:  729.723333673


     97%|█████████▋| 1228/1261 [27:22<00:48,  1.46s/it]

    1173.50744333 m 479.541638912 m
    curve_diff:  693.965804417


     97%|█████████▋| 1229/1261 [27:24<00:46,  1.45s/it]

    1268.09765111 m 577.141122095 m
    curve_diff:  690.956529018


     98%|█████████▊| 1230/1261 [27:25<00:44,  1.45s/it]

    1692.89876269 m 1246.13840961 m
    curve_diff:  446.760353075


     98%|█████████▊| 1231/1261 [27:27<00:43,  1.45s/it]

    1708.80784256 m 951.450764633 m
    curve_diff:  757.357077929


     98%|█████████▊| 1232/1261 [27:28<00:42,  1.47s/it]

    1871.82556251 m 743.60682724 m
    curve_diff:  1128.21873527


     98%|█████████▊| 1233/1261 [27:30<00:41,  1.47s/it]

    1712.77521702 m 3639.78293448 m
    curve_diff:  1927.00771746


     98%|█████████▊| 1234/1261 [27:31<00:40,  1.49s/it]

    2854.78286096 m 2659.20112502 m
    curve_diff:  195.581735946


     98%|█████████▊| 1235/1261 [27:33<00:38,  1.48s/it]

    2551.32076371 m 2486.1218134 m
    curve_diff:  65.1989503062


     98%|█████████▊| 1236/1261 [27:34<00:36,  1.47s/it]

    2478.30720476 m 1350.62624334 m
    curve_diff:  1127.68096141


     98%|█████████▊| 1237/1261 [27:36<00:35,  1.46s/it]

    3630.28960974 m 1296.98530058 m
    curve_diff:  2333.30430916


     98%|█████████▊| 1238/1261 [27:37<00:33,  1.46s/it]

    4132.05429194 m 959.502717249 m
    curve_diff:  3172.55157469


     98%|█████████▊| 1239/1261 [27:39<00:32,  1.46s/it]

    2417.23765618 m 2042.98837115 m
    curve_diff:  374.24928503


     98%|█████████▊| 1240/1261 [27:40<00:31,  1.49s/it]

    2536.15183783 m 1809.42773008 m
    curve_diff:  726.724107751


     98%|█████████▊| 1241/1261 [27:42<00:29,  1.47s/it]

    2619.72185688 m 1552.31445676 m
    curve_diff:  1067.40740012


     98%|█████████▊| 1242/1261 [27:43<00:27,  1.47s/it]

    2755.27835586 m 3444.75637236 m
    curve_diff:  689.478016501


     99%|█████████▊| 1243/1261 [27:44<00:26,  1.48s/it]

    3671.6337678 m 21222.2963489 m
    curve_diff:  17550.6625811


     99%|█████████▊| 1244/1261 [27:46<00:25,  1.48s/it]

    3034.44144511 m 2416.34440273 m
    curve_diff:  618.097042385


     99%|█████████▊| 1245/1261 [27:47<00:23,  1.47s/it]

    2968.85333534 m 3243.86779187 m
    curve_diff:  275.014456532


     99%|█████████▉| 1246/1261 [27:49<00:22,  1.48s/it]

    2855.18711084 m 13553.2110807 m
    curve_diff:  10698.0239699


     99%|█████████▉| 1247/1261 [27:51<00:21,  1.52s/it]

    3265.91198889 m 2023.71211247 m
    curve_diff:  1242.19987643


     99%|█████████▉| 1248/1261 [27:52<00:19,  1.50s/it]

    2928.5243796 m 1751.87982324 m
    curve_diff:  1176.64455636


     99%|█████████▉| 1249/1261 [27:53<00:17,  1.48s/it]

    3033.80671018 m 3082.20519279 m
    curve_diff:  48.398482614


     99%|█████████▉| 1250/1261 [27:55<00:16,  1.48s/it]

    2732.61662161 m 2029.3805915 m
    curve_diff:  703.236030113


     99%|█████████▉| 1251/1261 [27:56<00:14,  1.49s/it]

    1793.48213128 m 2837.59538343 m
    curve_diff:  1044.11325216


     99%|█████████▉| 1252/1261 [27:58<00:13,  1.48s/it]

    1887.57766976 m 6673.0551268 m
    curve_diff:  4785.47745704


     99%|█████████▉| 1253/1261 [27:59<00:11,  1.47s/it]

    2627.06554274 m 1202.37252805 m
    curve_diff:  1424.69301468


     99%|█████████▉| 1254/1261 [28:01<00:10,  1.50s/it]

    9341.54304331 m 1643.73980969 m
    curve_diff:  7697.80323361


    100%|█████████▉| 1255/1261 [28:02<00:08,  1.49s/it]

    3311.69094009 m 4268.01078497 m
    curve_diff:  956.31984488


    100%|█████████▉| 1256/1261 [28:04<00:07,  1.49s/it]

    12873.2029817 m 3291.68896499 m
    curve_diff:  9581.51401675


    100%|█████████▉| 1257/1261 [28:05<00:05,  1.46s/it]

    9665.23827461 m 3132.31511391 m
    curve_diff:  6532.9231607


    100%|█████████▉| 1258/1261 [28:07<00:04,  1.47s/it]

    11087.7872428 m 997.126072401 m
    curve_diff:  10090.6611704


    100%|█████████▉| 1259/1261 [28:08<00:02,  1.45s/it]

    224609.291657 m 1187.64922976 m
    curve_diff:  223421.642428


    100%|█████████▉| 1260/1261 [28:10<00:01,  1.47s/it]

    77115.8638088 m 887.845796985 m
    curve_diff:  76228.0180119





    [MoviePy] Done.
    [MoviePy] >>>> Video ready: result.mp4

    CPU times: user 25min 12s, sys: 2min 36s, total: 27min 48s
    Wall time: 28min 11s



![png](output/output_28_2523.png)



```python

```
