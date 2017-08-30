from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque

# load the camera calibration result
with open("./dist_pickle.p", mode='rb') as f:
    dist_pickle = pickle.load(f)

mtx, dist = dist_pickle["mtx"], dist_pickle["dist"]

def undist(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100), mag_thresh=(0, 200)):

    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # gradient magnitude
    gradmag = np.sqrt(sobelx**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    grad_binary = np.zeros_like(gradmag)
    grad_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((s_binary == 1) | (sxbinary == 1)) & (grad_binary == 1)] = 1

    return combined_binary


# hardcode the source and destination points
src_points = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst_points = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])


M = cv2.getPerspectiveTransform(src_points, dst_points)

def perspective(img):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def get_lane_lines(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

Minv = cv2.getPerspectiveTransform(dst_points, src_points)

def draw_lane_line(img, left_fitx, right_fitx, ploty):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img[:,:,1]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = deque([],maxlen=5)

        self.recent_fitted = deque([],maxlen=5)
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        self.yvals = None

        # recent fit line base pos
        self.recent_line_base_pos = None

    def set_diffs(self):
        if len(self.recent_xfitted) > 0 and self.best_fit is not None:
            self.diffs = self.current_fit - self.best_fit
        else:
            self.diffs = np.array([0,0,0], dtype='float')

    def set_line_pos_pos(self):
        y_eval = np.max(self.yvals)
        line_pos = self.current_fit[0]*y_eval**2 + self.current_fit[1]*y_eval + self.current_fit[2]
        center_pos = 640
        self.line_base_pos =abs((line_pos - center_pos)*3.7/700.0) # 3.7/700 meters per pixel in x dimension

    def sanity_check(self):
        flag = True

        if(self.recent_line_base_pos):
            if(abs(self.line_base_pos - self.recent_line_base_pos)/self.recent_line_base_pos > 0.3): # change over 30%
                print('lane width change too much')
                flag  = False

#         if (abs(self.line_base_pos) > 3): # over 6
#             print(self.line_base_pos)
#             print(recent_line_base_pos)
#             print('lane width too narrow or wide')
#             flag  = False

        # TODO what scale of change is unexpected change?
        if(len(self.recent_xfitted) > 0 and self.best_fit!=None and self.diffs!=None):
            fit_delta = self.diffs / self.best_fit
            if not (abs(fit_delta) < np.array([0.8,0.8,0.15])).all():
                print('fit change too much [%]', fit_delta)
                flag=False

        return flag


    def add_line(self):
        current_fit_xvals = self.current_fit[0]*self.yvals**2 + self.current_fit[1]*self.yvals + self.current_fit[2]
        self.recent_xfitted.appendleft(current_fit_xvals)

        self.recent_fitted.appendleft(self.current_fit)

    def set_bestx(self):
        fits = self.recent_xfitted
        if len(fits)>0:
            avg=0
            for fit in fits:
                avg +=np.array(fit)
            avg = avg / len(fits)
            self.bestx = avg

    def set_bestfit(self):
        recent_fitted = self.recent_fitted
        if len(recent_fitted)>0:
            avg=0
            for fit in recent_fitted:
                avg +=np.array(fit)
            avg = avg / len(recent_fitted)
            self.best_fit = avg

    def set_radius_of_curvature(self):
        y_eval = np.max(self.yvals)
        if self.best_fit is not None:
            self.radius_of_curvature = ((1 + (2*self.best_fit[0]*y_eval + self.best_fit[1])**2)**1.5) / np.absolute(2*self.best_fit[0])

    def remove_old_line(self):
        if(len(self.recent_fitted) > 0):
            self.recent_xfitted.pop()
            self.recent_fitted.pop()

    def add_candidate_line(self, lanex, laney, yvals):
        self.allx = lanex
        self.ally = laney
        self.yvals = yvals
        self.current_fit = np.polyfit(laney, lanex, 2)
        self.set_diffs()
        self.set_line_pos_pos()

        if self.sanity_check():
            self.detected = True
            self.add_line()
            self.set_bestx()
            self.set_bestfit()
            self.set_radius_of_curvature()
            self.recent_line_base_pos = self.line_base_pos
        else:
            self.detected=False
            self.remove_old_line()
            self.set_bestfit()
            self.set_bestx()


def pipeline(img):
    global left_line
    global right_line
    undist_img = undist(img)
    threshold_img = threshold(undist_img)
    perspective_img = perspective(threshold_img)
    leftx, lefty, rightx, righty = get_lane_lines(perspective_img)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

    left_line.add_candidate_line(leftx, lefty, ploty)
    right_line.add_candidate_line(rightx, righty, ploty)

    left_fitx = left_line.bestx
    right_fitx = right_line.bestx


    # we only need y from 0..720 and the response x to draw a line
    result = draw_lane_line(img, left_fitx, right_fitx, ploty)

    font = cv2.FONT_HERSHEY_SIMPLEX

    distance_of_center = round((left_line.line_base_pos - right_line.line_base_pos)/2, 2)
    if(distance_of_center < 0):
        str1 = str('Vehicle is '+str(abs(distance_of_center))+'m left of center')
    else:
        str1 = str('Vehicle is '+str(abs(distance_of_center))+'m right of center')
    cv2.putText(result,str1,(430,100), font, 1,(255,255,255),2,cv2.LINE_AA)

    if left_line.radius_of_curvature and right_line.radius_of_curvature:
        curvature = 0.5*(round(right_line.radius_of_curvature/1000,1) + round(left_line.radius_of_curvature/1000,1))
        str2 = str('radius of curvature: '+str(curvature)+'km')
        cv2.putText(result,str2,(430,150), font, 1,(255,255,255),2,cv2.LINE_AA)

#     str3 = str(round(right_line.current_fit[0],2)) + " "+str(round(right_line.current_fit[1],2)) + " "+str(round(right_line.current_fit[2],2))
#     cv2.putText(result,str3,(430,200), font, 1,(255,255,255),2,cv2.LINE_AA)

#     str4 = str(round(right_line.best_fit[0],2)) + " "+str(round(right_line.best_fit[1],2)) + " "+str(round(right_line.best_fit[2],2))
#     cv2.putText(result,str4,(430,250), font, 1,(255,255,255),2,cv2.LINE_AA)

    return result