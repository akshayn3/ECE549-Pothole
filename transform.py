import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # This is based on the specific geometry of the road and camera setup
    polygon = np.array([[
        (width * 0.1, height * 0.95),
        (width * 0.4, height * 0.6),
        (width * 0.6, height * 0.6),
        (width * 0.9, height * 0.95)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

def detect_yellow_edges(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a new range of yellow color in HSV
    lower_yellow = np.array([25, 0, 180])
    upper_yellow = np.array([50, 30, 220])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Use morphological operations to enhance the mask
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2)
    yellow_mask = cv2.erode(yellow_mask, kernel, iterations=1)

    edges = cv2.Canny(yellow_mask, 50, 150)

    return edges

def hough_lines(edges, rho, theta, threshold, minLineLength, maxLineGap):
    hough_lines = cv2.HoughLinesP(edges, rho, theta, threshold, lines=np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    return hough_lines

def average_slope_intercept(image, lines):
    right_fit = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope > 0:
                right_fit.append((slope, intercept))
    right_fit_average = np.average(right_fit, axis=0)
    right_line = draw_right_line(image, right_fit_average)
    
    return right_line

def draw_right_line(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, y1, x2, y2]

def draw_yellow_line(image, lines):
    if lines is not None:
        left_line = None
        max_length = 0
        mid_x = image.shape[1] / 2
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > max_length:
                    if abs(x1 + x2 - 2 * mid_x) < image.shape[1] / 4:  # Line should be near the middle
                        max_length = length
                        left_line = line

    return left_line[0]


def adjust_line_length(line, desired_length):
    # Extract points from the line
    x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    
    # Calculate the current length of the line
    current_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Avoid division by zero
    if current_length == 0:
        return line
    
    scale_factor = desired_length / current_length
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # Adjust endpoints based on the scale factor
    x1 = mid_x + (x1 - mid_x) * scale_factor
    y1 = mid_y + (y1 - mid_y) * scale_factor
    x2 = mid_x + (x2 - mid_x) * scale_factor
    y2 = mid_y + (y2 - mid_y) * scale_factor
    
    # Convert points to integers if needed
    adjusted_line = [int(x1), int(y1), int(x2), int(y2)]
    
    return adjusted_line

def normalize_line(right_line, left_line):
    right_line[0] = right_line[0] - 350
    right_line[2] = right_line[2] - 292
    right_line[1] = left_line[1]
    right_line[3] = left_line[3]

    # Make slope of right line equal to slope of left line
    slope_l = (left_line[3] - left_line[1]) / (left_line[2] - left_line[0])

    # print("Slope of left line: ", slope_l)
    right_line[0] = int(right_line[2] - (right_line[3] - right_line[1]) / (-slope_l))
    slope_r = (right_line[3] - right_line[1]) / (right_line[2] - right_line[0])
    # print("Slope of right line: ", slope_r)

    # print("Right line: ", right_line)

    return right_line


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        # print(lines)
        for line in lines:
            cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 10)
    return line_image

def perspective_transform(image, lines, file_num):
   
    src_points = np.float32([
        [lines[0][0], lines[0][1]],
        [lines[0][2], lines[0][3]],
        [lines[1][2], lines[1][3]],
        [lines[1][0], lines[1][1]]
    ])


    dst_points = np.float32([
        [0, image.shape[0]],
        [0, 0],
        [image.shape[1], 0],
        [image.shape[1], image.shape[0]]
    ])
    # print(dst_points)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    
    # Save the result
    cv2.imwrite('IMG_Warped/transformed_image_' + str(file_num) + '.jpg', warped)

    line_image = display_lines(image, lines)
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

    cv2.imwrite('IMG_Warped_Lines/transformed_image_with_lines_' + str(file_num) + '.jpg', combo_image)

def main():
    image_path = 'Videos/IMG_1336/frame001180.jpg'
    image = cv2.imread(image_path)

    left_edge = detect_yellow_edges(image)
    right_temp = detect_edges(image)
    right_edge = region_of_interest(right_temp)

    left_line = hough_lines(left_edge, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    right_line = hough_lines(right_edge, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

    left_line = draw_yellow_line(image, left_line)
    right_line = average_slope_intercept(image, right_line)
    
    left_extended_line = adjust_line_length(left_line, 200)
    right_extended_line = adjust_line_length(right_line, 200)

    right_extended_line = normalize_line(right_extended_line, left_extended_line)
    
    lines = [left_extended_line, right_extended_line]

    for i in os.listdir('Videos/IMG_1336'):
        print(i)
        image = cv2.imread('Videos/IMG_1336/' + i)
        file_num = i.split('.')[0].split('frame')[1]
        perspective_transform(image, lines, file_num)
        

    # line_image = display_lines(image, lines)
    # combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

    # cv2.imshow('Detected Lanes', combo_image)
    # # cv2.imwrite('detected_lanes.jpg', combo_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

main()