import cv2
import numpy as np
import pandas as pd
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize,medial_axis
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage.measure import regionprops_table, regionprops, label
from utils.math import tortuosity
import math



def vessel_width(thresholded_image: np.ndarray, landmarks: list):
    image = thresholded_image.copy()
    widths = []
    for j in landmarks:
        for el in j:
            w0 = w45 = w90 = w135 = 0
            w180 = w225 = w270 = w315 = 1
            while True:
                if image[el[0], el[1] + w0 + 1] != 0:
                    w0 += 1
                if image[el[0], el[1] - w180 - 1] != 0:
                    w180 += 1
                if image[el[0] - w45 - 1, el[1] + w45 + 1] != 0:
                    w45 += 1
                if image[el[0] + w225 + 1, el[1] - w225 - 1] != 0:
                    w225 += 1
                if image[el[0] - w90 - 1, el[1]] != 0:
                    w90 += 1
                if image[el[0] + w270 + 1, el[1]] != 0:
                    w270 += 1
                if image[el[0] - w135 - 1, el[1] - w135 - 1] != 0:
                    w135 += 1
                if image[el[0] + w315 + 1, el[1] + w315 + 1] != 0:
                    w315 += 1

                if image[el[0], el[1] + w0 + 1] == 0 and image[el[0], el[1] - w180 - 1] == 0:
                    widths.append([0, w0, w180])
                    break
                elif image[el[0] - w45 - 1, el[1] + w45 + 1] == 0 and image[el[0] + w225 + 1, el[1] - w225 - 1] == 0:
                    widths.append([45, w45, w225])
                    break
                elif image[el[0] - w90 - 1, el[1]] == 0 and image[el[0] + w270 + 1, el[1]] == 0:
                    widths.append([90, w90, w270])
                    break
                elif image[el[0] - w135 - 1, el[1] - w135 - 1] == 0 and image[el[0] + w315 + 1, el[1] + w315 + 1] == 0:
                    widths.append([135, w135, w315])
                    break

    return widths


def getEndPoints(skeleton):
    '''
    Find the endpoints of the skeleton.
    '''

    end_points = list()
    end_structs = [np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 0, 1],
                         [0, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 0, 0],
                         [1, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 1],
                         [0, 0, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [1, 0, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])]

    for i, struct in enumerate(end_structs):
        hits = nd.binary_hit_or_miss(skeleton, structure1=struct)

        if not np.any(hits):
            continue

        for y, x in zip(*np.where(hits)):
            end_points.append((x, y))
        # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in end_points:
        for point2 in end_points:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                end_points.remove(point2);
    # Remove duplicates
    end_points = list(set(end_points));        

    return end_points

def getIntersections(skeleton):
    """get all bifurcation and crossing point

    Args:
        skeleton ([nd array]): [description]

    Returns:
        [list]: [description]
    """
   #TODO: check if we need to distinguish crossings and bifurcations
    intersections_structs = [np.array([
                         [1, 0, 1],
                         [0, 1, 0],
                         [0, 1, 0]]),
               np.array([[0, 1, 0],
                         [0, 1, 1],
                         [1, 0, 0]]),
               np.array([[0, 0, 1],
                         [1, 1, 0],
                         [0, 0, 1]]),
               np.array([[1, 0, 0],
                         [0, 1, 1],
                         [0, 1, 0]]),
               np.array([[0, 1, 0],
                         [0, 1, 0],
                         [1, 0, 1]]),
               np.array([[0, 0, 1],
                         [1, 1, 0],
                         [0, 1, 0]]),
               np.array([[1, 0, 0],
                         [0, 1, 1],
                         [1, 0, 0]]),
               np.array([[0, 1, 0],
                         [1, 1, 0],
                         [0, 0, 1]]),
               np.array([[1, 0, 0],
                         [0, 1, 0],
                         [1, 0, 1]]),                             
               np.array([[1, 0, 1],
                         [0, 1, 0],
                         [1, 0, 0]]),
               np.array([[1, 0, 1],
                         [0, 1, 0],
                         [0, 0, 1]]),                             
               np.array([[0, 0, 1],
                         [0, 1, 0],
                         [1, 0, 1]]),                            
               np.array([[0, 1, 0],
                         [1, 1, 0],
                         [0, 1, 0]]),
               np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 0, 0]]),                             
               np.array([[0, 1, 0],
                         [0, 1, 1],
                         [0, 1, 0]]),                            
               np.array([[0, 0, 0],
                         [1, 1, 1],
                         [0, 1, 0]]),                            
               np.array([[0, 0, 1],
                         [1, 1, 1],
                         [0, 0, 1]]),                             
               np.array([[0, 1, 0],
                         [0, 1, 0],
                         [1, 1, 1]]),                            
               np.array([[1, 0, 0],
                         [1, 1, 1],
                         [1, 0, 0]]),                            
               np.array([[1, 1, 1],
                         [0, 1, 0],
                         [0, 1, 0]]),                             
               np.array([[0, 1, 1],
                         [1, 1, 0],
                         [0, 1, 0]]),                            
               np.array([[0, 1, 0],
                         [1, 1, 0],
                         [0, 1, 1]]),                            
               np.array([[0, 1, 1],
                         [1, 1, 0],
                         [0, 1, 1]]),                             
               np.array([[1, 1, 0],
                         [0, 1, 1],
                         [1, 1, 0]]),                            
               np.array([[1, 0, 1],
                         [1, 1, 1],
                         [0, 1, 0]]),    
               np.array([[1, 0, 0],
                         [1, 1, 1],
                         [0, 1, 0]]),                             
               np.array([[1, 1, 0],
                         [0, 1, 1],
                         [0, 1, 0]]),                            
               np.array([[0, 1, 0],
                         [0, 1, 1],
                         [1, 1, 0]]),                             
               np.array([[0, 0, 1],
                         [1, 1, 1],
                         [0, 1, 0]]),                             
               np.array([[0, 1, 0],
                         [1, 1, 1],
                         [1, 0, 1]]),                            
               np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 0, 1]]),
               np.array([[0, 1, 0],
                         [1, 1, 1],
                         [1, 0, 0]]),                             
               np.array([[1, 0, 0],
                         [0, 1, 1],
                         [1, 1, 0]]),                            
               np.array([[1, 1, 0],
                         [0, 1, 1],
                         [1, 0, 0]]),                            
               np.array([[1, 0, 1],
                         [1, 1, 0],
                         [0, 1, 0]]),                             
               np.array([[1, 0, 1],
                         [0, 1, 1],
                         [0, 1, 0]]),                            
               np.array([[0, 1, 0],
                         [0, 1, 1],
                         [1, 0, 1]]),                            
               np.array([[0, 1, 0],
                         [1, 1, 0],
                         [1, 0, 1]]),                             
               np.array([[0, 0, 1],
                         [1, 1, 0],
                         [0, 1, 1]]),                            
               np.array([[0, 1, 1],
                         [1, 1, 0],
                         [0, 0, 1]]),
                            ]                         
                         
                            
                            
    image = skeleton.copy();
    image = image/255;
    intersections = list()

    for i, struct in enumerate(intersections_structs):
        hits = nd.binary_hit_or_miss(image, structure1=struct)

        if not np.any(hits):
            continue

        for y, x in zip(*np.where(hits)):
            intersections.append((x, y))
      # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections
    

def getSkeletonIntersection(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of

    Returns: 
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6 
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]
                       
                        ];
    image = skeleton.copy();
    image = image/255;
    intersections = list();
    for x in range(1,len(image)-1):
        for y in range(1,len(image[x])-1):
            # If we have a white pixel
            if image[x][y] == 1:
                nb = neighbours(x,y,image);
                valid = True;
                if nb in validIntersection:
                    intersections.append((y,x));
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2);
    # Remove duplicates
    intersections = list(set(intersections));
    return intersections;

def finding_landmark_vessels(widths: list, landmarks: list, skeleton: np.ndarray, skeleton_rgb: np.ndarray):
    vessels = []
    for l in range(0, len(widths)):
        cgray = skeleton.copy()
        crgb = skeleton_rgb.copy()
        radius = int(np.ceil(widths[l][1] + widths[l][2] * 1.5))
        x0 = landmarks[l][0]
        y0 = landmarks[l][1]
        points = []
        dy = x = y = 0
        crgb[x0, y0] = [0, 255, 0]
        for start in range(0, 2):
            for rad in range(-radius, radius + 1):
                dy = int(np.round(np.sqrt(np.power(radius, 2) - np.power(rad, 2))))
                for loop in range(0, 2):
                    if start == 0:
                        x = x0 + rad
                        if loop == 0:
                            y = y0 - dy
                        else:
                            y = y0 + dy
                    else:
                        y = y0 + rad
                        if loop == 0:
                            x = x0 - dy
                        else:
                            x = x0 + dy

                    acum = 0
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            if all(crgb[x + i, y + j] == [0, 0, 255]):
                                acum += 1

                    if cgray[x, y] == 255 and acum == 0:
                        crgb[x, y] = [0, 0, 255]
                        cgray[x - 1:x + 2, y - 1:y + 2] = 0
                        cgray[x, y] = 255
                        points.append([x, y])
                    elif acum == 0:
                        crgb[x, y] = [255, 0, 0]
                        block = cgray[x - 1:x + 2, y - 1:y + 2]
                        connected_components = cv2.connectedComponentsWithStats(block.astype(np.uint8), 8, cv2.CV_8U)
                        for k in range(1, connected_components[0]):
                            mask = connected_components[1] == k
                            indexes = np.column_stack(np.where(mask))
                            for e in range(0, len(indexes)):
                                ix = x + indexes[e][0] - 1
                                iy = y + indexes[e][1] - 1
                                if e == int(len(indexes) / 2):
                                    crgb[ix, iy] = [0, 0, 255]
                                    points.append((ix, iy))

        vessels.append(points)
    return vessels
# def skeletonizeHM(img):
#     h1 = np.array([[0, 0, 0],[0, 1, 0],[1, 1, 1]])
#     m1 = np.array([[1, 1, 1],[0, 0, 0],[0, 0, 0]])
#     h2 = np.array([[0, 0, 0],[1, 1, 0],[0, 1, 0]])
#     m2 = np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])
#     hit_list = []
#     miss_list = []
#     for k in range(4):
#         hit_list.append(np.rot90(h1, k))
#         hit_list.append(np.rot90(h2, k))
#         miss_list.append(np.rot90(m1, k))
#         miss_list.append(np.rot90(m2, k))
#     img = img.copy()
#     while True:
#         last = img
#         for hit, miss in zip(hit_list, miss_list):
#             hm = m.binary_hit_or_miss(img, hit, miss)
#             img = np.logical_and(img, np.logical_not(hm))
#         if np.all(img == last):
#             break
#     return img
# def getSkeletonIntersectionHitOrMiss(skeleton):
#     # Construct the structuring element
#     kernel = np.array((
#         [1, 1, 1],
#         [0, 1, -1],
#         [0, 1, -1]), dtype="int")
#     image = skeleton.copy()
#     image = image/255
#     intersections = list()
#     output_image = cv.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
    





def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()
    
    
def separate_labels(labels):
    # generate dummy image:
    # labels = np.zeros((100,100), dtype=np.int) # this does not work on floats
    # # adding two rectangles, similar to output of your label function
    # labels[10:20, 10:20] = 1
    # labels[40:50, 40:60] = 2

    props = regionprops_table(labels,None,('label','coords','bbox','area','centroid'))
    
    vessels = dict(zip(props['label'],props['coords']))
    # for prop in props:
    #     print(prop['label']) # individual properties can be accessed via square brackets
    #     cropped_shape = prop['filled_image'] # this gives you the content of the bounding box as an array of bool.
    #     cropped_shape = 1 * cropped_shape # convert to integer
    #     print(prop['coords'])
    #     # save image with your favourite imsave. Data conversion might be neccessary if you use cv2   
    #     plt.imshow(cropped_shape)
    #     plt.axis("off")
    #     plt.title("component Image")
    #     plt.show() 
    return props

def estimate_width(thresh):
    
    label_img = label(thresh)
    regions = regionprops(label_img)

    fig, ax = plt.subplots()
    ax.imshow(thresh, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    ax.axis((0, 600, 600, 0))
    plt.show()
def connected_component_label(skeleton, branch_points):
    
    # Getting the input image
    img = skeleton.copy()
    # Converting those pixels with values 1-127 to 0 and others to 1
   # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # Applying cv2.connectedComponents() 
    # # Showing Original Image
    # plt.imshow(img)
    # plt.axis("off")
    # # plt.title("Orginal true Image")
    # plt.show()
    for point in branch_points:
       # print("point ", point, point[0], point[1])
        #print("prima ", img[point[1],point[0]])
        img[point[1],point[0]] = 0
        # img[point[1] +1,point[0]] = 0
        # img[point[1] -1,point[0]] = 0
        # img[point[1] ,point[0]-1] = 0
        # img[point[1] ,point[0]+1] = 0
        # img[point[1]-1 ,point[0]-1] = 0
        # img[point[1]-1 ,point[0]+1] = 0
        # img[point[1]+1 ,point[0]+1] = 0
        # img[point[1]+1 ,point[0]-1] = 0
        
    output = cv2.connectedComponentsWithStats(img,4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    imshow_components(labels)
    vessels = separate_labels(labels)
    vessels = pd.DataFrame(vessels)  
    # mask = np.zeros(img.shape, dtype="uint8")
    # for i in range(1, numLabels):
    # 	# extract the connected component statistics for the current
	#     # label
    #     x = stats[i, cv2.CC_STAT_LEFT]
    #     y = stats[i, cv2.CC_STAT_TOP]
    #     w = stats[i, cv2.CC_STAT_WIDTH]
    #     h = stats[i, cv2.CC_STAT_HEIGHT]
    #     area = stats[i, cv2.CC_STAT_AREA]
    # # ensure the width, height, and area are all neither too small
	# # nor too big
    #     keepWidth = w > 5 and w < 50
    #     keepHeight = h > 45 and h < 65
    #     keepArea = area > 500 and area < 1500
    #     # ensure the connected component we are examining passes all
    #     # three tests
    #     if all((keepWidth, keepHeight, keepArea)):
    #         # construct a mask for the current connected component and
    #         # then take the bitwise OR with the mask
    #         print("[INFO] keeping connected component '{}'".format(i))
    #         componentMask = (labels == i).astype("uint8") * 255
    #         mask = cv2.bitwise_or(mask, componentMask)
    # cv2.imshow("Image", img)
    # cv2.imshow("Characters", mask)
    # cv2.waitKey(0)
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    im = np.zeros((1000,1000), dtype="uint8")
    
    coords = vessels['coords']
    coords = vessel_clean(coords)
    all_tort = 0
    # curve_lengths = list()
    # segments_tort = list()
    # chord_lenghts = list()
    # for coord in coords:
       
    #     #coords_array = np.array([coord], dtype=np.int32)
    #     curve_length = tortuosity._curve_length(coord)
    #     chord_length = tortuosity._chord_length(coord)
    #     print(chord_length)
    #     tortuosity_measure = chord_length/(curve_length)
    #     print("tortuosity: ",tortuosity_measure)
        
    #     curve_lengths.append(curve_length)
    #     chord_lenghts.append(chord_length)
    #     segments_tort.append(tortuosity_measure)
        
        
        # cv2.polylines(im, coords_array, 1, 255)
        # cv2.imshow('image',im)
        
    
    # all_tort = np.average(segments_tort, weights=curve_lengths)
    # print("weighted tortuosity: " ,all_tort)
    # # Showing Original Image
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.title("Orginal Image")
    # plt.show()
    
    #Showing Image after Component Labeling
    return vessels
    
def vessel_clean(vessels):
    clean_vessels = list()
    for vessel in vessels:
        if len(vessel) > 10:
            clean_vessels.append(vessel)
    return clean_vessels
            
             
def principal_boxes(skeleton: np.ndarray, landmarks: list, size: int):
    junct = landmarks.copy()
    bifurcations_coordinates = []
    crossings_coordinates = []
    while True:
        if junct:
            junct = boxes_auxiliary(skeleton, junct, size, bifurcations_coordinates, crossings_coordinates)
        else:
            break

    return bifurcations_coordinates, crossings_coordinates             

def vessel_number(vessels: list, landmarks: list, skeleton_rgb: np.ndarray):
    skeleton = skeleton_rgb.copy()
    length = len(vessels)
    final_landmarks = []
    for v in range(0, length):
        if len(vessels[v]) == 3:
            skeleton[landmarks[v][0], landmarks[v][1]] = [0, 0, 255]
            final_landmarks.append(landmarks[v])
        elif len(vessels[v]) >= 4:
            skeleton[landmarks[v][0], landmarks[v][1]] = [255, 0, 0]
            final_landmarks.append(landmarks[v])

    return skeleton, final_landmarks

def boxes_auxiliary(skeleton: np.ndarray, landmarks: list, size: int, bifurcations_coordinates: list, crossings_coordinates: list):
    x = landmarks[0][0]
    y = landmarks[0][1]
    num_bifurcations = 0
    num_crossings = 0
    box = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            box.append([x + i, y + j])
            if all(skeleton[x + i, y + j] == [0, 0, 255]):
                num_bifurcations += 1
            elif all(skeleton[x + i, y + j] == [255, 0, 0]):
                num_crossings += 1

    landmarks = [val for val in landmarks if val not in box]

    if num_bifurcations > num_crossings:
        bifurcations_coordinates.append([y - 3 - size, x - 3 - size, y + 3 - size, x + 3 - size])
    else:
        crossings_coordinates.append([y - 3 - size, x - 3 - size, y + 3 + size, x + 3 + size])

    return landmarks