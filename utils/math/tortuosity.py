import math
import numpy as np
from utils.math import math as m
from scipy import interpolate
from matplotlib import pyplot as plt
import cv2
from numpy.core.fromnumeric import mean

def two_point_dist(pt1, pt2):
    return np.linalg.norm([x2 - x1 for x2, x1 in zip(pt1, pt2)])    
def avg_pts(pts):
    dims = len(pts[0])
    avg_pt = []
    for dim in range(dims):
        avg_pt.append(np.mean([pt[dim] for pt in pts]).astype(int))
    return avg_pt

def _distance_2p(x1, y1, x2, y2):
    """
    calculates the distance between two given points
    :param x1: starting x value
    :param y1: starting y value
    :param x2: ending x value
    :param y2: ending y value
    :return: the distance between [x1, y1] -> [x2, y2]
    """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _curve_length(coord):
    """
    calculates the length(distance) of the given curve, iterating from point to point.
    :param coord: array [[x1,y1],....,[xn,yn]]the x component of the curve , the y component of the curve
    :return: the curve length
    """
    distance = 0
    for i in range(0, len(coord) - 1):
        distance += _distance_2p(coord[i][0], coord[i][1], coord[i+1][0], coord[i+1][1])
    return distance


def _chord_length(coord):
    """
    distance between starting and end point of the given curve

    :param coord: array [[x1,y1],....,[xn,yn]]the x component of the curve , the y component of the curve
    :return: the chord length of the given curve
    """
    return _distance_2p(coord[0][0], coord[0][1], coord[len(coord) - 1][0], coord[len(coord) - 1][1])

def distance_measure_tortuosity(coord):
    """
    Distance measure tortuosity defined in:
    William E Hart, Michael Goldbaum, Brad Côté, Paul Kube, and Mark R Nelson. Measurement and
    classification of retinal vascular tortuosity. International journal of medical informatics,
    53(2):239–252, 1999.

    :param coord: the list of pair of coords (x,y)
    :return: the arc-chord tortuosity measure
    """
    if len(coord) < 2:
        raise ValueError("Given curve must have at least 2 elements")

    return _curve_length(coord) / _chord_length(coord)

def _detect_inflection_points(x, y):
    """
    This method detects the inflection points of a given curve y=f(x) by applying a convolution to
    the y values and checking for changes in the sign of this convolution, each sign change is
    interpreted as an inflection point.
    It will ignore the first and last 2 pixels.
    :param x: the x values of the curve
    :param y: the y values of the curve
    :return: the array position in x of the inflection points.
    """
    cf = np.convolve(y, [1, -1])
    inflection_points = []
    for iterator in range(2, len(x)):
        if np.sign(cf[iterator]) != np.sign(cf[iterator - 1]):
            inflection_points.append(iterator - 1)
    return inflection_points

def distance_inflection_count_tortuosity(x, y):
    """
    Calculates the tortuosity by using arc-chord ratio multiplied by the curve inflection count
    plus 1

    :param x: the list of x points of the curve
    :param y: the list of y points of the curve
    :return: the inflection count tortuosity
    """
    coords = [[x,y] for x,y in zip(x,y)]
    return distance_measure_tortuosity(coords) * (len(_detect_inflection_points(x, y)) + 1)

def _curve_to_image(x, y):
    # get the maximum and minimum x and y values
    mm_values = np.empty([2, 2], dtype=np.int64)
    mm_values[0, :] = 99999999999999
    mm_values[1, :] = -99999999999999
    for i in range(0, len(x)):
        if x[i] < mm_values[0, 0]:
            mm_values[0, 0] = x[i]
        if x[i] > mm_values[1, 0]:
            mm_values[1, 0] = x[i]
        if y[i] < mm_values[0, 1]:
            mm_values[0, 1] = y[i]
        if y[i] > mm_values[1, 1]:
            mm_values[1, 1] = y[i]
    distance_x = mm_values[1, 0] - mm_values[0, 0]
    distance_y = mm_values[1, 1] - mm_values[0, 1]
    # calculate which square with side 2^n of size will contain the line
    image_dim = 2
    while image_dim < distance_x or image_dim < distance_y:
        image_dim *= 2
    image_dim *= 2
    # values to center the
    padding_x = (mm_values[1, 0] - mm_values[0, 0]) // 2
    padding_y = (mm_values[1, 1] - mm_values[0, 1]) // 2

    image_curve = np.full([image_dim, image_dim], False)

    for i in range(0, len(x)):
        x[i] = x[i] - mm_values[0, 0]
        y[i] = y[i] - mm_values[0, 1]

    for i in range(0, len(x)):       
        image_curve[x[i], y[i]] = True
        
    return image_curve

def linear_regression_tortuosity(x, y, sampling_size=6, retry=True):
    """
    This method calculates a tortuosity measure by estimating a line that start and ends with the
    first and last points of the given curve, then samples a number of pixels from the given line
    and calculates its determination coefficient, if this value is closer to 1, then the given
    curve is similar to a line.

    This method assumes that the given parameter is a sorted list.

    Returns the determination coefficient for the given curve
    :param x: the x component of the curve
    :param y: the y component of the curve
    :param sampling_size: how many pixels
    :param retry: if regression fails due to a zero division, try again by inverting x and y
    :return: the coefficient of determination of the curve.
    """
    if len(x) < 4:
        raise ValueError("Given curve must have more than 4 elements")
    try:
        x.sort()
        y.sort()
        min_point_x = x[0]
        min_point_y = y[0]

        slope = (y[len(y) - 1] - min_point_y)/(x[len(x) - 1] - min_point_x)

        y_intercept = min_point_y - slope*min_point_x

        sample_distance = max(round(len(x) / sampling_size), 1)

        # linear regression function
        def f_y(x1):
            return x1 * slope + y_intercept

        # calculate y_average
        y_average = 0
        item_count = 0
        for i in range(1, len(x) - 1, sample_distance):
            y_average += y[i]
            item_count += 1
        y_average /= item_count

        # calculate determination coefficient
        top_sum = 0
        bottom_sum = 0
        for i in range(1, len(x) - 1, sample_distance):
            top_sum += (f_y(x[i]) - y_average) ** 2
            bottom_sum += (y[i] - y_average) ** 2

        r_2 = top_sum / bottom_sum
    except ZeroDivisionError:
        if retry:
            #  try inverting x and y
            r_2 = linear_regression_tortuosity(y, x, retry=False)
        else:
            r_2 = 1  # mark not applicable vessels as not tortuous?
    if math.isnan(r_2):  # pragma: no cover
        r_2 = 0
    return r_2



def tortuosity_density(x, y):
    """
    Defined in "A Novel Method for the Automatic Grading of Retinal Vessel Tortuosity" by Grisan et al.
    DOI: 10.1109/IEMBS.2003.1279902

    :param x: the x points of the curve
    :param y: the y points of the curve
    :return: tortuosity density measure
    """
    inflection_points = _detect_inflection_points(x, y)
    n = len(inflection_points)
    coord_ext = [ [x,y] for x,y in zip(x,y)]
    if not n:
        return 0
    starting_position = 0
    sum_segments = 0
    # we process the curve dividing it on its inflection points
    for in_point in inflection_points:
        segment_x = x[starting_position:in_point]
        segment_y = y[starting_position:in_point]
        coords = [ [x,y] for x,y in zip(segment_x,segment_y)]
        chord = _chord_length(coords)
        if chord:
            sum_segments += _curve_length(coords) / _chord_length(coords) - 1
        starting_position = in_point

    return (n - 1)/n + (1/_curve_length(coord_ext))*sum_segments


def squared_curvature_tortuosity(x, y):
    """
    See Measurement and classification of retinal vascular tortuosity by Hart et al.
    DOI: 10.1016/S1386-5056(98)00163-4
    :param x: the x values of the curve
    :param y: the y values of the curve
    :return: the squared curvature tortuosity of the given curve
    """
    curvatures = []
    x_values = range(1, len(x)-1)
    for i in x_values:
        x_1 = m.derivative1_centered_h1(i, x)
        x_2 = m.derivative2_centered_h1(i, x)
        y_1 = m.derivative1_centered_h1(i, y)
        y_2 = m.derivative2_centered_h1(i, y)
        curvatures.append((x_1*y_2 - x_2*y_1)/(y_1**2 + x_1**2)**1.5)
    return abs(np.trapz(curvatures, x_values))


def smooth_tortuosity_cubic(x, y, name):
    """
    TODO
    :param x: the list of x points of the curve
    :param y: the list of y points of the curve

    :return:
    """
    spline = interpolate.CubicSpline(x, y)
    points = [(x,y) for x,y in zip(x,y)]
    data = np.array(points)
    tck,u = interpolate.splprep(data.transpose(), s=0)
    unew = np.arange(0, 1.01, 0.01)
    out = interpolate.splev(unew, tck)

    fig = plt.figure()
    fig.suptitle(name, fontsize=20)
    plt.plot(out[0], out[1], color='orange')
    plt.plot(data[:,0], data[:,1], 'ob')
    plt.show()
    return spline(x[0])

def tortuosity_measure(img):
    k = 0

    contours,hier = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    torts = []
    arc_torts = []
    for cnt in contours:
        output = cv2.drawContours(np.zeros(img.shape,dtype = np.uint8),[cnt],-1,(255,255,255),1)
        count = np.count_nonzero(output)
        if count>=30:            

            x,y,h,w = cv2.boundingRect(cnt)

            roi = output[y:y+w, x:x+h]

            r,c = roi.shape
            back = np.zeros((r+2,c+2),dtype=np.uint8)
            back[1:r+1,1:c+1] = roi

            iroi = back.copy()
            # cv2.imshow('iroi',iroi)
            # cv2.waitKey(0)
            order = order_points(iroi)

            

            inflections = contour_inflections(iroi)
            clean = []
            [clean.append(x) for x in inflections if x not in clean]

            angles = get_angles(inflections)
            c_angles = [incom for incom in angles if str(incom) != 'nan']
            torts.append(mean(c_angles))

            arcbased = arclength(order,iroi)
            arc_torts.append(arcbased)

    tortuos = (180/mean(torts))

    return mean(torts), tortuos, mean(arc_torts) 


def arclength(order,img):
    h = distance(order[0],order[-1])

    arclength = 0
    n = len(order)
    for i in range(n-1):
        arclength+= distance(order[i],order[i+1])
    
    t = arclength/h

    return t
def order_points(img):
    global r,c,visit
    r,c = img.shape
    visit = np.zeros((r,c))
    x,y = start_point(img,r,c)

    flag = True
    points = [[x,y]]

    while(flag):
        x,y = dfs(img,x,y)
        if x!=-1 and y!=-1:
            points.append([x,y])
        else:
            flag=False

    return points

def getinflections(img,points):
    kernels = [np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32),
                np.array([[0,1,0],[0,1,0],[0,1,0]],dtype=np.float32),
                np.array([[0,0,1],[0,1,0],[1,0,0]],dtype=np.float32),
                np.array([[0,0,0],[1,1,1],[0,0,0]],dtype=np.float32)]

    pts = []
    img[img==255] = 1

    for i,j in points:
        roi = img[i-1:i+2, j-1:j+2]
        flag = 0
        for k in kernels:
            p = np.sum(k)
            r = np.sum(np.multiply(roi,k))
            if(r==p):
                flag=1
                break

        if flag==0:
            pts.append([i,j])

    return pts
def compute_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # if(angle == 0 or str(angle) == 'nan'):
    #     print('found : '+str(angle))
    #     print(a,b,c)
    #     print()

    return np.degrees(angle)

def get_angles(inflects):
    if(len(inflects)<3):
        return [180]

    n = len(inflects)
    angles = []
    for i in range(n-2):
        angles.append(compute_angle(inflects[i],inflects[i+1],inflects[i+2]))

    return angles

def contour_inflections(img):
    cnts = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)[-2]

    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    arclen = cv2.arcLength(cnt, True)

    epsilon = arclen * 0.0075
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # canvas = cv2.cvtColor(np.zeros(img.shape,np.uint8),cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(canvas, [approx], -1, (0,0,255), 1)
    pts = []
    for pt in approx:
        i,j = pt[0][1], pt[0][0]
        if([i,j] not in pts):
            pts.append([i,j])
            # canvas[i][j] = [0,255,0]
    
    # cv2.imshow('smooth',canvas)
    # cv2.waitKey(0)
    
    return pts
def start_point(img,r,c):
    im = img.copy()

    im[im==255] = 1
    dummy = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
    for i in range(1,r-1):
        for j in range(1,c-1):
            if im[i][j]==1:
                roi = im[i-1:i+2, j-1:j+2]
                p = np.sum(roi)
                if(p==2):
                    dummy[i][j] = [0,0,255]
                    return (i,j)
                
def dfs(mat,i,j):
    
    visit[i][j] = 1

    if (j!=0 and visit[i][j-1]==0 and mat[i][j-1]!=0):
        return (i,j-1)

    if (j+1<c and visit[i][j+1]==0 and mat[i][j+1]!=0):
        return (i,j+1)

    if (i-1>=0 and visit[i-1][j]==0 and mat[i-1][j]!=0):
        return (i-1,j)
    
    if (i+1<r and  visit[i+1][j]==0 and mat[i+1][j]!=0):
        return (i+1,j)

    if (i-1>=0 and j-1>=0 and visit[i-1][j-1]==0 and mat[i-1][j-1]!=0):
        return (i-1,j-1)
        

    if (i-1>=0 and j+1<c and visit[i-1][j+1]==0 and mat[i-1][j+1]!=0):
        return (i-1,j+1)

    if (i+1<r and j-1>=0 and visit[i+1][j-1]==0 and mat[i+1][j-1]!=0):
        return (i+1,j-1)
    if (i+1<r and j+1<c and visit[i+1][j+1]==0 and mat[i+1][j+1]!=0):
        return (i+1,j+1)

    return (-1,-1)           


def distance(a,b):
    res = (b[0]-a[0])**2 + (b[1]-a[1])**2
    res = math.sqrt(res)

    return res