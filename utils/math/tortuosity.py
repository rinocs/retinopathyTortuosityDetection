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

    image_curve = np.full([image_dim, image_dim], 0)

    for i in range(0, len(x)):
        x[i] = x[i] - mm_values[0, 0]
        y[i] = y[i] - mm_values[0, 1]

    for i in range(0, len(x)):       
        image_curve[x[i],y[i]] = 255
        
    image_curve = np.asarray(image_curve, dtype="uint8")
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

# % ========================================= PARAMETER EXPLANATION ==========================================
# % VTI: vessel tortuosity index.
# % sd: standard deviation of the angels between lines tangent to every pixel along the centerline.
# % mean_dm: average distance measure between inflection points along the centerline.
# % num_inflection_pts: number of inflection points along the centerline.
# % num_critical_pts: number of critical points along the centerline.
# % len_arch: length of vessel (arch) which is number of centerline pixels.
# % len_cord: length of vessel chord which is the shortest path connecting vessel end points.
# % VTI = (len_arch * sd * num_critical_pts * (mean_dm)) / len_cord;

#  compute mean standard deviation of angels between lines tangent to each pixel along centerline and a reference axis
def sd_theta(x,y):
    """
    Compute standard deviation (SD) of angles between lines tangent to each pixel on the
    centerline and a reference-axis (i.e. x-axis) for a curve defined by x & y coordinates.

    Please cite the following paper if you use this code.
    Khansari, et al. "Method for quantitative assessment of retinal vessel tortuosity in % optical coherence 
    tomography angiography applied to sickle cell retinopathy." Biomedical optics express 8.8 (2017):3796-3806.


    Args:
        x ([array]): [the list of x points of the curve]
   
        y ([array]): [the list of y points of the curve]
    """
    
    #compute ratio of derivative of y over x for determining tangent lines at each point on the curve
    
    x = np.array(x)
    y = np.array(y)
    dy = np.divide(np.diff(y),np.diff(x))
    
    #slope of reference axis (i.e. x-axis)
    m1 = 0
    slope = np.zeros(len(x)-1)
    theta = np.zeros(len(x)-1)  

    #repeat for all pixels of the curve
    for k in range(0,len(x) -2):
        # tangent line to the curve
        # tan_line = np.multiply((x-x[k]),dy[k])+y[k]
        tan_line = ((x-x[k])*dy[k])+y[k]
        # slope of tangent line.
        coefficients = np.polyfit(x,tan_line,1)
        #save slopes in a vector
        slope[k] = coefficients[0]     
        m2 = coefficients[0]
        #compute angle between the tangent line and the x-axis (m1 = 0)
        angle = np.arctan((m1-m2)/(1+m1*m2))*(180/np.pi)
        # save angle in a vector
        theta[k] = angle
    #remove NaNs, if any
    theta_final = theta[~np.isnan(theta)]
    #SD of angles between tangent lines and x axis. Note that SD is divided by 100 to lie in range of 0 and 1
    
    SD = np.std(abs(theta_final))/100 
    return SD, slope

def mean_distance_measure(x,y):
    """[summary]
    Mean distance measure (DM) between points where the convexity of a curve changes. The curve or vessel 
    centerline is defined by x & y coordinates.

    Distance measure is the ratio of vessel length to its chord length. This can be used as a rough approximation 
    of tortuosity. However, this global estimation may not match human perception of tortuosity (Grisan, et al
    2008). In the current work, we used local distance measure between inflection points and showed that it better 
    matches with visual perception of tortuosity.

    Please cite the following paper if you use this code :)
    Khansari, et al. "Method for quantitative assessment of retinal vessel tortuosity in % optical coherence 
    tomography angiography applied to sickle cell retinopathy." Biomedical optics express 8.8 (2017):3796-3806.

    Args:
        x ([array]): [the list of x points of the curve]
        y ([array]): [the list of y points of the curve]
    """
    # index init
    idx_ifp = 0
    N= 0
    DM = []
  
    
    # curvature approx
    dx = np.diff(x)  # 1st derivative of xvalues
    dy = np.diff(y)  # 1st derivative of yvalues

    # dx2 = (dx[0:len(dx)-2]+ dx[1:])/2  # 2nd derivative of x values
    # dy2 = (dy[0:len(dy)-2]+ dy[1:])/2  # 2nd derivative of y values
    # dy2=dy/dx

    dx2=0.5*(dx[:-1]+dx[1:])
    dy2=0.5*(dy[:-1]+dy[1:])
    
    # remove the last element to match length of the 1st and 2nd derivatives to enable vector multiplication
    dx = dx[:-1]
    dy = dy[:-1]
    
    k = np.divide((((dx*dy2)-(dx2*dy))),((((dx)**2)+((dy)**2))**(3/2)))# curvature of the curve based on x and y coordinates
    k = k+np.finfo(float).eps# adding epsilon to avoid zero values. Due to discrete integral, inflection points can be very close to zero
    curvature = np.mean(abs(k));

    #Detecting points of changes in sign of the curvature (inflection points).
    # *** The DM between the 1st point on the curve and the first inflection point was computed separately. Similarly, DM between the last inflection
    # point and the end point of the vessel segment was computed separetly.
    
    for i in range(0, len(k) -1 ):
        previous = k[i]
        current = k[i+1]
        if previous*current < 0: # point off convexity change
            N = N+1 # count number of inflection point
            if N== 1:
                idx = i+1
                chord_len = np.sqrt((x[idx]-x[0])**2 + (y[idx]-y[0])**2) # chord length between the 1st point and the 1st inflection point
                
                coord = [(a,b) for a,b in zip(x[0:idx+1],y[0:idx+1]) ]
                arc_length = _curve_length(coord) #arc length between the 1st point and 1st inflection point
                dm = arc_length/chord_len
                DM.insert(idx_ifp,dm ) # DM between the 1st curve point and the 1st inflection point
                previous_pt = idx # record index of the inflection point
            elif N>1 : # compute DM for the 2nd and the rest of inflection point
                idx_ifp += 1
                idx = i+1  # index for saving value 
                chord_len = np.sqrt((x[idx]-x[previous_pt])**2 + (y[idx]- y[previous_pt])**2) #chord length between inflection points
                coord = [(a,b) for a,b in zip(x[previous_pt :idx+1],y[previous_pt:idx+1]) ] #
                arc_length = _curve_length(coord) # arc length between  inflection points
                DM.insert(idx_ifp,arc_length/chord_len)
                previous_pt = idx
    if N >= 1:
        idx_ifp = idx_ifp+1
        chord_len = np.sqrt((x[len(x)-1]-x[previous_pt])**2 + (y[len(y)-1]- y[previous_pt])**2) 
        coord = [(a,b) for a,b in zip(x[previous_pt:],y[previous_pt:]) ]
        arc_length = _curve_length(coord)
        DM.insert(idx_ifp, arc_length/chord_len)
        
    if N < 1:
        print(len(x))
        chord_len = np.sqrt((x[(len(x)-1)]-x[0])**2 +(y[(len(y)-1)]-y[0])**2 )
        coord = [(a,b) for a,b in zip(x,y) ]
        arc_length = _curve_length(coord)
        DM.insert(idx_ifp, arc_length/chord_len)
    if N >=1 :
        DM.pop(0)
        DM = DM[:-1]
        DM = np.array(DM)
        # DM = DM[~np.isnan(DM)]
        ipf =  len(DM)+1
    else:
        ipf =1 
        
    
    mean_dm = np.mean(DM)
    
    if np.isnan(mean_dm):
        mean_dm = 1
    return mean_dm , ipf, curvature,DM
    
def num_critical_points(x,y):
    
    """
    Determine number of critical points in a curve defined by x and y coordinates
    A critical point is a point on the curve where the derivative vanishes (either zero or doesn't exist).
    In such a point, there is a change in sign of the slope of tangent lines.

    Please cite the following paper if you use this code
    Khansari, et al. "Method for quantitative assessment of retinal vessel tortuosity in % optical coherence 
    tomography angiography applied to sickle cell retinopathy." Biomedical optics express 8.8 (2017):3796-3806.
        
    """
    N = 0
    x = np.array(x)
    y = np.array(y)
    dy = np.diff(y)/np.diff(x) # compute ratio of derivative of y over x to determine tangent lines at each point on the curve
    slope = np.zeros( len(x)-1) 
    ######## matlab i numeri piccoli li mette a 0 ecco il perché delal differenza tra il valore calcolato in  python e matlab: proviamo con una epsilon piccola
    # epsilon=0.00000000000000004 # MSE 0.062
    epsilon=0.00000000000000001 # MSE 0.0568
    # epsilon=0.000000000000000009 # MSE 0.069
    # epsilon=0.000000000000000005 # MSE 0.06901
    # epsilon=0.0000000000000000001 # MSE 0.06901
        
    # compute slope of tangent lines for every pixel along the curve
    for k in range(0, len(x)-1):
        tang = (x - x[k])*dy[k]+y[k] # tangent line to the curve
        coefficients = np.polyfit(x,tang,1) # slope of the tangent line
        slope[k] = coefficients[0] # save slope    
    slope[np.abs(slope) < epsilon] = 0
    for ii in range(0,len(slope)-1):
        previous = slope[ii]
        current = slope[ii+1]
        if previous*current < 0:
            N = N+1 # add 1 to the number of crtical points(twists)
    if N == 0:
        N=1
        
    return N
    
def vessel_tort_index(arch_len, sd,num_crit_points,mean_dm, chord_len):
    """
    VTI: vessel tortuosity index.
    sd: standard deviation of the angels between lines tangent to every pixel along the centerline.
    mean_dm: average distance measure between inflection points along the centerline.
    num_inflection_pts: number of inflection points along the centerline.
    num_critical_pts: number of critical points along the centerline.
    len_arch: length of vessel (arch) which is number of centerline pixels.
    len_cord: length of vessel chord which is the shortest path connecting vessel end points.
    VTI = (len_arch * sd * num_critical_pts * (mean_dm)) / len_cord;


    Args:
        arch_len ([type]): [description]
        sd ([type]): [description]
        num_crit_points ([type]): [description]
        mean_dm ([type]): [description]
        chord_len ([type]): [description]
    """
    arch_len = float(arch_len)
    sd = float(sd)
    num_crit_points = float(num_crit_points)
    mean_dm = float(mean_dm)
    chord_len = float(chord_len)
    
    
    return (arch_len*sd*num_crit_points*(mean_dm))/chord_len