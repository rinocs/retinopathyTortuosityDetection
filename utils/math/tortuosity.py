

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
    print("checking:")
    print(coord[0][0] )
    print(" ")
    print(coord[0][1])
    print(coord[len(coord) - 1][0])
    print(coord[len(coord) - 1][1])
    print(_distance_2p(coord[0][0], coord[0][1], coord[len(coord) - 1][0], coord[len(coord) - 1][1]))
    print("checking_end_table")
    return _distance_2p(coord[0][0], coord[0][1], coord[len(coord) - 1][0], coord[len(coord) - 1][1])


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