import cv2
import numpy as np
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize,medial_axis
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from utils.imageproc import image_util
from utils.vessels import vessels_util
from utils.math import tortuosity
import math
from plantcv import plantcv as pcv

img = cv2.imread('sample/CHASE/train/label/hImage_01L_1stHO.png')
img1 = cv2.imread('sample/CHASE/train/image/hImage_01L.jpg')
img2 = cv2.imread('sample/CHASE/train/image/hImage_01L.jpg',0)
#functions.connected_component_label('myProject/sample/vessel1.png')
if img is None:
    print('Error loading image')
    exit()  
resized = image_util.image_resize(img)    

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

#ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY )
cv2.imshow("thresh1", thresh)
cv2.waitKey(0)

#ret,frame = cv2.threshold(img,127,255,0)
binary = thresh > filters.threshold_otsu(thresh)
np.unique(binary)
# binary = image_util.get_uint_image(binary)


#thin = image_util.thinning_zhang_suen(img)
imgGray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
ret,threshold_img = cv2.threshold(imgGray,127,255,cv2.THRESH_BINARY)
# frame = image_util.get_uint_image(frame)

##### SKELETON 
skel, distance = medial_axis(binary, return_distance=True)

skeleton = skeletonize(binary)
# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel
cv2.imshow("skel", dist_on_skel)
cv2.waitKey(0)
skeleton = image_util.get_uint_image(skeleton)

skeleton_rgb = image_util.bin_to_bgr_(skeleton)
#branch_locations = getSkeletonIntersection(skeleton)
branch_locations = vessels_util.getIntersections(skeleton)
branch_locations2 = vessels_util.getIntersections(dist_on_skel)
end_points = vessels_util.getEndPoints(dist_on_skel)

all_points = np.array( branch_locations)

vessels_data = vessels_util.connected_component_label(skeleton.copy(), branch_locations)
vessels_coord = vessels_data["coords"]

v_width = vessels_util.vessel_width(thresh, vessels_coord)

active_neigh = []
for el in vessels_coord[0]:
    active_neigh.append(image_util.active_neihbours(el[0], el[1], thresh))




# # Create Window
# source_window = 'Source'
# cv2.namedWindow(source_window)
# cv2.imshow(source_window, thresh)
# max_thresh = 255
# thresh_val = 100 # initial threshold
# cv2.createTrackbar('Canny Thresh:', source_window, thresh_val, max_thresh, image_util.thresh_callback)
# image_util.thresh_callback(thresh_val,skeleton)
# cv2.waitKey()

# mean_torts, tortuos, mean_arc_torts = tortuosity.tortuosity_measure(imgGray)
# print("tortuosity111:" + tortuos)


# for points in branch_locations:
    
#     if not points:
#           print("List is empty")
#           continue 
#     #cv2.drawContours(skeleton, np.array([points]), 0, (255,0,0), 2)
#     cv2.circle(skeleton,tuple(points) , 2, (255, 255, 0), 5)
# #         cv2.circle(img,point , 2, (255, 0, 0), 5)
    
print(thresh[17][379])
skel2 = skeleton.copy()
for points in branch_locations2:
        
    if not points:
          print("List is empty")
          continue 
    #cv2.drawContours(skeleton, np.array([points]), 0, (255,0,0), 2)
    cv2.circle(skel2,tuple(points) , 2, (255, 255, 0), 5)
#         cv2.circle(img,point , 2, (255, 0, 0), 5)
    
    
mask = np.zeros_like(gray)
skeleton_contours_img = skeleton.copy()
skeleton_contours_img = cv2.cvtColor(skeleton_contours_img, cv2.COLOR_GRAY2BGR) 
_ ,contours, hierarchy  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[1]
res_img = gray.copy()
size = np.size(gray)
kernel = np.ones((2,2), np.uint8)
_,skeleton_contours, _ = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
largest_skeleton_contour = max(skeleton_contours, key=cv2.contourArea)
cv2.drawContours(skeleton_contours_img, contours, -1, (0,255,0), 1)

cv2.imshow("skel_contours", skeleton_contours_img)
cv2.waitKey(0)
skel_rows,skel_cols = np.where(skeleton > 0)
skel_points = [(x,y) for x,y in zip(skel_rows,skel_cols)]
# for point in skel_points:
#     gray[point[0]][point[1]] = 0

# cv2.imshow("gray_withoutà_skeleton", gray)

skeleton_1 = pcv.morphology.skeletonize(mask=thresh)
cv2.imshow("skeleton_1", skeleton_1)
cv2.waitKey(0)

pcv.params.line_thickness = 3

# Prune the skeleton  

# Inputs:
#   skel_img = Skeletonized image
#   size     = Pieces of skeleton smaller than `size` should get removed. (Optional) Default `size=0`. 
#   mask     = Binary mask for debugging (optional). If provided, debug images will be overlaid on the mask.
pruned, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=70, mask=thresh)

# pruned, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton_1, size=70, mask=mask)

branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=skeleton, mask=thresh, label="default")
tip_pts_mask = pcv.morphology.find_tips(skel_img=skeleton, mask=thresh, label="default")
leaf_obj, stem_obj = pcv.morphology.segment_sort(skel_img=skeleton, 
                                                     objects=edge_objects,
                                                     mask=thresh)
# all_objects = leaf_obj + stem_obj
cv2.imshow("seg_img", seg_img)
cv2.waitKey(0)

segmented_img, labeled_img = pcv.morphology.segment_id(skel_img=skeleton,
                                                           objects=leaf_obj,
                                                           mask=thresh)
cv2.imshow("labeled_img", labeled_img)
cv2.imshow("segmented_img", segmented_img)
cv2.waitKey(0)

labeled_img1  = pcv.morphology.segment_path_length(segmented_img=segmented_img, 
                                                      objects=leaf_obj, label="default")
cv2.imshow("labeled_img1", labeled_img1)
cv2.waitKey(0)

labeled_img2 = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, 
                                                          objects=leaf_obj, label="default")
cv2.imshow("labeled_img2", labeled_img2)
cv2.waitKey(0)

labeled_img3 = pcv.morphology.segment_curvature(segmented_img=segmented_img, 
                                                   objects=leaf_obj, label="default")

cv2.imshow("labeled_img3", labeled_img3)
cv2.waitKey(0)

pcv.outputs.save_results(filename="leaf.json")



segmented_img, labeled_img = pcv.morphology.segment_id(skel_img=skeleton,
                                                           objects=stem_obj,
                                                           mask=thresh)
cv2.imshow("labeled_img", labeled_img)
cv2.imshow("segmented_img", segmented_img)
cv2.waitKey(0)

labeled_img1  = pcv.morphology.segment_path_length(segmented_img=segmented_img, 
                                                      objects=stem_obj, label="default")
cv2.imshow("labeled_img1", labeled_img1)
cv2.waitKey(0)

# labeled_img2 = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, 
#                                                           objects=stem_obj, label="default")
# cv2.imshow("labeled_img2", labeled_img2)
# cv2.waitKey(0)

# labeled_img3 = pcv.morphology.segment_curvature(segmented_img=segmented_img, 
#                                                    objects=stem_obj, label="default")

# cv2.imshow("labeled_img3", labeled_img3)
# cv2.waitKey(0)

pcv.outputs.save_results(filename="stem.json")
# skel_points = np.column_stack(np.where(skeleton>0)) 

# # Extend the skeleton past the edges of the banana
# x,y = zip(*skel_points)
# z = np.polyfit(x,y,7)
# f = np.poly1d(z)
# x_new = np.linspace(0, img.shape[1],300)
# y_new = f(x_new)
# extension = list(zip(x_new, y_new))
# imag = gray.copy()
# for point in range(len(extension)-1):
#     a = tuple(np.array(extension[point], int))
#     b = tuple(np.array(extension[point+1], int))
#     cv2.line(imag, a, b, (0,255,0), 1)
#     cv2.line(mask, a, b, 255, 1)   
# mask_px = np.count_nonzero(mask)


# Find the distance between points in the contour of the banana
# Only look at distances that cross the mid line
def is_collision(mask_px, mask, a, b):
    temp_image = mask.copy()
    cv2.line(temp_image, a, b, 0, 2)
    new_total = np.count_nonzero(temp_image)
    if new_total != mask_px: return True
    else: return False

def distance(a,b): return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# distances = []
# for point_a in cnt[:int(len(cnt)/2)]:
#     temp_distance = 0
#     close_distance = imag.shape[0] * imag.shape[1]
#     close_points = (0,0),(0,0)
#     for point_b in cnt:
#         A, B = tuple(point_a[0]), tuple(point_b[0])
#         dist = distance(tuple(point_a[0]), tuple(point_b[0]))
#         if is_collision(mask_px, mask, A, B):
#             if dist < close_distance:
#                 close_points = A, B
#                 close_distance = dist
#     cv2.line(imag, close_points[0], close_points[1], (234,234,123), 1)
#     distances.append((close_distance, close_points))
#     cv2.imshow('img',imag)
#     cv2.waitKey(1)    
    
# max_thickness = max(distances)
# a, b = max_thickness[1][0], max_thickness[1][1]
# cv2.line(imag, a, b, (0,255,0), 4)
# print("maximum thickness = ", max_thickness[0])


# cv2.drawContours(res_img, contours, -1, (0,255,75), 2)

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(dist_on_skel, cmap='magma')
ax[1].contour(gray, [0.5], colors='w')
ax[1].set_title('medial_axis')
ax[1].axis('off')

ax[2].imshow(skeleton, cmap=plt.cm.gray)
ax[2].set_title('skeletonize')
ax[2].axis('off')    

ax[3].imshow(resized, cmap=plt.cm.gray)
ax[3].set_title('crossover')
ax[3].axis('off')    

fig.tight_layout()
plt.show()

cv2.imshow("prova", img)
cv2.imshow("prova1", res_img)
cv2.imshow("Frame2", skel2)
cv2.imshow("Frame3", gray)
cv2.imshow("Frame4", dist_on_skel)

cv2.waitKey(0)
cv2.destroyAllWindows()