import cv2
import numpy as np
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize,medial_axis
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from utils.imageproc import image_util
from utils.vessels import vessels_util


img = cv2.imread('myProject/sample/vessel1.png')
#functions.connected_component_label('myProject/sample/vessel1.png')
if img is None:
    print('Error loading image')
    exit()  
resized = image_util.image_resize(img)    
frame = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret, frame = cv2.threshold(frame, 127, 1, cv2.THRESH_BINARY)
#ret,frame = cv2.threshold(img,127,255,0)
#binary = frame > filters.threshold_otsu(frame)
#np.unique(binary)

skel, distance = medial_axis(frame, return_distance=True)

skeleton = skeletonize(frame)
# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel


print(np.any(skeleton[:, :] == 1 ))
colour_frame = img

rows = frame.shape[0]
cols = frame.shape[1]
skeleton = image_util.get_uint_image(skeleton)
skeleton_rgb = image_util.bin_to_bgr_(skeleton)
#branch_locations = getSkeletonIntersection(skeleton)
branch_locations = vessels_util.getIntersections(skeleton)
end_points = vessels_util.getEndPoints(skeleton)

all_points = np.array( branch_locations)

plt.scatter(all_points[:,0], all_points[:,1])
plt.show()
skeleton_copy = skeleton.copy()
vessels_util.connected_component_label(skeleton_copy, branch_locations)

#v_width = functions.vessel_width(frame,branch_locations)
#vessels = functions.finding_landmark_vessels(v_width, branch_locations, skeleton, skeleton_rgb)

for points in branch_locations:
    
    if not points:
          print("List is empty")
          continue 
    #cv2.drawContours(skeleton, np.array([points]), 0, (255,0,0), 2)
    cv2.circle(skeleton,tuple(points) , 2, (255, 255, 0), 5)
#         cv2.circle(img,point , 2, (255, 0, 0), 5)
    
fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(frame, cmap=plt.cm.gray)
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(dist_on_skel, cmap='magma')
ax[1].contour(frame, [0.5], colors='w')
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
cv2.imshow("Frame", skeleton)
#cv2.imshow("Frame1", img)

cv2.waitKey(0)
cv2.destroyAllWindows()