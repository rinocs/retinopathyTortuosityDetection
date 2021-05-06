import cv2
import numpy as np
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize,medial_axis
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from utils.imageproc import image_util
from utils.vessels import vessels_util

def extract_bv(image):		
	b,green_fundus,r = cv2.split(image)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	contrast_enhanced_green_fundus = clahe.apply(green_fundus)

	# applying alternate sequential filtering (3 times closing opening)
	r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
	f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
	f5 = clahe.apply(f4)		

	# removing very small contours through area parameter noise removal
	ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
	mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
	contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) <= 200:
			cv2.drawContours(mask, [cnt], -1, 0, -1)			
	im = cv2.bitwise_and(f5, f5, mask=mask)
	ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)			
	newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	

	# removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
	#vessels and also in an interval of area
	fundus_eroded = cv2.bitwise_not(newfin)	
	xmask = np.ones(fundus_eroded.shape[:2], dtype="uint8") * 255
	xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
	for cnt in xcontours:
		shape = "unidentified"
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   				
		if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
			shape = "circle"	
		else:
			shape = "veins"
		if(shape=="circle"):
			cv2.drawContours(xmask, [cnt], -1, 0, -1)	
	
	finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
	blood_vessels = cv2.bitwise_not(finimage)
	return blood_vessels	

img = cv2.imread('myProject/sample/Vena-93.tif')
#functions.connected_component_label('myProject/sample/vessel1.png')
if img is None:
    print('Error loading image')
    exit()  


bv = extract_bv(img)    
cv2.imshow("Frame", bv)
binary = bv >   filters.threshold_otsu(bv)
bv = cv2.bitwise_not(bv)
ret, bw_img = cv2. threshold(bv,127,1,cv2.THRESH_BINARY)
print(np.unique(bv))
print(np.unique(bw_img))

cv2.imshow("Frame", bw_img)
skeleton = skeletonize(bw_img)

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
    



cv2.imshow("Frame", skeleton)
#cv2.imshow("Frame1", img)

cv2.waitKey(0)
cv2.destroyAllWindows()