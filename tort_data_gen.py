import cv2
import numpy as np
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize, disk,binary_dilation,binary_closing,medial_axis
from skimage.measure import label, regionprops
from skimage.morphology import erosion, dilation
from skimage.morphology import disk
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from utils.imageproc import image_util
from utils.vessels import vessels_util
from utils.math import tortuosity
import math
import csv
import mahotas as mh
import os 
import logging
from skan import draw
from skan import skeleton_to_csgraph, summarize, Skeleton
from utils.vessels import vessels_util


veinPath = "sample/tort/Reduced_Veins_Iso/"
arteryPath= "sample/tort/Reduced_Arteries_Iso/"
csvPath = "sample/tort/"


def plot_comparison(original, filtered, filter_name):
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(50, 50), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    plt.show()
    
def remove_small_branches(branch_loc, skeleton):
    """[summary]

    Args:
        branch_loc ([tuple]): [description]
        skeleton ([image]): [description]
    """    
    for point in branch_loc:
        skeleton[point[1],point[0]] = 0
        # skeleton[point[1] +1,point[0]] = 0
        # skeleton[point[1] -1,point[0]] = 0
        # skeleton[point[1] ,point[0]-1] = 0
        # skeleton[point[1] ,point[0]+1] = 0
        # skeleton[point[1]-1 ,point[0]-1] = 0
        # skeleton[point[1]-1 ,point[0]+1] = 0
        # skeleton[point[1]+1 ,point[0]+1] = 0
        # skeleton[point[1]+1 ,point[0]-1] = 0
    
    l = len(branch_loc)
    # skeleton = removing(skeleton, tuple(branch_loc))

    label_image, nregions = label(skeleton,return_num=True)
    print(nregions)
    plt.imshow(label_image)
    plt.show()
    coordinates_to_delete=[]
    for i in range(1,nregions+1):
        if howmanypixels(label_image==i)==1:
            label_image[label_image==i]*0    
        elif howmanypixels(label_image==i)<100:
            coordinates_to_delete.append(getcoordinates(label_image==i))
                
    # We remove the original pixels that correspond to those of a non-interest branch
    # We have to prun again to eliminate the 1 pixel remaining of
    skeleton2=removing(skeleton,coordinates_to_delete)
   
                    
    # Deleting non-interest regions
    # props=regionprops(label_image)
    
    # va = []
    # for prop in props:
    #     va.append(prop.area)
    # index = []    
    # va.sort()
    # for idx,val in enumerate(va) :
    #     if val > 50 :
    #         index.append(idx)    
    # if l > 2:
    #     va_sort = index[l:]
    # else :
    #     va_sort = index
    # v = np.arange(nregions)

    # # vkill = []
    # # for i,val in enumerate(va):
        
    # vkill = np.setdiff1d(v,va_sort)+1
    
    # for kkill in vkill:
    #     skeleton[label_image==kkill]=0
        
    for point in branch_loc:
        skeleton[point[1],point[0]] = 1
        # skeleton[point[1] +1,point[0]] = 1
        # skeleton[point[1] -1,point[0]] = 1
        # skeleton[point[1] ,point[0]-1] = 1
        # skeleton[point[1] ,point[0]+1] = 1
        # skeleton[point[1]-1 ,point[0]-1] = 1
        # skeleton[point[1]-1 ,point[0]+1] = 1
        # skeleton[point[1]+1 ,point[0]+1] = 1
        # skeleton[point[1]+1 ,point[0]-1] = 1
    return skeleton2
    
    
    
    
        
            

#### Function that finds the endpoints where the prunning will start
def endPoints(skel):
    endpoint1=np.array([[0, 0, 0], 
                         [0, 1, 0],
                         [2, 1, 2]])
    
    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])
    
    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])
    
    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])
    
    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])
    
    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])
    
    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])
    endpoint9=np.array([[0, 0, 0],      
                        [0, 1, 1],      
                        [1, 0, 1]])
    
    
    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    ep9=mh.morph.hitmiss(skel,endpoint9)
    ep = ep2+ep3+ep6+ep7+ep8+ep9
    return ep

#### Function that finds the endpoints where the prunning will start
def endPoints2(skel):
    endpoint1=np.array([[1, 0, 0], 
                         [1, 1, 0],
                         [2, 0, 0]])
    
    endpoint2=np.array([[0, 0, 1],
                        [0, 1, 1],
                        [0, 0, 2]])
    
    endpoint3=np.array([[0, 1, 0],
                        [1, 1, 0],
                        [2, 0, 0]])
    
    endpoint4=np.array([[0, 1, 0],
                        [0, 1, 1],
                        [0, 0, 2]])
    endpoint5=np.array([[0,0,0],
                        [1,1,0],
                        [0,1,0]])
        
    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    
    ep = ep1+ep2+ep3+ep4+ep5
    return ep

# Main prunning function
def pruning(skeleton, size):
    '''remove iteratively end points "size" 
       times from the skeleton
    '''
    for i in range(0, size):
        endpoints = endPoints(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton

def pruning2(skeleton, size):
    '''remove iteratively end points "size" 
       times from the skeleton
    '''
    for i in range(0, size):
        endpoints = endPoints2(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton

# Function that eliminates possible abnormalities in the upper part of the skeleton
def cleaningtop(branch):
    c=1
    for i in range(len(branch)):
            for j in range(len(branch)):
                if c<10:
                    if branch[i][j]==1:
                        branch[i][j]=0
                        c+=1
    return branch
def cleaningBottom(branch):
    c=1
    for i in reversed(range(len(branch))):
            for j in reversed(range(len(branch))):
                if c<10:
                    if branch[i][j]==1:
                        branch[i][j]=0
                        c+=1
    return branch

def howmanypixels(branch):
    l=0
    for i in range(len(branch)):
        for j in range(len(branch)):
            if branch[i][j]==1:
                l+=1
    return l

# Get the coordinates of the pixels belonging to each branch
def getcoordinates(branch):
    pixel_graph, coordinates, degrees = skeleton_to_csgraph(branch,unique_junctions=False)
    return coordinates


# Removing pixels of non-interest from the original image taking their position from the list coordinates_to_eliminate
def removing(image,coordinates):
    
    for e in coordinates:
        c=e[1:]
        for coords in c:
                image[int(coords[0]),int(coords[1])]=0
    return image

with open(csvPath+'veinArtTort.csv', mode='w+') as csv_file:
    fieldnames = ['image', 'curve_length', 'chord_length','sd_theta','num_inflection_pts','num_critical_points','curvature','VTI','distance_tort']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    
        
    for filename in os.listdir(veinPath):
       
        if filename.endswith(".tif") : 
            # print(os.path.join(trainPath, filename))
            # continue
            
            img = image_util.load_image(os.path.join(veinPath, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            
            # img = extract_bv(img)
            # cv2.imshow("extracted_bv",img)
            # cv2.waitKey(0)
            img = image_util.image_resize(img)   
            # plt.figure(figsize=(10,10))
            # plt.imshow(img,cmap='Greys_r')
            # plt.show()
            print(img.shape)
            h,w = img.shape[:2]
            #
            # Drop top and bottom area of image with black parts.
            img= img[60:h-60, :]
            h, w = img.shape[:2]
            plt.figure(figsize=(10,10))
            plt.imshow(img,cmap='Greys_r')
            plt.show()
            # Threshold image
            ret,th1 = cv2.threshold(img,99,255,cv2.THRESH_BINARY_INV)
            # th1 = cv2.bitwise_not(th1)
            
            # Morphological operation (EROSION) on binary image resulting from bin treshold
            selem1 = disk(5)
            selem = disk(3)
            eroded = erosion(th1, selem)
            dilated = dilation(eroded, selem)
            dilated = dilation(dilated, selem1)
            # dilated = erosion(dilated, selem1)
            plt.figure(figsize=(10,10))
            plt.imshow(dilated,cmap='Greys_r')
            plt.show()
            
            # get rid of thinner lines
            # kernel = np.ones((5,5),np.uint8)
            # kernel1 = np.ones((3,3), np.uint8) 
            # th1 = cv2.dilate(th1,kernel,iterations = 2)
            # th1 = cv2.dilate(th1, kernel1, iterations=3)   
            # Determine contour of all blobs found
            
                       
                       
               
            
            
            # bv = cv2.bitwise_not(th1)
            # bw_img = image_util.get_uint_image(th1)
            # bw_img = cv2.erode(bw_img, kernel1, iterations=2)   
            # bw_img = image_util.get_uint_image(bv)
            # cv2.imshow("Frame", bw_img*255)
            # cv2.waitKey(0)
            # Labeling process
            label_image, nregions = label(dilated,return_num=True)
            # Deleting non-interest regions
            props=regionprops(label_image)
            va = []
            for prop in props:
                va.append(prop.area)
            indkeep = np.array(np.argmax(va))
            v = np.arange(nregions)
            vkill = np.setdiff1d(v,indkeep)+1 
            I2 = dilated.copy()
            for kkill in vkill:
                I2[label_image==kkill]=0
                
                
            plt.figure(figsize=(10,10))
            plt.subplot(1, 2, 1)
            plt.imshow(I2,cmap='Greys_r')
            plt.subplot(1, 2, 2)
            plt.imshow(dilated,cmap="Greys_r")
            plt.show()
            
            
            
            # Making the contour more regular
            # selem1 = disk(3)
            selem = disk(1)
            # eroded = erosion(I2, selem)
            
            dilated = dilation(I2, selem)
            # dilated = dilation(eroded, selem)
            # eroded = erosion(I2, selem)
            # closed=binary_closing(eroded,selem)
            plot_comparison(I2,dilated,'dilate 2')
            
            
            _, contours0, hierarchy = cv2.findContours( dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.approxPolyDP(cnt, 4, True) for cnt in contours0[0]]
            
            # Skeletonization
            skeleton=skeletonize(dilated >0)
            # Ploting the skeleton
            plt.figure(figsize=(15,15))
            plt.suptitle("skeleton of closed")
            plt.imshow(skeleton,cmap='Greys_r')
            plt.show()
            
            
            skeleton=cleaningtop(skeleton)
            fig=plt.figure(figsize=(15,15))
            fig.suptitle("cleaningtop of skeleton")
            plt.imshow(skeleton,cmap='Greys_r')
            plt.show()
            
            # skeleton=cleaningBottom(skeleton)
            # fig=plt.figure(figsize=(15,15))
            # fig.suptitle("cleaningbottom of skeleton")
            # plt.imshow(skeleton,cmap='Greys_r')
            # plt.show()
            
            skeleton1 = pruning(skeleton,40)
            skeleton1 = pruning2(skeleton1,60)
            fig=plt.figure(figsize=(15,15))
            plt.imshow(skeleton1,cmap='Greys_r')
            fig.suptitle("pruning of skeleton")
            plt.show()
            
            
            skeleton2=skeletonize(skeleton1)
            plt.figure(figsize=(15,15))
            plt.imshow(skeleton2,cmap='Greys_r')
            plt.suptitle("skeleton of skeleton")
            plt.show()
            
            
            
            
            branched_image  = skeleton2.copy()
            
            for i in range(3):
                branch_to_image = branched_image.copy()
                branch_to_image = image_util.get_uint_image(branch_to_image)
                branch_locations = vessels_util.getIntersections(branched_image)
                if not branch_locations :
                    break
                branched_image = remove_small_branches(branch_locations, branched_image.copy())
                for points in branch_locations:
        
                    if not points:
                        print("List is empty")
                        continue 
                
                    cv2.circle(branch_to_image,tuple(points) , 2, (255, 255, 0), 5)        


                plt.figure(figsize=(10,10))
                plt.subplot(1, 2, 1)
                plt.imshow(branched_image,cmap='Greys_r')
                plt.subplot(1, 2, 2)
                plt.imshow(branch_to_image,cmap="Greys_r")
                plt.show()
               
                        
                        
                        
            label_image, nregions = label(branched_image,return_num=True)
            # Deleting non-interest regions
            props=regionprops(label_image)
            va = []
            for prop in props:
                va.append(prop.area)
            indkeep = np.array(np.argmax(va))
            v = np.arange(nregions)
            vkill = np.setdiff1d(v,indkeep)+1 
            I2 = branched_image.copy()
            for kkill in vkill:
                I2[label_image==kkill]=0
                
                
            fig, ax = plt.subplots(figsize=(10,10))
            draw.overlay_skeleton_2d(dilated, skeleton2, dilate=1, axes=ax)
            plt.show()
            
            
            
            # # We analise the skeleton to find the 3 branches
            # pixel_graph, coordinates, degrees = skeleton_to_csgraph(skeleton2)
                        
                        
            #  # To separate the branches
            # i=degrees==2
            # fig=plt.figure(figsize=(15,15))
            # plt.imshow(i)           
            # plt.show()
            
            
            # # Labelling of each branch
            # labeled_branch, nbranches = label(i,return_num=True)
            # fig=plt.figure(figsize=(15,15))
            # plt.imshow(labeled_branch,cmap='jet')
            # plt.show()
            
            # We remove the small branches in order to remove the attached branches to our main structure
            # labeled_branch2=labeled_branch
            # coordinates_to_delete=[]
            # for i in range(1,nbranches+1):
            #     if howmanypixels(labeled_branch2==i)==1:
            #         labeled_branch2[labeled_branch2==i]*0    
            #     elif howmanypixels(labeled_branch2==i)<30:
            #         coordinates_to_delete.append(getcoordinates(labeled_branch2==i))
                        
            # # We remove the original pixels that correspond to those of a non-interest branch
            # # We have to prun again to eliminate the 1 pixel remaining of
            # skeleton3=removing(skeleton2,coordinates_to_delete)
            # pre_final_skeleton=pruning2(skeleton3,3)
            # fig=plt.figure(figsize=(15,15))
            # plt.imshow(pre_final_skeleton,cmap='GnBu')
            # plt.suptitle("skeleton3")
            # plt.show()
                        
                        
            # # # We remove the small branches again in order to remove the remaining branches 
            # # labeled_branch3, nbranches3 = label(pre_final_skeleton,return_num=True)
            # # coordinates_to_delete2=[]
            # # for i in range(1,nbranches3+1):
            # #     if howmanypixels(labeled_branch3==i)==1:
            # #         labeled_branch3[labeled_branch3==i]*0    
            # #     elif howmanypixels(labeled_branch3==i)<30:
            # #         coordinates_to_delete2.append(getcoordinates(labeled_branch3==i))
            
            # # skeleton4=removing(pre_final_skeleton,coordinates_to_delete2)
            # final_skeleton=pruning2(skeleton3,5)
            # fig=plt.figure(figsize=(15,15))
            # plt.imshow(final_skeleton,cmap='GnBu')
            # plt.show()
                        
                        
            # Draw all contours
            vis = np.zeros((h, w, 3), np.uint8)
            cv2.drawContours( vis, contours, -1, (128,255,255), 3, cv2.LINE_AA)                        
                                    
                        
                        
            # Draw the contour with maximum perimeter (omitting the first contour which is outer boundary of image
            #        Not necessary in this case
            vis2 = np.zeros((h, w, 3), np.uint8)
            # print(contours)
            perimeter=[]
            for cnt in contours[1:]:
                perimeter.append(cv2.arcLength(cnt,True))
            print (perimeter)
            print(max(perimeter))
            maxindex= perimeter.index(max(perimeter))
            print (maxindex)

            cv2.drawContours( vis2, contours, maxindex +1, (255,0,0), -1)

            # # Determine and draw main axis
            # length = 300
            # (x,y),(MA,ma),angle = cv2.fitEllipse(contours0[0])
            # print  (np.pi , angle)
            # print (angle * np.pi / 180.0)
            # print (np.cos(angle * np.pi / 180.0))
            # x2 =  int(round(x + length * np.cos((angle-90) * np.pi / 180.0)))
            # y2 =  int(round(y + length * np.sin((angle-90) * np.pi / 180.0)))
            # cv2.line(vis2, (int(x), int(y)), (x2,y2), (0,255,0),5)
            # print (x,y,x2,y2)

            # Show all images
            titles = ['Original Image','Threshold','Contours', 'Result', 'final Skeleton', "Skel"]
            images=[img, th1, vis, vis2,skeleton2, branched_image]
            for i in range(6):
                plt.subplot(2,3,i+1)
                plt.imshow(images[i],'gray')
                plt.title(titles[i]), plt.xticks([]), plt.yticks([])
            plt.show()
                        
                        
                        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            # bv = extract_bv(resized)    
            # cv2.imshow("Frame", bv)
            # binary = bv >   filters.threshold_otsu(bv)
            # bv = cv2.bitwise_not(bv)
            # ret, bw_img = cv2.threshold(bv,103,255,cv2.THRESH_BINARY)
            # kernel = np.ones((3,3), np.uint8)
            # for i in range(0,5):
                
            #     bw_img = cv2.erode(bw_img, kernel, iterations=1)
            #     # img_erosion = cv2.dilate(img_erosion, kernel, iterations=1)
                
            # bw_img = image_util.get_uint_image(bw_img)
            # cv2.imshow("Frame", bw_img*255)
            # skeleton = skeletonize(bw_img >0)

            # skeleton = image_util.get_uint_image(skeleton)
            # skeleton_rgb = image_util.bin_to_bgr_(skeleton)
            # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # ret, thresh = cv2.threshold(gray,127,255,0)
            # thresh = cv2.bitwise_not(thresh)
            # size = np.size(thresh)
            # skel = np.zeros(thresh.shape,np.uint8)
            
            # #get a cross shaped kernel
            # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            # i =0 
            # while True:
            #     i +=1
            #     print(i)
            #     open = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, element)
            #     temp = cv2.subtract(thresh, open)
            #     eroded = cv2.erode(thresh, element )
            #     skel = cv2.bitwise_or(skel,temp)
            #     thresh = eroded.copy() 
            #     if cv2.countNonZero(thresh)==0:
            #         break
                           
                        
            
            
            
            
            
            # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
            # # Taking a matrix of size 5 as the kernel
            # kernel = np.ones((5,5), np.uint8)
            
            # # The first parameter is the original image,
            # # kernel is the matrix with which image is 
            # # convolved and third parameter is the number 
            # # of iterations, which will determine how much 
            # # you want to erode/dilate a given image. 
            # img_erosion = cv2.erode(gray, kernel, iterations=2)
            
            # cv2.imshow("eroded",img_erosion)
            # cv2.waitKey(0)
            # ret, thresh = cv2.threshold(img_erosion, 200, 255, cv2.THRESH_OTSU )
            # # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 1)
            # thresh = cv2.bitwise_not(thresh)
            # binary = thresh > filters.threshold_otsu(thresh)
            
            # "Convert gray images to binary images using Otsu's method"
            
            # Otsu_Threshold = filters.threshold_otsu(resized)   
            # BW_Original = resized < Otsu_Threshold    # must set object region as 1, background region as 0 !
            
            # np.unique(binary)
            
            # ##### SKELETON 
            # skel, distance = medial_axis(binary, return_distance=True)
            
            # # skel = image_util.skeleton(gray)
            # skeleton = skeletonize(binary)
            # # Distance to the background for pixels of the skeleton
            # # dist_on_skel = distance * skel
            # skeleton = image_util.get_uint_image(skeleton)
            # BW_Original = image_util.get_uint_image(BW_Original)
            # skel = image_util.get_uint_image(skel)
            # cv2.imshow("skel",skel)
            # cv2.imshow("skeleton",skeleton)
            # # cv2.imshow("thin",thin)
            # # cv2.imshow('original',BW_Original)
            # # cv2.imshow("tresh",bw_img*255)
            # # # cv2.imshow("binary",binary)
            # cv2.waitKey(0)
            
            
  