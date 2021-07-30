import cv2
import numpy as np
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize, disk,binary_dilation,binary_closing,medial_axis
from skimage.measure import label, regionprops
from skimage.morphology import erosion, dilation
from skimage.morphology import disk
from sklearn import cluster
from skimage import data
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
from fil_finder import FilFinder2D
import astropy.units as u


veinSkeletonPath = "sample/tort/Reduced_Veins_Iso/skeleton/"
veinPath = "sample/tort/Reduced_Veins_Iso/"
arteryPath= "sample/tort/Reduced_Arteries_Iso/manual/"
arterySkeletonPath = "sample/tort/Reduced_Arteries_Iso/skeleton/"
csvPath = "sample/tort/"


def plot_comparison(original, filtered, filter_name):
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 15), sharex=True,
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
        skeleton[point[1] ,point[0]-1] = 0
        skeleton[point[1] ,point[0]+1] = 0
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
    # coordinates_to_delete=[]
    # for i in range(1,nregions+1):
    #     if howmanypixels(label_image==i)==1:
    #         label_image[label_image==i]*0    
    #     elif howmanypixels(label_image==i)<30:
    #         coordinates_to_delete.append(getcoordinates(label_image==i))
                
    # We remove the original pixels that correspond to those of a non-interest branch
    # We have to prun again to eliminate the 1 pixel remaining of
    # skeleton2=removing(skeleton,coordinates_to_delete)
   
                    
    # Deleting non-interest regions
    props=regionprops(label_image)
    
    va = []
    for prop in props:
        va.append(prop.area)
    index = []    
    va.sort()
    for idx,val in enumerate(va) :
        if val > 50 :
            index.append(idx)    
    # if l > 2:
    #     va_sort = index[l:]
    # else :
    #     va_sort = index
    
    v = np.arange(nregions)

    # vkill = []
    # for i,val in enumerate(va):
        
    vkill = np.setdiff1d(v,index)+1
    
    for kkill in vkill:
        skeleton[label_image==kkill]=0
        
    for point in branch_loc:
        skeleton[point[1],point[0]] = 1
        # skeleton[point[1] +1,point[0]] = 1
        # skeleton[point[1] -1,point[0]] = 1
        skeleton[point[1] ,point[0]-1] = 1
        skeleton[point[1] ,point[0]+1] = 1
        # skeleton[point[1]-1 ,point[0]-1] = 1
        # skeleton[point[1]-1 ,point[0]+1] = 1
        # skeleton[point[1]+1 ,point[0]+1] = 1
        # skeleton[point[1]+1 ,point[0]-1] = 1
    return skeleton
    
    
    
    
        
            

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
def myPruning (skeleton, size):
    '''remove iteratively end points "size" 
    times from the skeleton
    '''
    for i in range(0, size):
        endpoints = vessels_util.getEndPoints(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton

    
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

def km_clust(array, n_clusters):
        
    # Create a line array, the lazy way
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = cluster.KMeans(n_clusters=n_clusters, n_init=4)
    # Solve the k-means clustering problem
    k_m.fit(X)
    v_kmeans = k_m.predict(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_
    return(values, labels, v_kmeans)

####################################################

max_block = 1000
max_c = 200
max_value = 255
max_type = 1
max_binary_value = 255
trackbar_type = 'Type: \n 0: gaussian \n 1:Mean \n '
trackbar_value = 'Value'
trackbar_block_size = "block size"
trackbar_c_value = "c value"
window_name = 'Threshold Demo'


#######################################################
def Threshold_Demo(val):
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncateds
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    threshold_value = cv2.getTrackbarPos(trackbar_value, window_name)
    _, dst = cv2.threshold(blurred, threshold_value, max_binary_value, threshold_type )
    cv2.imshow(window_name, dst)
    
def Adaptive_Threshold_Demo(val):
    #0: Binary
    #1: Binary Inverted
    #2: Threshold Truncated
    #3: Threshold to Zero
    #4: Threshold to Zero Inverted
    global thresh
    block_size = cv2.getTrackbarPos(trackbar_block_size, window_name)
    c_value = cv2.getTrackbarPos(trackbar_c_value, window_name)
    threshold_type = cv2.getTrackbarPos(trackbar_type, window_name)
    method = 0
    if threshold_type == 0:
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else: 
        method = cv2.ADAPTIVE_THRESH_MEAN_C    
    block_size = max(3,block_size)
    # adaptive_method = 
    thresh = cv2.adaptiveThreshold(gray, 255,method, cv2.THRESH_BINARY , block_size, c_value)
    cv2.imshow(window_name, thresh)
    cv2.imshow("original", gray)
    

#######################################################





with open(csvPath+'arteryArtTort.csv', mode='w+') as csv_file:
    fieldnames = ['image', 'curve_length', 'chord_length','sd_theta','num_inflection_pts','num_critical_points','curvature','VTI','distance_tort']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    
        
    for filename in os.listdir(arteryPath):
       
        if filename.endswith(".tif") : 
            # print(os.path.join(trainPath, filename))
            # continue
            
            img = image_util.load_image(os.path.join(arteryPath, filename))
            img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
            # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
 
            # # equalize the histogram of the Y channel
            # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            
            # # convert the YUV image back to RGB format
            # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            
            # cv2.imshow('Color input image', img)
            # cv2.imshow('Histogram equalized', img_output)
            
            # cv2.waitKey(0)
            
            img = image_util.image_resize(img)   
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
            # enhance image
                 
            # gray = cv2.fastNlMeansDenoising(gray,None,10,7,21)
            gray = cv2.equalizeHist(gray)    
            # blurred = cv2.GaussianBlur(gray, (7, 7), cv2.BORDER_DEFAULT)
            plt.figure(figsize=(10,10))
            plt.suptitle("gray")
            plt.imshow(gray,cmap='Greys_r')
            plt.show()
            
            
            plt.subplot(131),plt.imshow(gray,'gray')
            plt.subplot(132),plt.imshow(img,'gray')
            plt.show()
            
          
            print(gray.shape)
            print(filename)
            h,w = gray.shape[:2]
            #
            # Drop top and bottom area of image with black parts.
            gray= gray[10:h-40, 20:]
            h, w = gray.shape[:2]
            plt.figure(figsize=(10,10))
            plt.suptitle("gray")
            plt.imshow(gray,cmap='Greys_r')
            plt.show()
            
            
            # Group similar grey levels using x clusters
            values, labels, v_kmeans = km_clust(gray, n_clusters = 4)
            # Create the segmented array from labels and values
            img_segm = np.choose(labels, values)
            # Reshape the array as the original image
            img_segm.shape = gray.shape
            v_kmeans.shape = gray.shape
            # Get the values of min and max intensity in the original image
           
            vmin = gray.min()
            vmax = gray.max()
            fig = plt.figure(1)
            # Plot the original image
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(gray,cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            ax1.set_title('Original image')
            # Plot the simplified color image
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(img_segm, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
            ax2.set_title('Simplified levels')

            plt.show()
            plt.imshow(v_kmeans, cmap="Greys_r")
            plt.show()    

            
            # img_segm = np.logical_not(img_segm)
            img_segm=image_util.get_uint_image(img_segm) 
            
            
            index=np.argmax(values)
            vessel=v_kmeans==index
            plt.figure(figsize=(10,10))
            plt.imshow(vessel,cmap='Greys_r')
            plt.show()
            
            vessel = image_util.get_uint_image(vessel)
            
            
            # val = [150,100]
            # label = labels == index
            # img_vessel= np.choose(label, val)
            # img_vessel.shape= gray.shape
            # plt.imshow( img_vessel,cmap='Greys_r')
            # plt.show()
            # # img_vessel = np.logical_not(img_vessel)
            # img_vessel=image_util.get_uint_image(img_vessel) 
            # plt.imshow( img_vessel,cmap='Greys_r')
            # plt.show()
            # # Reshape as 1D array
            # img_flat = gray.reshape((-1, 1))
            # fig = plt.figure(1)
            # ax1 = fig.add_subplot(1,2,1)
            # ax1.imshow(gray,cmap=plt.cm.gray)
            # ax2 = fig.add_subplot(1,2,2)
            # # Plot the histogram with 256 bins
            # ax2.hist(img_flat,256)

            # plt.show()        
                    


            # fig = plt.figure(figsize=(20, 20))
            # plt.subplot(311),plt.imshow(gray, 'gray'),plt.title('Input'),plt.axis('off')
            # plt.subplot(312),plt.imshow(backgroundRemoved, 'gray'),plt.title('Background Removed'),plt.axis('off')

            
            
            
            cv2.namedWindow(window_name)
            cv2.namedWindow("original")
            cv2.createTrackbar(trackbar_type, window_name , 0, max_type, Adaptive_Threshold_Demo)
            cv2.createTrackbar(trackbar_block_size, window_name , 3, max_block, Adaptive_Threshold_Demo)
            # Create Trackbar to choose Threshold value
            cv2.createTrackbar(trackbar_c_value, window_name , 2, max_c, Adaptive_Threshold_Demo)
            # Call the function to initialize
            Adaptive_Threshold_Demo(0)
            # Wait until user finishes program
            cv2.waitKey(0)
            
            # # # # Threshold image
            # ret,th1 = cv2.threshold(gray,101,255,cv2.THRESH_BINARY_INV)
            # # th1 = np.logical_not(th1)
            # # th1 = cv2.bitwise_not(th1)
            
            # plt.figure(figsize=(10,10))
            # plt.suptitle("binary inv")
            # plt.imshow(th1,cmap='Greys_r')
            # plt.show()
            
            # ret2, th2 = cv2.threshold(gray, 99, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            # plt.figure(figsize=(10,10))
            # plt.suptitle("binary inv + otsu")
            # plt.imshow(th2,cmap='Greys_r')
            # plt.show()
            
            
            # # thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 8)
            plt.figure(figsize=(10,10))
            plt.suptitle("adaptive")
            plt.imshow(thresh,cmap='Greys_r')
            plt.show()
            
            
            
            # Morphological operation (EROSION) on binary image resulting from bin treshold
            selem1 = disk(3)
            selem = disk(3)
            eroded = erosion(thresh, selem)
            dilated = dilation(eroded, selem)
            dilated = dilation(dilated, selem1)
            # dilated = erosion(dilated, selem1)
            plt.figure(figsize=(10,10))
            plt.imshow(dilated,cmap='Greys_r')
            plt.show()
            
            
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
            
            contour_image = image_util.get_uint_image(dilated)
            _, contours0, hierarchy = cv2.findContours( contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.approxPolyDP(cnt, 4, True) for cnt in contours0[0]]
            
            
            # Skeletonization
            skeleton=skeletonize(dilated >0)
            # Ploting the skeleton
            # plt.figure(figsize=(15,15))
            # plt.suptitle("skeleton of closed")
            # plt.imshow(skeleton,cmap='Greys_r')
            # plt.show()
            
            
            # skeleton=cleaningtop(skeleton)
            fig=plt.figure(figsize=(15,15))
            fig.suptitle("cleaningtop of skeleton")
            plt.imshow(skeleton,cmap='Greys_r')
            plt.show()
            
            # skeleton=cleaningBottom(skeleton)
            # fig=plt.figure(figsize=(15,15))
            # fig.suptitle("cleaningbottom of skeleton")
            # plt.imshow(skeleton,cmap='Greys_r')
            # plt.show()
            
            skeleton1 = pruning(skeleton,30)
            # skeleton1 = pruning(skeleton1,30)
            # fig=plt.figure(figsize=(15,15))
            # plt.imshow(skeleton1,cmap='Greys_r')
            # fig.suptitle("pruning of skeleton")
            # plt.show()
            
            
            # skeleton2=skeletonize(skeleton1)
            # plt.figure(figsize=(15,15))
            # plt.imshow(skeleton2,cmap='Greys_r')
            # plt.suptitle("skeleton of skeleton")
            # plt.show()
            
            
            
            
            
            
            

            
            
            
            # branched_image  = skeleton2.copy()
            
            # for i in range(3):
            #     branch_to_image = branched_image.copy()
            #     branch_to_image = image_util.get_uint_image(branch_to_image)
            #     branch_locations = vessels_util.getIntersections(branched_image)
            #     if not branch_locations :
            #         break
            #     branched_image = remove_small_branches(branch_locations, branched_image.copy())
            #     for points in branch_locations:
        
            #         if not points:
            #             print("List is empty")
            #             continue 
                
            #         cv2.circle(branch_to_image,tuple(points) , 2, (255, 255, 0), 5)        


            #     plt.figure(figsize=(10,10))
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(branched_image,cmap='Greys_r')
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(branch_to_image,cmap="Greys_r")
            #     plt.show()
               
                        
                        
                        
            # label_image, nregions = label(branched_image,return_num=True)
            # # Deleting non-interest regions
            # props=regionprops(label_image)
            # va = []
            # for prop in props:
            #     va.append(prop.area)
            # indkeep = np.array(np.argmax(va))
            # v = np.arange(nregions)
            # vkill = np.setdiff1d(v,indkeep)+1 
            # I2 = branched_image.copy()
            # for kkill in vkill:
            #     I2[label_image==kkill]=0
                
                

            
                        # fil finder part, search longest path on skeleton
            fil = FilFinder2D(skeleton1, distance=250 * u.pc, mask=skeleton1)
            # fil.preprocess_image(flatten_percent=85)
            fil.create_mask(border_masking=True, verbose=False,
            use_existing_mask=True)
            fil.medskel(verbose=False)
            fil.analyze_skeletons(branch_thresh=50* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

            # Show the longest path
            # plt.imshow(fil.skeleton, cmap='gray')
            # plt.contour(fil.skeleton_longpath, colors='r')
            # plt.axis('off')
            # plt.show()
            
            final_skeleton = fil.skeleton_longpath 
            splitted = filename.split('.')
            filename = arterySkeletonPath + splitted[0] + "_skel.png"
            cv2.imwrite(filename, final_skeleton*255)
            
            fig, ax = plt.subplots(figsize=(10,10))
            draw.overlay_skeleton_2d(gray, final_skeleton, dilate=1, axes=ax)
            plt.show()
            # # We analise the skeleton to find the 3 branches
            # pixel_graph, coordinates, degrees = skeleton_to_csgraph(skeleton2)
            # branch_data = summarize(Skeleton(skeleton2))
            # draw.overlay_skeleton_networkx(pixel_graph, coordinates, image=skeleton2)
            # i=degrees==2
            # fig=plt.figure(figsize=(15,15))
            # plt.imshow(i)
            
            
            # Labelling of each branch
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
                        
            # We remove the original pixels that correspond to those of a non-interest branch
            # We have to prun again to eliminate the 1 pixel remaining of
            # skeleton3=removing(skeleton2,coordinates_to_delete)
            # pre_final_skeleton=pruning2(skeleton3,3)
            # fig=plt.figure(figsize=(15,15))
            # plt.imshow(pre_final_skeleton,cmap='GnBu')
            # plt.suptitle("skeleton3")
            # plt.show()
                        
                        
            # # We remove the small branches again in order to remove the remaining branches 
            # labeled_branch3, nbranches3 = label(pre_final_skeleton,return_num=True)
            # coordinates_to_delete2=[]
            # for i in range(1,nbranches3+1):
            #     if howmanypixels(labeled_branch3==i)==1:
            #         labeled_branch3[labeled_branch3==i]*0    
            #     elif howmanypixels(labeled_branch3==i)<30:
            #         coordinates_to_delete2.append(getcoordinates(labeled_branch3==i))
            
            # skeleton4=removing(pre_final_skeleton,coordinates_to_delete2)
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
            # print (perimeter)
            # print(max(perimeter))
            maxindex= perimeter.index(max(perimeter))
            # print (maxindex)

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
            images=[img, thresh, vis, vis2,skeleton, final_skeleton ]
            # for i in range(6):
            #     plt.subplot(2,3,i+1)
            #     plt.imshow(images[i],'gray')
            #     plt.title(titles[i]), plt.xticks([]), plt.yticks([])
            # plt.show()
                        
                        
                        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
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
            
            
  