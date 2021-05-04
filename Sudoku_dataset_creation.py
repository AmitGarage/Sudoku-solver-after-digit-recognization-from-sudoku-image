#!/usr/bin/env python
# coding: utf-8

import os , shutil
from os.path import isfile, join
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image , ImageFilter
import torchvision.transforms as transforms
import numpy as np
#from spectral import imshow
from skimage.filters import threshold_local,median
from skimage import measure,feature
import skimage.transform as tnf
from skimage.morphology import erosion,dilation,closing,square,disk,binary_dilation,binary_erosion
from skimage.segmentation import clear_border

# Model used for training
class cnn_model(nn.Module) :
    def __init__(self) :
        super(cnn_model , self).__init__()
        self.conv1 = nn.Conv2d(1,20,kernel_size=5)
        self.conv2 = nn.Conv2d(20,40,kernel_size=5)
        self.conv2_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(640,120)
        self.fc2 = nn.Linear(120,10)
        
    def forward(self , input_x) :
        input_x = F.relu(F.max_pool2d(self.conv1(input_x),2))
        input_x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(input_x)),2))
        input_x = input_x.view(-1,640)
        input_x = self.fc1(input_x)
        input_x = F.dropout(input_x,training=self.training)
        input_x = self.fc2(input_x)
        return F.log_softmax(input_x)

# Opening image from which sudoku needs to extract
sudoku_image = Image.open('/storage/Suduko_solver_data/Suduko81.jpg')
#sudoku_image = Image.open('/storage/Suduko_solver_data/data/mnist_png/testing/8/8_derived_92_124.jpg')
# Converting image to grayscale
sudoku_image = sudoku_image.convert('L')
#sudoku_image1 = sudoku_image.filter(ImageFilter.FIND_EDGES) 
#sudoku_image = sudoku_image+sudoku_image1
#sudoku_image = sudoku_image.rotate(90)
#sudoku_image = sudoku_image.rotate(90)
#sudoku_image = sudoku_image.rotate(90)
#sudoku_image = sudoku_image.rotate(-90)
sukodu_image_array = np.array(sudoku_image)
#sukodu_image_array = sukodu_image_array[10:-10,10:-10]
#sukodu_image_array[sukodu_image_array > 120] = 255
#sukodu_image_array[sukodu_image_array < 220] = 0
plt.figure()
plt.imshow(sukodu_image_array, cmap='gray')

#sukodu_image_array = sukodu_image_array[1520:1540,1600:1650]
sukodu_image_array[sukodu_image_array > 100] = 255
#print(sukodu_image_array)
#plt.figure()
#plt.imshow(sukodu_image_array, cmap='gray')

#print(sukodu_image_array.shape)


# Computing a threshold mask image based on local pixel neighborhood
threshold = threshold_local(sukodu_image_array,block_size=47,offset=20)
# Extracting image above threshold value
filtered_image = sukodu_image_array > threshold
plt.figure()
plt.imshow(filtered_image, cmap='gray')

# Extracting contours from image
contours = measure.find_contours(filtered_image)
#print(len(contours))
#print(contours[720])
#print(contours[720][:,1])
#print(contours[720][:,0])

#fig, ax = plt.subplots()
#ax.imshow(filtered_image, cmap=plt.cm.gray)

#for contour in contours[10:1000]:
#    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

#ax.axis('image')
#ax.set_xticks([])
#ax.set_yticks([])
#plt.show()

contour_list = []
contour_number = 0
# Creating contour list to extract contour with maximum area
for n , contour in enumerate(contours) :
    contour_details = []
    contour_details.append(len(contour))
    contour_details.append(contour_number)
    contour_list.append(contour_details)
    contour_number += 1
    
# Sorting contour list in reverse order
contour_list.sort(key=lambda x:x[0] ,reverse = True)
contour_list = np.array(contour_list)
#print(contour_list[:,1])
contour_to_traverse = []
# Traversing contour list and extracting coordinates
for i in contour_list[:,1] :
    contour_to_traverse.append(contours[i])

#print(contours[100:728],contour_to_traverse[0])

# Function to calculate area proportion based on original area and cropped area
def calculate_area_proportion ( original_height , original_width ,height , width ) :
    cropped_area = height * width
    original_area = original_height * original_width
    area_proportion = (cropped_area/original_area)*100
    return area_proportion


fig , axis = plt.subplots()
x1,x2,y1,y2 = 0,0,0,0
axis.imshow(filtered_image, interpolation='nearest',cmap=plt.cm.gray)
original_height , original_width = filtered_image.shape
appproximate_sudoku = []
c1,c2,c3,c4 = np.array([]),np.array([]),np.array([]),np.array([])
#final_coordinates = np.array([])
for contour_cordinates in contour_to_traverse :
    # Only traverse when 1st coordinates matches with last coordinates and no of coordinates in that contour is greater than 3
    if contour_cordinates[0,0] == contour_cordinates[len(contour_cordinates)-1,0] and contour_cordinates[0,1] == contour_cordinates[len(contour_cordinates)-1,1] and len(contour_cordinates) > 3:
        new_sudoku = contour_cordinates.copy()
        # Approximate a polygonal with the specified tolerance.
        appproximate_sudoku = measure.approximate_polygon(new_sudoku,tolerance=20.0)
        #print(min(appproximate_sudoku[:,0]),max(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1]),max(appproximate_sudoku[:,1]))
        # Traversing sudoku and comparing polygon coordinates with min/max of polygon coordinates - orginal height / weight with a margin of 6% and find all 4 coordinates
        for row in appproximate_sudoku :
            #print(row[0],row[1])
            #print(min(appproximate_sudoku[:,0])-(original_height*0.06),min(appproximate_sudoku[:,0])+(original_height*0.06),min(appproximate_sudoku[:,1])-(original_width*0.06),min(appproximate_sudoku[:,1])+(original_width*0.06))
            if row[0] >= min(appproximate_sudoku[:,0])-(original_height*0.06) and row[0] <= min(appproximate_sudoku[:,0])+(original_height*0.06) and row[1] >= min(appproximate_sudoku[:,1])-(original_width*0.06) and row[1] <= min(appproximate_sudoku[:,1])+(original_width*0.06):
                c1 = np.array([row[0],row[1]])
                #print('c1')
            #print(min(appproximate_sudoku[:,0])-(original_height*0.06),min(appproximate_sudoku[:,0])+(original_height*0.06),max(appproximate_sudoku[:,1])-60,max(appproximate_sudoku[:,1])+(original_width*0.06))
            if row[0] >= min(appproximate_sudoku[:,0])-(original_height*0.06) and row[0] <= min(appproximate_sudoku[:,0])+(original_height*0.06) and row[1] >= max(appproximate_sudoku[:,1])-(original_width*0.06) and row[1] <= max(appproximate_sudoku[:,1])+(original_width*0.06):
                c2 = np.array([row[0],row[1]])
                #print('c2')
            #print(max(appproximate_sudoku[:,0])-(original_height*0.06),max(appproximate_sudoku[:,0])+(original_height*0.06),max(appproximate_sudoku[:,1])-60,max(appproximate_sudoku[:,1])+(original_width*0.06))
            if row[0] >= max(appproximate_sudoku[:,0])-(original_height*0.06) and row[0] <= max(appproximate_sudoku[:,0])+(original_height*0.06) and row[1] >= max(appproximate_sudoku[:,1])-(original_width*0.06) and row[1] <= max(appproximate_sudoku[:,1])+(original_width*0.06):
                c3 = np.array([row[0],row[1]])
                #print('c3')
            #print(max(appproximate_sudoku[:,0])-(original_height*0.06),max(appproximate_sudoku[:,0])+(original_height*0.06),min(appproximate_sudoku[:,1])-60,min(appproximate_sudoku[:,1])+(original_width*0.06))
            if row[0] >= max(appproximate_sudoku[:,0])-(original_height*0.06) and row[0] <= max(appproximate_sudoku[:,0])+(original_height*0.06) and row[1] >= min(appproximate_sudoku[:,1])-(original_width*0.06) and row[1] <= min(appproximate_sudoku[:,1])+(original_width*0.06):
                c4 = np.array([row[0],row[1]])
                #print('c4')
        #print(np.array([min(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1])]))
        #print(len(c2))
        # If any of coordinates not found , then will be used as below
        if len(c1) == 0 :
            c1 = np.array([min(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1])])
        if len(c2) == 0 :
            c2 = np.array([min(appproximate_sudoku[:,0]),max(appproximate_sudoku[:,1])])
        if len(c3) == 0 :
            c3 = np.array([max(appproximate_sudoku[:,0]),max(appproximate_sudoku[:,1])])
        if len(c4) == 0 :
            c4 = np.array([max(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1])])
        #c5 = np.array([min(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1])])
        c5 = c1
        #print(c1,c2,c3,c4)
        final_coordinates = np.array([c1,c2,c3,c4,c5])
        appproximate_sudoku = final_coordinates
        #print(final_coordinates)
        #print(type(appproximate_sudoku))
        #print(contour_cordinates[0,:],min(appproximate_sudoku[:,0]),max(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1]),max(appproximate_sudoku[:,1]))
        axis.plot(appproximate_sudoku[:, 1], appproximate_sudoku[:, 0], linewidth=2)
        break

#print(len(appproximate_sudoku))
#for i in range(len(appproximate_sudoku)):
#    axis.scatter(appproximate_sudoku[i][1],appproximate_sudoku[i][0])

# Arranging sudoku coordinates in order either in clockwise / Anit-clockwise
# Flip array(entries in each column) in left-right direction
desired_matrix = np.fliplr(appproximate_sudoku[0:4])
#print(desired_matrix)
# Sorting along y axis
desired_matrix = desired_matrix[desired_matrix[:,1].argsort()]
#print(desired_matrix)
desired_matrix1 = desired_matrix[:2]
desired_matrix2 = desired_matrix[2:]
#print(desired_matrix1,desired_matrix2)
# Sorting along x axis
desired_matrix1 = desired_matrix1[desired_matrix1[:,0].argsort()]
desired_matrix2 = desired_matrix2[desired_matrix2[:,0].argsort()[::-1]]
#print(desired_matrix1,desired_matrix2)
desired_matrix = np.concatenate((desired_matrix1,desired_matrix2),axis=0)
#print(desired_matrix)
#print(np.sqrt(np.sum((desired_matrix[0]-desired_matrix[1])**2)))
#print(np.sqrt(np.sum((desired_matrix[1]-desired_matrix[2])**2)))
#print(np.sqrt(np.sum((desired_matrix[2]-desired_matrix[3])**2)))
#print(np.sqrt(np.sum((desired_matrix[3]-desired_matrix[0])**2)))
# Calculating height and width from extracted sudoku coordinates
cropped_height = np.sqrt(np.sum((desired_matrix[0]-desired_matrix[1])**2))
cropped_width = np.sqrt(np.sum((desired_matrix[2]-desired_matrix[3])**2))
#print(calculate_area_proportion(original_height , original_width ,cropped_height , cropped_width))
# Calculating area percentage of extracted sudoku over orignal image
# If less than 20% , then whole image is sudoku and setting coordinates again
if calculate_area_proportion(original_height , original_width ,cropped_height , cropped_width) < 20 :
    c1 = np.array([0,0])
    c2 = np.array([original_height,0])
    c3 = np.array([original_height,original_width])
    c4 = np.array([0,original_width])
    c5 = c1
    final_coordinates = np.array([c1,c2,c3,c4,c5])
    appproximate_sudoku = final_coordinates
    #print(final_coordinates)
    #print(type(appproximate_sudoku))
    #print(contour_cordinates[0,:],min(appproximate_sudoku[:,0]),max(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1]),max(appproximate_sudoku[:,1]))
    axis.plot(appproximate_sudoku[:, 1], appproximate_sudoku[:, 0], linewidth=2)
    desired_matrix = appproximate_sudoku[0:4]
    #print(desired_matrix)

#for i in range(len(appproximate_sudoku)):
#    axis.scatter(appproximate_sudoku[i][1],appproximate_sudoku[i][0])

# Initalizing final sudoku size over which extracted image will be transformed
final_sudoku_size = np.array(((0,0),(270,0),(270,270),(0,270)))
transformation = tnf.ProjectiveTransform()
transformation_estimation = transformation.estimate(final_sudoku_size,desired_matrix)
#print(transformation_estimation,transformation)
# Warp an image according to a given coordinate transformation
Wrapping = tnf.warp(filtered_image,transformation,output_shape=(270,270))
#print(Wrapping,np.amax(Wrapping))
# Inverting images values to convert black to white and vice versa
Wrapping = abs(Wrapping-np.amax(Wrapping))
#print(Wrapping,np.amax(Wrapping))
plt.figure()
plt.imshow(Wrapping, cmap='gray')

# Removing borders ( value of all pixels on most of column or row is 1 / 0 ) if any on all the 4 sides of image
#print(Wrapping[0,:],Wrapping[-1,:],Wrapping[:,0],Wrapping[:,-1],len(Wrapping[0,:]))
reduce_x_0 , reduce_x_1 , reduce_y_0 , reduce_y_1 = 0,0,0,0
for x in range(int(len(Wrapping[0,:])/2)) :
    #print(np.count_nonzero(Wrapping[x,:]==1),np.count_nonzero(Wrapping[-x-1,:]==1),np.count_nonzero(Wrapping[:,x]==1),np.count_nonzero(Wrapping[:,-x-1]==1),len(Wrapping[0,:]),int(len(Wrapping[0,:])/2))
    #print(np.count_nonzero(Wrapping[x,:]==0),np.count_nonzero(Wrapping[-x-1,:]==0),np.count_nonzero(Wrapping[:,x]==0),np.count_nonzero(Wrapping[:,-x-1]==0),len(Wrapping[0,:]),int(len(Wrapping[0,:])/2))
    if np.count_nonzero(Wrapping[x,:]==1) > len(Wrapping[0,:])-5 or np.count_nonzero(Wrapping[x,:]==0) > len(Wrapping[0,:])-5:
        reduce_x_0 += 1
        print('reduce_x_0 - '+str(reduce_x_0))
    if np.count_nonzero(Wrapping[-x-1,:]==1) > len(Wrapping[0,:])-5 or np.count_nonzero(Wrapping[-x-1,:]==0) > len(Wrapping[0,:])-5:
        reduce_x_1 += 1
        print('reduce_x_1 - '+str(reduce_x_1))
    if np.count_nonzero(Wrapping[:,x]==1) > len(Wrapping[:,0])-5 or np.count_nonzero(Wrapping[:,x]==0) > len(Wrapping[:,0])-5:
        reduce_y_0 += 1
        print('reduce_y_0 - '+str(reduce_y_0))
    if np.count_nonzero(Wrapping[:,-x-1]==1) > len(Wrapping[:,0])-5 or np.count_nonzero(Wrapping[:,-x-1]==0) > len(Wrapping[:,0])-5:
        reduce_y_1 += 1
        print('reduce_y_1 - '+str(reduce_y_1))
    if ( np.count_nonzero(Wrapping[x,:]==1) < len(Wrapping[0,:])-5 or np.count_nonzero(Wrapping[x,:]==0) > len(Wrapping[0,:])-5 ) and ( np.count_nonzero(Wrapping[-x-1,:]==1) < len(Wrapping[0:])-5 or np.count_nonzero(Wrapping[-x-1,:]==0) > len(Wrapping[0:])-5 )and ( np.count_nonzero(Wrapping[:,x]==1) < len(Wrapping[:,0])-5 or np.count_nonzero(Wrapping[:,x]==0) > len(Wrapping[:,0])-5 ) and ( np.count_nonzero(Wrapping[:-x-1]==1) < len(Wrapping[:,0])-5 or np.count_nonzero(Wrapping[:-x-1]==0) > len(Wrapping[:,0])-5 ) :
        print('break')
        print(reduce_x_0,reduce_x_1,reduce_y_0,reduce_y_1)
        if reduce_x_0 != 0 or reduce_x_1 != 0 or reduce_y_0 != 0 or reduce_y_1 != 0:
            if reduce_x_1 == 0 :
                reduce_x_1 = 1
            if reduce_y_1 == 0 :
                reduce_y_1 = 1
            Wrapping = Wrapping[reduce_x_0:-reduce_x_1-1,reduce_y_0:-reduce_y_1-1]
            original_height , original_width = Wrapping.shape
            c1 = np.array([0,0])
            c2 = np.array([original_height,0])
            c3 = np.array([original_height,original_width])
            c4 = np.array([0,original_width])
            c5 = c1
            final_coordinates = np.array([c1,c2,c3,c4,c5])
            appproximate_sudoku = final_coordinates
            desired_matrix = appproximate_sudoku[0:4]
            # Image size reduced because of border removal . So , again transformation
            final_sudoku_size = np.array(((0,0),(270,0),(270,270),(0,270)))
            transformation = tnf.ProjectiveTransform()
            transformation_estimation = transformation.estimate(final_sudoku_size,desired_matrix)
            print(transformation_estimation,transformation)
            Wrapping = tnf.warp(Wrapping,transformation,output_shape=(270,270))
        break;
plt.figure()
#print(reduce)
plt.imshow(Wrapping, cmap='gray')

selem = square(1)
#print(selem)
#print(closing(Wrapping,selem))
plt.figure()
plt.imshow(closing(Wrapping,selem), cmap='gray')
#Closing remove small dark spots and connect small bright cracks
final_wrapped_image = closing(Wrapping,selem)

#plt.figure()
#plt.imshow(binary_dilation(Wrapping,selem), cmap='gray')
#final_wrapped_image = binary_dilation(Wrapping,selem)

#selem=disk(2)
#plt.figure()
#plt.imshow(closing(erosion(Wrapping,selem),square(4)), cmap='gray')
#plt.imshow(dilation(Wrapping,selem), cmap='gray')
#final_wrapped_image = erosion(Wrapping,selem)

# Initalizing 9 x 9 sudoku array with 0
Sudoku = np.zeros([9,9],dtype=int)

# Function for removing boundary
def boundary_removal( input_image ) :
    height,width = input_image.shape
    input_image[:3,:] = 0
    input_image[:,:3] = 0
    input_image[height-3:,:] = 0
    input_image[:,width-3:] = 0
    return input_image

# Function to extract digit from sudoku
def digit_encoder() :
    # Loading pretrained weights
    pretrained_weights = torch.load('/storage/Suduko_solver_data/model_2.pth')
    pretrained_model = cnn_model()
    pretrained_model.load_state_dict(pretrained_weights)
    pretrained_model.eval()
    height , width = final_wrapped_image.shape
    #print(height , width)
    for x in range(9) :
        for y in range(9) :
            #print((x*int(height/9))+1,((x+1)*int(height/9))-1,(y*int(width/9))+1 ,((y+1)*int(width/9))-1)
            # Cutting 30 x 30 image from sudoku for digit prediction
            cell_dimensions = final_wrapped_image[(x*int(height/9))+1 : ((x+1)*int(height/9))-1,(y*int(width/9))+1 : ((y+1)*int(width/9))-1]
            plt.figure()
            plt.imshow(cell_dimensions, cmap='gray')
            # Removing boundary
            cell_dimensions = boundary_removal(cell_dimensions)
            plt.figure()
            plt.imshow(cell_dimensions, cmap='gray')
            # To check no of values are 0
            Zero_digit_check = np.count_nonzero(cell_dimensions==0)
            #print(Zero_digit_check)
            cell_dimensions=np.expand_dims(np.expand_dims(cell_dimensions, axis=0), axis=0)
            cell_dimensions = cell_dimensions.astype(float)
            #print(cell_dimensions.shape)
            #print(cell_dimensions)
            #test_dataset= torchvision.datasets.Image(cell_dimensions,transform=Image_transformation_function)
            # Converting to tensor for loading to device for prediction
            test_dataset = torch.from_numpy(cell_dimensions) #Image_transformation_function(cell_dimensions)
            test_dataset.to(device_to_use)
            #training_data_loader = torch.utils.data.DataLoader(MNIST_dataset_training,batch_size=64,shuffle=True)
            predicted_output = pretrained_model(test_dataset.float())
            print(predicted_output)
            predicted_digit = predicted_output.data.max(1,keepdim=True)[1]
            plt.title('Expected Digit - '+str(predicted_digit))
            #print(predicted_digit,predicted_output)
            # If most of values are 0 , then can be considered as 0 values
            if ( predicted_digit == 1 and Zero_digit_check > 760 ) or predicted_digit == 0:
                image_saved_count=len(list(os.listdir('/storage/Suduko_solver_data/Checking_generated_images/0_expected/')))
                image_saved_count += 1
                saved_image.save('/storage/Suduko_solver_data/Checking_generated_images/0_expected/0_derived_'+str(image_saved_count)+'.png')
            else :
                image_saved_count=len(list(os.listdir('/storage/Suduko_solver_data/Checking_generated_images/'+str(predicted_digit.item())+'/')))
                image_saved_count += 1
                saved_image.save('/storage/Suduko_solver_data/Checking_generated_images/'+str(predicted_digit.item())+'/'+str(predicted_digit.item())+'_derived_'+str(image_saved_count)+'.png')
            #print(predicted_digit)


digit_encoder()

# Displaying images of all folders

image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/1/') if isfile(join('/storage/Suduko_solver_data/Checking_generated_images/1/',file))]
path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/1/'
for x in image_files :
    Image_to_be_displayed = Image.open(path_input_file+x)
    plt.figure()
    plt.imshow(Image_to_be_displayed, cmap='gray')
    plt.title(x)

image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/2/') if isfile(join('/storage/Suduko_solver_data/Checking_generated_images/2/',file))]
path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/2/'
for x in image_files :
    Image_to_be_displayed = Image.open(path_input_file+x)
    plt.figure()
    plt.imshow(Image_to_be_displayed, cmap='gray')
    plt.title(x)

image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/3/') if isfile(join('/storage/Suduko_solver_data/Checking_generated_images/3/',file))]
path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/3/'
for x in image_files :
    Image_to_be_displayed = Image.open(path_input_file+x)
    plt.figure()
    plt.imshow(Image_to_be_displayed, cmap='gray')
    plt.title(x)

image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/4/') if isfile(join('/storage/Suduko_solver_data/Checking_generated_images//4',file))]
path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/4/'
for x in image_files :
    Image_to_be_displayed = Image.open(path_input_file+x)
    plt.figure()
    plt.imshow(Image_to_be_displayed, cmap='gray')
    plt.title(x)

image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/5/') if isfile(join('/storage/Suduko_solver_data/Checking_generated_images/5/',file))]
path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/5/'
for x in image_files :
    Image_to_be_displayed = Image.open(path_input_file+x)
    plt.figure()
    plt.imshow(Image_to_be_displayed, cmap='gray')
    plt.title(x)

image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/6/') if isfile(join('/storage/Suduko_solver_data/Checking_generated_images/6/',file))]
path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/6/'
for x in image_files :
    Image_to_be_displayed = Image.open(path_input_file+x)
    plt.figure()
    plt.imshow(Image_to_be_displayed, cmap='gray')
    plt.title(x)

image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/7/') if isfile(join('/storage/Suduko_solver_data/Checking_generated_images/7/',file))]
path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/7/'
for x in image_files :
    Image_to_be_displayed = Image.open(path_input_file+x)
    plt.figure()
    plt.imshow(Image_to_be_displayed, cmap='gray')
    plt.title(x)

image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/8/') if isfile(join('/storage/Suduko_solver_data/Checking_generated_images/8/',file))]
path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/8/'
for x in image_files :
    Image_to_be_displayed = Image.open(path_input_file+x)
    plt.figure()
    plt.imshow(Image_to_be_displayed, cmap='gray')
    plt.title(x)

image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/9/') if isfile(join('/storage/Suduko_solver_data/Checking_generated_images/9/',file))]
path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/9/'
for x in image_files :
    Image_to_be_displayed = Image.open(path_input_file+x)
    plt.figure()
    plt.imshow(Image_to_be_displayed, cmap='gray')
    plt.title(x)


# Moving images from one folder to another
for i in range(1,10) :
    image_files = [file for file in os.listdir('/storage/Suduko_solver_data/Checking_generated_images/'+str(i)+'/')]
    path_input_file = '/storage/Suduko_solver_data/Checking_generated_images/'+str(i)+'/'
    image_saved_count=len(list(os.listdir('/storage/Suduko_solver_data/New_folder/'+str(i)+'/')))
    image_saved_count += 1
    for x in image_files :
        #x = x[:-4]+'_'+str(image_saved_count)+'.jpg'
        os.rename(path_input_file+x,'/storage/Suduko_solver_data/New_folder/'+str(i)+'/'+x[:-4]+'new_'+str(image_saved_count)+'.jpg')
        image_saved_count += 1
        #print(path_input_file+x)
        #shutil.move(path_input_file+x,'/storage/Suduko_solver_data/Generated_images/'+str(i)+'/'+x)

for i in range(1,10) :
    image_files = [file for file in os.listdir('/storage/Suduko_solver_data/New_folder/'+str(i)+'/') if isfile(join('/storage/Suduko_solver_data/New_folder/'+str(i)+'/',file))]
    path_input_file = '/storage/Suduko_solver_data/New_folder/'+str(i)+'/'
    for x in image_files :
        print(path_input_file+x)
        shutil.copy(path_input_file+x,'/storage/Suduko_solver_data/data/mnist_png/training/'+str(i)+'/'+x)

#for i in range(0,10) :
#    image_files = [file for file in os.listdir('/storage/data/mnist_png/training/'+str(i)+'/')]
#    path_input_file = '/storage/data/mnist_png/training/'+str(i)+'/'
#    for x in image_files :
#        print(path_input_file+x)
#        shutil.copy(path_input_file+x,'/storage/Suduko_solver_data/data/mnist_png/training/'+str(i)+'/'+x)


#Image_to_be_displayed = Image.open('/storage/Suduko_solver_data/Checking_generated_images/2/2_derived_7.jpg').convert('L')
#plt.figure()
#plt.imshow(Image_to_be_displayed, cmap='gray')
#image_array = np.asarray(Image_to_be_displayed)
#image_array[image_array <= 200] = 0
#image_array[image_array > 200] = 255
#print(np.resize(image_array,(28,28)))
#print(image_array)
#image_array = np.resize(image_array,(28,28))
#plt.figure()
#plt.imshow(Image.fromarray(image_array), cmap='gray')
#Image.fromarray(image_array).save('/storage/Suduko_solver_data/check.png')
#Image_to_be_displayed = Image.open('/storage/Suduko_solver_data/check.png')
#plt.figure()
#plt.imshow(Image_to_be_displayed, cmap='gray')
#print(np.array(Image_to_be_displayed))
#image_array = np.array(Image_to_be_displayed)
#image_array[image_array <= 200] = 0
#image_array[image_array > 200] = 255
#plt.figure()
#plt.imshow(Image.fromarray(image_array), cmap='gray')
#Image.fromarray(image_array).save('/storage/Suduko_solver_data/check1.png')
#Image_to_be_displayed = Image.open('/storage/Suduko_solver_data/check1.png')
#plt.figure()
#plt.imshow(Image_to_be_displayed, cmap='gray')
