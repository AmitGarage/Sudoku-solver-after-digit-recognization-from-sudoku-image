# In[22]:


import numpy as np
#from spectral import imshow
from skimage.filters import threshold_local
from skimage import measure,feature
import skimage.transform as tnf
from skimage.morphology import erosion,dilation,closing,square,disk,binary_dilation,binary_erosion
from skimage.segmentation import clear_border


# In[93]:


sudoku_image = Image.open('/storage/Suduko_solver_data/Suduko120.jpg')
sudoku_image = sudoku_image.convert('L')
sukodu_image_array = np.array(sudoku_image)
#sukodu_image_array = sukodu_image_array[10:-10,10:-10]
#sukodu_image_array[sukodu_image_array > 100] = 255
plt.figure()
plt.imshow(sukodu_image_array, cmap='gray')


# In[94]:


print(sukodu_image_array.shape)


# In[95]:


threshold = threshold_local(sukodu_image_array,block_size=47,offset=20)
filtered_image = sukodu_image_array > threshold
plt.figure()
plt.imshow(filtered_image, cmap='gray')


# In[96]:


contours = measure.find_contours(filtered_image)
print(len(contours))
#print(contours[720])
#print(contours[720][:,1])
#print(contours[720][:,0])


# In[97]:


fig, ax = plt.subplots()
ax.imshow(filtered_image, cmap=plt.cm.gray)

for contour in contours[10:1000]:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()


# In[98]:


contour_list = []
contour_number = 0
for n , contour in enumerate(contours) :
    contour_details = []
    contour_details.append(len(contour))
    contour_details.append(contour_number)
    contour_list.append(contour_details)
    contour_number += 1
contour_list.sort(key=lambda x:x[0] ,reverse = True)
contour_list = np.array(contour_list)
print(contour_list[:,1])
contour_to_traverse = []
for i in contour_list[:,1] :
    contour_to_traverse.append(contours[i])


# In[99]:


print(contours[100:728],contour_to_traverse[0])


# In[100]:


def calculate_area_proportion ( original_height , original_width ,height , width ) :
    cropped_area = height * width
    original_area = original_height * original_width
    area_proportion = (cropped_area/original_area)*100
    return area_proportion


# In[101]:


fig , axis = plt.subplots()
x1,x2,y1,y2 = 0,0,0,0
axis.imshow(filtered_image, interpolation='nearest',cmap=plt.cm.gray)
original_height , original_width = filtered_image.shape
appproximate_sudoku = []
c1,c2,c3,c4 = np.array([]),np.array([]),np.array([]),np.array([])
#final_coordinates = np.array([])
for contour_cordinates in contour_to_traverse :
    if contour_cordinates[0,0] == contour_cordinates[len(contour_cordinates)-1,0] and contour_cordinates[0,1] == contour_cordinates[len(contour_cordinates)-1,1] and len(contour_cordinates) > 3:
        new_sudoku = contour_cordinates.copy()
        appproximate_sudoku = measure.approximate_polygon(new_sudoku,tolerance=20.0)
        print(min(appproximate_sudoku[:,0]),max(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1]),max(appproximate_sudoku[:,1]))
        for row in appproximate_sudoku :
            print(row[0],row[1])
            print(min(appproximate_sudoku[:,0])-(original_height*0.06),min(appproximate_sudoku[:,0])+(original_height*0.06),min(appproximate_sudoku[:,1])-(original_width*0.06),min(appproximate_sudoku[:,1])+(original_width*0.06))
            if row[0] >= min(appproximate_sudoku[:,0])-(original_height*0.06) and row[0] <= min(appproximate_sudoku[:,0])+(original_height*0.06) and row[1] >= min(appproximate_sudoku[:,1])-(original_width*0.06) and row[1] <= min(appproximate_sudoku[:,1])+(original_width*0.06):
                c1 = np.array([row[0],row[1]])
                print('c1')
            print(min(appproximate_sudoku[:,0])-(original_height*0.06),min(appproximate_sudoku[:,0])+(original_height*0.06),max(appproximate_sudoku[:,1])-60,max(appproximate_sudoku[:,1])+(original_width*0.06))
            if row[0] >= min(appproximate_sudoku[:,0])-(original_height*0.06) and row[0] <= min(appproximate_sudoku[:,0])+(original_height*0.06) and row[1] >= max(appproximate_sudoku[:,1])-(original_width*0.06) and row[1] <= max(appproximate_sudoku[:,1])+(original_width*0.06):
                c2 = np.array([row[0],row[1]])
                print('c2')
            print(max(appproximate_sudoku[:,0])-(original_height*0.06),max(appproximate_sudoku[:,0])+(original_height*0.06),max(appproximate_sudoku[:,1])-60,max(appproximate_sudoku[:,1])+(original_width*0.06))
            if row[0] >= max(appproximate_sudoku[:,0])-(original_height*0.06) and row[0] <= max(appproximate_sudoku[:,0])+(original_height*0.06) and row[1] >= max(appproximate_sudoku[:,1])-(original_width*0.06) and row[1] <= max(appproximate_sudoku[:,1])+(original_width*0.06):
                c3 = np.array([row[0],row[1]])
                print('c3')
            print(max(appproximate_sudoku[:,0])-(original_height*0.06),max(appproximate_sudoku[:,0])+(original_height*0.06),min(appproximate_sudoku[:,1])-60,min(appproximate_sudoku[:,1])+(original_width*0.06))
            if row[0] >= max(appproximate_sudoku[:,0])-(original_height*0.06) and row[0] <= max(appproximate_sudoku[:,0])+(original_height*0.06) and row[1] >= min(appproximate_sudoku[:,1])-(original_width*0.06) and row[1] <= min(appproximate_sudoku[:,1])+(original_width*0.06):
                c4 = np.array([row[0],row[1]])
                print('c4')
        #print(np.array([min(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1])]))
        print(len(c2))
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

desired_matrix = np.fliplr(appproximate_sudoku[0:4])
#print(desired_matrix)
desired_matrix = desired_matrix[desired_matrix[:,1].argsort()]
#print(desired_matrix)
desired_matrix1 = desired_matrix[:2]
desired_matrix2 = desired_matrix[2:]
#print(desired_matrix1,desired_matrix2)
desired_matrix1 = desired_matrix1[desired_matrix1[:,0].argsort()]
desired_matrix2 = desired_matrix2[desired_matrix2[:,0].argsort()[::-1]]
#print(desired_matrix1,desired_matrix2)
desired_matrix = np.concatenate((desired_matrix1,desired_matrix2),axis=0)
print(desired_matrix)
print(np.sqrt(np.sum((desired_matrix[0]-desired_matrix[1])**2)))
print(np.sqrt(np.sum((desired_matrix[1]-desired_matrix[2])**2)))
print(np.sqrt(np.sum((desired_matrix[2]-desired_matrix[3])**2)))
print(np.sqrt(np.sum((desired_matrix[3]-desired_matrix[0])**2)))
cropped_height = np.sqrt(np.sum((desired_matrix[0]-desired_matrix[1])**2))
cropped_width = np.sqrt(np.sum((desired_matrix[2]-desired_matrix[3])**2))
print(calculate_area_proportion(original_height , original_width ,cropped_height , cropped_width))
if calculate_area_proportion(original_height , original_width ,cropped_height , cropped_width) < 20 :
    c1 = np.array([0,0])
    c2 = np.array([original_height,0])
    c3 = np.array([original_height,original_width])
    c4 = np.array([0,original_width])
    c5 = c1
    final_coordinates = np.array([c1,c2,c3,c4,c5])
    appproximate_sudoku = final_coordinates
    print(final_coordinates)
    #print(type(appproximate_sudoku))
    #print(contour_cordinates[0,:],min(appproximate_sudoku[:,0]),max(appproximate_sudoku[:,0]),min(appproximate_sudoku[:,1]),max(appproximate_sudoku[:,1]))
    axis.plot(appproximate_sudoku[:, 1], appproximate_sudoku[:, 0], linewidth=2)
    desired_matrix = appproximate_sudoku[0:4]
    print(desired_matrix)
    
for i in range(len(appproximate_sudoku)):
    axis.scatter(appproximate_sudoku[i][1],appproximate_sudoku[i][0])


# In[102]:


final_sudoku_size = np.array(((0,0),(270,0),(270,270),(0,270)))
transformation = tnf.ProjectiveTransform()
transformation_estimation = transformation.estimate(final_sudoku_size,desired_matrix)
print(transformation_estimation,transformation)
Wrapping = tnf.warp(filtered_image,transformation,output_shape=(270,270))
print(Wrapping,np.amax(Wrapping))
Wrapping = abs(Wrapping-np.amax(Wrapping))
print(Wrapping,np.amax(Wrapping))
plt.figure()
plt.imshow(Wrapping, cmap='gray')


# In[103]:


#print(Wrapping[0,:],Wrapping[-1,:],Wrapping[:,0],Wrapping[:,-1],len(Wrapping[0,:]))
reduce_x_0 , reduce_x_1 , reduce_y_0 , reduce_y_1 = 0,0,0,0
for x in range(int(len(Wrapping[0,:])/2)) :
    print(np.count_nonzero(Wrapping[x,:]==1),np.count_nonzero(Wrapping[-x-1,:]==1),np.count_nonzero(Wrapping[:,x]==1),np.count_nonzero(Wrapping[:,-x-1]==1),len(Wrapping[0,:]),int(len(Wrapping[0,:])/2))
    print(np.count_nonzero(Wrapping[x,:]==0),np.count_nonzero(Wrapping[-x-1,:]==0),np.count_nonzero(Wrapping[:,x]==0),np.count_nonzero(Wrapping[:,-x-1]==0),len(Wrapping[0,:]),int(len(Wrapping[0,:])/2))
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
            final_sudoku_size = np.array(((0,0),(270,0),(270,270),(0,270)))
            transformation = tnf.ProjectiveTransform()
            transformation_estimation = transformation.estimate(final_sudoku_size,desired_matrix)
            print(transformation_estimation,transformation)
            Wrapping = tnf.warp(Wrapping,transformation,output_shape=(270,270))
        break;
plt.figure()
print(reduce)
plt.imshow(Wrapping, cmap='gray')


# In[104]:


selem = square(1)
print(selem)
print(closing(Wrapping,selem))
plt.figure()
plt.imshow(closing(Wrapping,selem), cmap='gray')
final_wrapped_image = closing(Wrapping,selem)


# In[105]:


plt.figure()
plt.imshow(binary_dilation(Wrapping,selem), cmap='gray')
#final_wrapped_image = binary_dilation(Wrapping,selem)


# In[106]:


#selem=disk(2)
plt.figure()
plt.imshow(closing(erosion(Wrapping,selem),square(4)), cmap='gray')
#plt.imshow(dilation(Wrapping,selem), cmap='gray')
#final_wrapped_image = erosion(Wrapping,selem)


# In[107]:


Sudoku = np.zeros([9,9],dtype=int)


# In[108]:


def boundary_removal( input_image ) :
    height,width = input_image.shape
    input_image[:3,:] = 0
    input_image[:,:3] = 0
    input_image[height-3:,:] = 0
    input_image[:,width-3:] = 0
    return input_image


# In[109]:


#plt.imshow(cell_dimensions, cmap='gray')


# In[110]:


def digit_encoder() :
    pretrained_weights = torch.load('/storage/Suduko_solver_data/model_2.pth')
    pretrained_model = cnn_model()
    pretrained_model.load_state_dict(pretrained_weights)
    pretrained_model.eval()
    height , weight = final_wrapped_image.shape
    print(height , weight)
    for x in range(9) :
        for y in range(9) :
            #print((x*int(height/9))+1,((x+1)*int(height/9))-1,(y*int(weight/9))+1 ,((y+1)*int(weight/9))-1)
            cell_dimensions = final_wrapped_image[(x*int(height/9))+1 : ((x+1)*int(height/9))-1,(y*int(weight/9))+1 : ((y+1)*int(weight/9))-1]
            plt.figure()
            plt.imshow(cell_dimensions, cmap='gray')
            cell_dimensions = boundary_removal(cell_dimensions)
            plt.figure()
            plt.imshow(cell_dimensions, cmap='gray')
            Zero_digit_check = np.count_nonzero(cell_dimensions==0)
            #print(Zero_digit_check)
            cell_dimensions=np.expand_dims(np.expand_dims(cell_dimensions, axis=0), axis=0)
            cell_dimensions = cell_dimensions.astype(float)
            #print(cell_dimensions.shape)
            #print(cell_dimensions)
            #test_dataset= torchvision.datasets.Image(cell_dimensions,transform=Image_transformation_function)
            test_dataset = torch.from_numpy(cell_dimensions) #Image_transformation_function(cell_dimensions)
            test_dataset.to(device_to_use)
            #training_data_loader = torch.utils.data.DataLoader(MNIST_dataset_training,batch_size=64,shuffle=True)
            predicted_output = pretrained_model(test_dataset.float())
            print(predicted_output)
            predicted_digit = predicted_output.data.max(1,keepdim=True)[1]
            plt.title('Expected Digit - '+str(predicted_digit))
            #print(predicted_digit,predicted_output)
            if predicted_digit == 1 and Zero_digit_check > 760 :
                Sudoku[x][y] = 0
            else :
                Sudoku[x][y] = predicted_digit
            #print(predicted_digit)


# In[111]:


digit_encoder()