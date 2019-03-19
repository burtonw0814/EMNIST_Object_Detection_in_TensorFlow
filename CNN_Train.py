# Object detection CNN, similar to https://arxiv.org/pdf/1701.06659.pdf

import numpy as np
import tensorflow as tf
import glob
import cv2
import random
import os
import os.path
import csv

from skimage import io
from skimage import transform as sktf
from skimage import data, color
from skimage.transform import rescale, resize
import time
from skimage import exposure

tf.__version__

def get_jaccard(box_height, box_width, anchor_height, anchor_width):
    inte=np.amin((box_height,anchor_height))*np.amin((box_width,anchor_width))
    union=box_height*box_width +  anchor_height*anchor_width - inte
    return inte/(union+1e-8)

# Get box ground for object detection component
def get_box_ground(im, _box, pc1, pc2, num_anchor_boxes, im_path, pw, ph, num_grid_cells1, num_grid_cells2, gc0, gc1, num_classes):
    
    ft_map1=[]; ft_map2=[]; ft_map3=[]; ft_map4=[]; lambda1=[]; lambda2=[];
    for j in range(len(pw)):
        ft_map1.append(np.zeros((num_grid_cells1[j],num_grid_cells2[j],num_anchor_boxes*2)))
        ft_map2.append(np.zeros((num_grid_cells1[j],num_grid_cells2[j],num_anchor_boxes*2)))
        ft_map3.append(np.zeros((num_grid_cells1[j],num_grid_cells2[j],num_anchor_boxes*1)))
        ft_map4.append(np.zeros((num_grid_cells1[j],num_grid_cells2[j],num_classes*num_anchor_boxes)))
        lambda1.append(np.ones((ft_map1[j].shape[0],ft_map1[j].shape[1],num_anchor_boxes))) # box present mask
        lambda2.append(np.ones((ft_map1[j].shape[0],ft_map1[j].shape[1],num_anchor_boxes))) # box not present mask
        cache=[];

    # Get all ground truth bounding boxes, consider multiple anchor boxes
    reduct=(2**5)
    #print(_box)
     
    if not _box: # If no boxes, return grid cells with all zeros, and corresponding loss masks
        lambda1=[];
        for j in range(len(pw)):
            lambda1.append(np.zeros((ft_map1[j].shape[0],ft_map1[j].shape[1],num_anchor_boxes))) # box not present mask
  
    else:
        # REMEMBER _BOX[i] is [class, x1, x2, y1, y2]
        for k in range(len(_box)): # Each box

            box=_box[k]
            ht=im.shape[0]
            wt=im.shape[1]
            
            #Need to figure out which scale to assign the ground to
            b_height=np.float32(box[4])-np.float32(box[3]); b_width=np.float32(box[2])-np.float32(box[1]); # Dims of box in %
            ref_val=np.amax((b_height, b_width)) # Take max
            jaccard_list=[];
            for kk in range(len(pw)):
                j_list=[];
                for jj in range(len(pw[kk])):
                    j_list.append(get_jaccard(b_height, b_width, ph[kk][jj], pw[kk][jj]))
                jaccard_list.append(j_list)
            #print(jaccard_list)
			
            g_idx_list=[];
            #print(np.argmax(jaccard_list))
            closest=(np.argmax(jaccard_list)) # scale then anchor box at that scale
            #print(int(closest/num_anchor_boxes), (closest%num_anchor_boxes))
            g_idx_list.append((int(closest/num_anchor_boxes), (closest%num_anchor_boxes))) # Get anchor box with maximum jaccard index
            #print(g_idx_list)
            for kk in range(len(jaccard_list)): # Then assign other anchor boxes with jaccard >0.5
                for jj in range(len(jaccard_list[kk])):
                    if (kk!=g_idx_list[0][0] or jj!=g_idx_list[0][1]) and jaccard_list[kk][jj]>0.5:
                        g_idx_list.append((kk,jj))
					
            for kk in range(len(g_idx_list)):
                g_idx=g_idx_list[kk][0] # Which scale?
                anch_idx=g_idx_list[kk][1] # Which anchor box at that scale?
                #print(g_idx, anch_idx)
                # box is [object class, h1, h2, v1, v2] in percentages of image dims (0-1)

                # print(box) # Find center of bounding box in [%width, %height] of original image
                center=[];
                center.append((np.float32(box[1])+np.float32(box[2]))/2) # Center x coord in %
                center.append((np.float32(box[3])+np.float32(box[4]))/2) # Center y coord in %
                #print(center)

                # Determine grid cell coordinates [x,y] that ground truth belongs in
                cell_coord=(int(center[0]*num_grid_cells2[g_idx]), int(center[1]*num_grid_cells1[g_idx])) # Percentage times num grid cells in right box, floored

                # Determine ground truth box parameters according to format of YOLOV2 paper
                bx=center[0]*num_grid_cells2[g_idx]-cell_coord[0]
                by=center[1]*num_grid_cells1[g_idx]-cell_coord[1]
                bw=np.log(((np.float32(box[2])-np.float32(box[1]))/(pw[g_idx][anch_idx]+ 1e-10))+1e-10) # log of ratio between bounding box dims and anchor box dims
                bh=np.log(((np.float32(box[4])-np.float32(box[3]))/(ph[g_idx][anch_idx]+ 1e-10))+1e-10)
                if np.isnan(bw) or np.isnan(bw) or np.isinf(bw) or np.isinf(bw):
                    print('WANRIASGFDREWQSAF KHAAAAAAAAAAAAAAAAAAAN CQAWEREWRVCERGV')
                    print(box, pw, ph, im_path)
                #print(bx, by, bw, bh)
            
                # ASSIGN TO CORRECT ANCHOR BOX AT APPROPRIATE SCALE
                if 1==1: #for iiii in range(num_anchor_boxes):
                    if ft_map3[g_idx][cell_coord[1], cell_coord[0], anch_idx]==0: # Make sure anchor box is available at that scale
                        # Populate feature map --> of desired anchor box
                        ft_map1[g_idx][cell_coord[1],cell_coord[0],0+(2*anch_idx)]=bx
                        ft_map1[g_idx][cell_coord[1],cell_coord[0],1+(2*anch_idx)]=by
                        ft_map2[g_idx][cell_coord[1],cell_coord[0],0+(2*anch_idx)]=bw
                        ft_map2[g_idx][cell_coord[1],cell_coord[0],1+(2*anch_idx)]=bh
                        ft_map3[g_idx][cell_coord[1],cell_coord[0],0+anch_idx]=1 # Objectness score should be 1 because box is present
                        ft_map4[g_idx][cell_coord[1],cell_coord[0], num_classes*anch_idx + int(box[0])]=1 # Assign class
                        break # Stop looking for empty anchor boxes if we already found one

        #lambda1=lambda1*np.expand_dims(ft_map[:,:,1],axis=-1); # Initialized mask is multiplied by either 0 or 1 based on grounf truth objectness score
        for jjj in range(len(lambda1)):
            lambda1[jjj]=lambda1[jjj]*ft_map3[jjj]; # Initialized mask is multiplied by either 0 or 1 based on ground truth objectness score
            lambda2[jjj]=lambda2[jjj]-lambda1[jjj]

        #cache=((bx, by, bw, bh, cell_coord[0], cell_coord[1], pw, ph))

    return ft_map1, ft_map2, ft_map3, ft_map4, lambda1, lambda2


# Get heatmap ground truth for keypoint detection compeonnt o
def get_ground(im_path,  box_path, pc1, pc2, num_anchor_boxes, pw, ph, num_grid_cells1, num_grid_cells2, gc0, gc1, num_classes, train_mode=True):
    
    im=cv2.imread(im_path, cv2.IMREAD_COLOR)
    if train_mode==True:
        blur_param=int(np.random.uniform(low=0,high=1))
        if blur_param>0.5:
            s_param=np.random.uniform(low=0.2,high=10)
            im=skimage.filters.gaussian(im, sigma=s_param);
    im_4_show=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    mean_vec=(104, 117, 124)
    im=im.astype(np.float16)
    for j in range(3):
        im[:,:,j]=im[:,:,j]-mean_vec[j] # Subtract channel-wise means
     
    _box=[]; 
	
    # Import box 
    with open(box_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            _box.append(row)
    im_re=resize(im, (pc1, pc2), preserve_range=True)
    im_4_show=resize(im_4_show, (pc1, pc2), preserve_range=True).astype(np.uint8)

    ####################### BOX GROUND #####################################################################
    ft_map1, ft_map2, ft_map3, ft_map4,  lambda1, lambda2 = get_box_ground(im, _box, pc1, pc2, num_anchor_boxes, im_path, pw, ph, num_grid_cells1, num_grid_cells2, gc0, gc1, num_classes)
   
    #print(len(ft_map1), len(ft_map2), len(ft_map3),  len(lambda1), len(lambda2))
    ########################################################################################################      
    return im_re, im_4_show, ft_map1, ft_map2, ft_map3, ft_map4,  lambda1, lambda2, []

def color_augment(x_new):
    
    # im and gr are both 3dim tensors of [height, width, depth]
    contrast_flag1=np.random.binomial(1,0.3,1)
    if contrast_flag1==1:
        # Get random params for contrast and gamma correction
        LB=np.random.uniform(low=0, high=25)
        UB=np.random.uniform(low=75, high=100)
        v_min, v_max = np.percentile(x_new, (LB, UB))
        x_new = exposure.rescale_intensity(x_new, in_range=(v_min, v_max))
        
    contrast_flag2=np.random.binomial(1,0.4,1)
    if contrast_flag2==1:
        # gamma and gain parameters are between 0 and 1
        G=np.random.uniform(low=0.85, high=1)
        x_new = exposure.adjust_gamma(x_new, gamma=G, gain=1)
        
    return x_new

def resize_function(input_image, ground_truth_image, pc1, pc2):
    temp_in = resize(input_image, (pc1, pc2), preserve_range=True)
    temp_out = np.round(resize(ground_truth_image, (pc1, pc2), preserve_range=True))
    return temp_in, temp_out

def resize_function2(input_image, pc1, pc2):
    temp_in = resize(input_image, (pc1, pc2), preserve_range=True)
    return temp_in

def random_files(num_files):
    x = ([[i+1] for i in range(num_files)])
    shuflist=random.sample(x,len(x))
    list_files=[]
    s=''
    for i in range(num_files):
        ID=str(shuflist[i][0])
        while len(ID)<6:
            ID='0'+ID
        list_files.append(ID)
            
    return list_files
    
#LeakyReLU activation
def leakyrelu(x, alpha=0.2):
    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)

# Layer wrappers
def conv_layer(inputs, is_training, channels_in, channels_out, stvs, strides=1, scopename="Conv", f_size=3):
    with tf.name_scope(scopename):
        
        s=''
        weightname=(scopename,'_weights')
        biasname=(scopename,'_bias')
        
        w=tf.get_variable(name=s.join(weightname), initializer=tf.random_normal([f_size, f_size, channels_in, channels_out],stddev=stvs))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)

        tf.summary.histogram(("Weight" + scopename),w)

        x = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME')
        x=IN(x, scopename=scopename)#x=tf.layers.batch_normalization(x, training=is_training)
        return tf.nn.elu(x)
        
def maxpool2d(x, k=2, scopename="Pool"):
    with tf.name_scope(scopename):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='VALID')

def upconv2d(x, channels_in, channels_out, stvs, stride=2, k=2, scopename="Upconv"):
    with tf.name_scope(scopename):
        w=tf.get_variable(name=scopename+'_up_w', initializer=tf.random_normal([k, k, channels_out, channels_in],stddev=stvs))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
        
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, channels_out]) # [BS doubleheight doubl width  halvedepth]
        return tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='SAME')

def concatenate(in1, in2, scopename="Concat"):
    with tf.name_scope(scopename):
        return tf.concat([in1, in2], 3)
    
def residual(inputs, is_training, channels_in, channels_out, scopename, f_size=3):
    with tf.name_scope(scopename):
        stvs=0.01

        x=conv_layer(inputs, is_training, channels_in, channels_in, stvs, f_size=f_size, scopename=scopename+'_1')
        
        x=linear_conv_layer(x,channels_in, channels_out, stvs, f_size=f_size, use_bias=False, scopename=scopename+'_2')
        #x=tf.layers.batch_normalization(x, training=is_training)
        x=IN(x, scopename=scopename)#x=tf.contrib.layers.layer_norm(x)
        
        if channels_in==channels_out:
            skip_tensor=inputs
        else: 
            skip_tensor=linear_conv_layer(inputs, channels_in, channels_out, stvs, f_size=1, scopename=scopename+'_3')

        return tf.nn.elu(x+skip_tensor)
    
# Layer wrappers
def linear_conv_layer(inputs, channels_in, channels_out, stvs, strides=1, scopename="Conv", f_size=3, use_bias=True, dil=1):
    with tf.name_scope(scopename):
        s=''
        weightname=(scopename,'_weights')
        biasname=(scopename,'_bias')
        
        w=tf.get_variable(name=s.join(weightname), initializer=tf.random_normal([f_size, f_size, channels_in, channels_out],stddev=stvs))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
        
        x = tf.nn.conv2d(inputs, w, strides=[1, strides, strides, 1], padding='SAME',dilations=[1,1,dil,dil])
        
        if use_bias==True:
            b=tf.get_variable(name=s.join(biasname), initializer=tf.random_normal([channels_out],stddev=stvs))
            x = tf.nn.bias_add(x, b)
        
        return x

def ASPP(x, is_training, c_in, c_out, d_start, stvs, scopename='ASPP'):

    x1=linear_conv_layer(x, c_in, int(c_out/4), stvs, scopename=scopename+'1')
    x2=linear_conv_layer(x, c_in, int(c_out/4), stvs, dil=2, scopename=scopename+'2')
    x3=linear_conv_layer(x, c_in, int(c_out/4), stvs, dil=4, scopename=scopename+'3')
    x4=linear_conv_layer(x, c_in, int(c_out/4), stvs, dil=6, scopename=scopename+'4')

    x=tf.concat([x1,x2,x3,x4], axis=-1)
    x=conv_layer(x,is_training, int(c_out/4)*4, c_out, stvs, scopename=scopename+'5')
    
    return x

def IN(x, scopename='IN_LAYER_'):
    s_name=scopename+'_s';
    b_name=scopename+'_b';

    epsilon = 1e-3
    scale = tf.get_variable(name=s_name, initializer=tf.ones([x.get_shape()[-1]]))
    beta = tf.get_variable(name=b_name, initializer=tf.zeros([x.get_shape()[-1]]))
    batch_mean, batch_var = tf.nn.moments(x,[1,2],keep_dims=True)
    x=tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)
    
    return x

def hourglass(x,  is_training, inputdepth,  depthstart, num_anchor_boxes, num_classes, scopename, re, getter): 
    with tf.variable_scope('Model', reuse=re, custom_getter=getter):
        stvs=0.005; FSIZE=3

        if 1==1:
            conv1=residual(x, is_training, 3, depthstart, f_size=FSIZE, scopename="conv1")
            pool1=maxpool2d(conv1,scopename="Pool1")

            conv2=residual(pool1, is_training, depthstart, depthstart*2, f_size=FSIZE, scopename="conv2")
            pool2=maxpool2d(conv2,scopename="Pool2")

            conv3=residual(pool2, is_training, depthstart*2, depthstart*4, f_size=FSIZE, scopename="conv3")
            pool3=maxpool2d(conv3,scopename="Pool3")

            conv4=residual(pool3, is_training, depthstart*4, depthstart*8, f_size=FSIZE, scopename="conv4")
            pool4=maxpool2d(conv4,scopename="Pool4")

            conv5=residual(pool4, is_training, depthstart*8, depthstart*16, f_size=FSIZE, scopename="conv5")
            pool5=maxpool2d(conv5,scopename="Pool4")
            throat_dim=16
            #throat_conv1=residual(pool5, is_training, depthstart*16, depthstart*throat_dim, scopename="throat_res1")
            throat_conv1=ASPP(pool5,is_training, depthstart*16, depthstart*throat_dim, depthstart, stvs, scopename='ASPP1')


            # TO DO: VERIFY FILTER DEPTH
            # Get grid cell output for bounding box detection
            branch1_t=residual(throat_conv1, is_training, depthstart*throat_dim, depthstart, f_size=3, scopename="KHAN1")
            map1a = tf.nn.sigmoid(linear_conv_layer(branch1_t, depthstart, 2*num_anchor_boxes, stvs/1, f_size=1, scopename='L1a')+1e-8)# x and y center in % of grid cell
            map2a = linear_conv_layer(branch1_t, depthstart, 2*num_anchor_boxes, stvs, f_size=1, scopename='L2a')+ 1e-10 # height and width
            map3a =  tf.nn.sigmoid(linear_conv_layer(branch1_t, depthstart, 1*num_anchor_boxes, stvs/1, f_size=1, scopename='L3a')+1e-8)# Objectness score
            map4a=[];
            for j in range(num_anchor_boxes): # 1 softmax layer for each anchor box
                map4a.append(tf.nn.softmax(linear_conv_layer(branch1_t, depthstart, num_classes, stvs/1, f_size=1, scopename=('L4a'+str(j)))+1e-8))
            map4a=tf.concat(map4a, axis=-1)

            # Begin upsampling
            upconv1=upconv2d(throat_conv1, depthstart*throat_dim, depthstart*16, stvs, scopename="pupconv1")
            conc1=conv5+upconv1
            pconv6a = residual(conc1, is_training, depthstart*16, depthstart*8, f_size=FSIZE, scopename="pconv6a")
            branch2_t=residual(pconv6a, is_training, depthstart*8, depthstart, f_size=3, scopename="KHAN2")
            map1b = tf.nn.sigmoid(linear_conv_layer(branch2_t, depthstart, 2*num_anchor_boxes, stvs/1, f_size=1, scopename='L1b')+1e-8)# x and y center in % of grid cell
            map2b = linear_conv_layer(branch2_t, depthstart, 2*num_anchor_boxes, stvs, f_size=1, scopename='L2b')+ 1e-10 # height and width
            map3b =  tf.nn.sigmoid(linear_conv_layer(branch2_t, depthstart, 1*num_anchor_boxes, stvs/1, f_size=1, scopename='L3b')+1e-8)# Objectness score
            map4b=[];
            for j in range(num_anchor_boxes): # 1 softmax layer for each anchor box
                map4b.append(tf.nn.softmax(linear_conv_layer(branch2_t, depthstart, num_classes, stvs/1, f_size=1, scopename=('L4b'+str(j)))+1e-8))
            map4b=tf.concat(map4b, axis=-1)
				
            cache_l1=(map1a, map1b);
            cache_l2=(map2a, map2b);
            cache_l3=(map3a, map3b);
            cache_l4=(map4a, map4b);
    if re!=True:
        return cache_l1, cache_l2, cache_l3 , cache_l4    
    else:
        return cache_l1, cache_l2, cache_l3, cache_l4      # If EMA pass, only return logits

# Definition of network
def conv_net(X, is_training, inputdepth, num_anchor_boxes, num_classes, re=None, getter=None):

    depth_start=64
    cache_l1, cache_l2, cache_l3, cache_l4 = hourglass(X, is_training, inputdepth, depth_start, num_anchor_boxes, num_classes, 'Hourglass1', re=re, getter=getter)

    return  cache_l1, cache_l2, cache_l3, cache_l4    

def box_loss_function(y1, y2, y3, y4, l1, l2, l3, l4, lambda1, lambda2, lambda1d, lambda2d, lambda1c):
    lambda_coord=10
    lambda_no_obj=0.5
    
    # pt 1 -- box center location for grid cells containing boxes
    # Lambda1 must be a [batch, height, width, num_anchor_boxes] mask with all zeros except where a bounding box occurs in ground truth
    # l1 must be passed through a sigmoid because it hasn't been scaled yet
    j1=lambda_coord*tf.reduce_sum(lambda1d*(tf.square(tf.subtract(l1,y1))))
    
    # pt 2 -- box dimensions --> This part is ambiguous b/c yolo2 doesnt define loss function but the ...
    # prediction method for these vals are different from yolo1
    # Just do tw and tw squared-error IN log space
    # Again, here we apply the lambda1 mask that includes the inidicator function and the loss weight as described in YOLO1
    j2=lambda_coord*tf.reduce_sum(lambda1d*tf.square(tf.subtract(l2,y2)))
    
    # pt 3 -- loss for objectness score for   featuremaps where box is present in ground truth
    j3=2*tf.reduce_sum(lambda1*tf.square(tf.subtract(l3,y3)))
    
    # pt 4 -- loss for objectness score for featuremaps where box is NOT present in ground truth
    j4=lambda_no_obj*tf.reduce_sum(lambda2*tf.square(tf.subtract(l3,y3)))

    j5=lambda_coord*tf.reduce_sum(lambda1c*(tf.square(tf.subtract(l4,y4))))

    return j1+j2+j3+j4+j5, j1, j2, j3, j4, j5


def build_graph(input_depth,  num_grid_cells1, num_grid_cells2, num_anchor_boxes, num_classes):
    
    ## Define placeholders
    is_training = tf.placeholder(tf.bool) 
    X=tf.placeholder(tf.float32, [None, None, None, input_depth])

    # Place holders for grid cell values
    with tf.name_scope("Box_Ground"):
        Y_b1a=tf.placeholder(tf.float32, [None, None, None, 2*num_anchor_boxes])
        Y_b2a=tf.placeholder(tf.float32, [None, None, None, 2*num_anchor_boxes])
        Y_b3a=tf.placeholder(tf.float32, [None, None, None, 1*num_anchor_boxes])
        Y_b4a=tf.placeholder(tf.float32, [None, None, None, num_anchor_boxes*num_classes])

        Y_b1b=tf.placeholder(tf.float32, [None, None, None, 2*num_anchor_boxes])
        Y_b2b=tf.placeholder(tf.float32, [None, None, None, 2*num_anchor_boxes])
        Y_b3b=tf.placeholder(tf.float32, [None, None, None, 1*num_anchor_boxes])
        Y_b4b=tf.placeholder(tf.float32, [None, None, None, num_anchor_boxes*num_classes])


    with tf.name_scope("Loss_Weights"):
        lambda1a=tf.placeholder(tf.float32, [None, num_grid_cells1[0], num_grid_cells2[0], num_anchor_boxes]) # Box present mask
        lambda2a=tf.placeholder(tf.float32, [None, num_grid_cells1[0], num_grid_cells2[0], num_anchor_boxes]) # Box not-present mask
        lambda1da=tf.placeholder(tf.float32, [None, num_grid_cells1[0], num_grid_cells2[0], 2*num_anchor_boxes]) # Box present mask
        lambda2da=tf.placeholder(tf.float32, [None, num_grid_cells1[0], num_grid_cells2[0], 2*num_anchor_boxes]) # Box not-present mask
        lambda1ca=tf.placeholder(tf.float32, [None, num_grid_cells1[0], num_grid_cells2[0], num_classes*num_anchor_boxes]) # Box present mask

        lambda1b=tf.placeholder(tf.float32, [None, num_grid_cells1[1], num_grid_cells2[1], num_anchor_boxes]) # Box present mask
        lambda2b=tf.placeholder(tf.float32, [None, num_grid_cells1[1], num_grid_cells2[1], num_anchor_boxes]) # Box not-present mask
        lambda1db=tf.placeholder(tf.float32, [None, num_grid_cells1[1], num_grid_cells2[1], 2*num_anchor_boxes]) # Box present mask
        lambda2db=tf.placeholder(tf.float32, [None, num_grid_cells1[1], num_grid_cells2[1], 2*num_anchor_boxes]) # Box not-present mask
        lambda1cb=tf.placeholder(tf.float32, [None, num_grid_cells1[1], num_grid_cells2[1], num_classes*num_anchor_boxes]) # Box present mask
        
    # Define flow graph
    cache_l1, cache_l2, cache_l3, cache_l4  =  conv_net(X, is_training, input_depth,  num_anchor_boxes, num_classes)
	
    # Aggregate CNN outputs
    cache_y1=(Y_b1a, Y_b1b);
    cache_y2=(Y_b2a, Y_b2b);
    cache_y3=(Y_b3a, Y_b3b);
    cache_y4=(Y_b4a, Y_b4b);
    cache_lambda1=(lambda1a,lambda1b);
    cache_lambda2=(lambda2a,lambda2b);
    cache_lambda1d=(lambda1da,lambda1db);
    cache_lambda2d=(lambda2da,lambda2db);
    cache_lambda1c=(lambda1ca,lambda1cb);


    # Define Loss
    with tf.name_scope("Loss"):	
        box_loss=0
        bl_list=[];
        box_loss_weights=(1,1)
        for j in range(len(cache_y1)): # Iterate over scales
            box_loss1, j1, j2, j3, j4, j5 = box_loss_function(cache_y1[j], cache_y2[j], cache_y3[j], cache_y4[j], cache_l1[j], cache_l2[j], cache_l3[j], cache_l4[j], cache_lambda1[j], cache_lambda2[j], cache_lambda1d[j], cache_lambda2d[j], cache_lambda1c[j])
            box_loss=box_loss+(box_loss_weights[j]*box_loss1)
            bl_list.append(box_loss1)
		    
            tf.summary.scalar(("BoxLoss1_" + str(j)), box_loss1)
            tf.summary.scalar(("j1_" + str(j)),j1)
            tf.summary.scalar(("j2_" + str(j)),j2)
            tf.summary.scalar(("j3_" + str(j)),j3)
            tf.summary.scalar(("j4_" + str(j)),j4)
            tf.summary.scalar(("j5_" + str(j)),j5)
			
        tf.summary.scalar("BoxLoss1",box_loss)

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.00001) # BEST SO FAR 0.0001
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
        loss_op = reg_term + box_loss
            
        tf.summary.scalar("TotalLoss", loss_op)
        tf.summary.scalar("Reg_Loss", reg_term )
            
    # Define optimizer
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = tf.placeholder(tf.float32)#0.000005
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 50000, 0.999, staircase=True) 
        #with tf.control_dependencies([ema_op]):
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, global_step=global_step)  

        tf.summary.scalar("Step", global_step)
        tf.summary.scalar("Learning_Rate", learning_rate )
            
    # Define writer for Tensorboard
    for j in range(len(cache_l1)):
        l3_split=tf.split(cache_l3[j], num_anchor_boxes, axis=-1)
        y3_split=tf.split(cache_y3[j], num_anchor_boxes, axis=-1)
        for kk in range(num_anchor_boxes):
            tf.summary.image(('L3_' + str(j) + 'ANCHOR' + str(kk)), tf.cast(l3_split[kk],tf.float32), input_depth)
            tf.summary.image(('Y_b3_' + str(j)+ 'ANCHOR' + str(kk)), tf.cast(y3_split[kk],tf.float32), input_depth)

    X_4_SHOW = tf.placeholder(tf.float32, [None, None, None, 3])
    tf.summary.image('Image', tf.cast(X_4_SHOW,tf.float32), input_depth) 
    writer=tf.summary.FileWriter("./TB/1")
    summ=tf.summary.merge_all()

    # Initialize the variables
    init = tf.global_variables_initializer() 

    #Define saver for model saver
    saver = tf.train.Saver(max_to_keep=1)
    
    return X, X_4_SHOW, loss_op, cache_l1, cache_l2, cache_l3,  cache_l4, cache_y1, cache_y2, cache_y3, cache_y4, cache_lambda1, cache_lambda2, cache_lambda1d, cache_lambda2d, cache_lambda1c, train_op, writer,  summ, init, saver, is_training, starter_learning_rate, box_loss, bl_list



######## START ################

num_epochs = 100000
batch_size =2
BS=batch_size
input_depth = 3
model_depth= 5
# Number of anchor boxes at each scale
num_classes=47
reduct=(2**model_depth)

display_step = 500
save_step = 2000
pc1=224; pc2=288;

# LISTS BECAUSE MULTIPLE BOUNDING BOX LOGS
num_grid_cells1=(7,14);
num_grid_cells2=(9,18);
gc0=(int(pc1/num_grid_cells1[0]),int(pc1/num_grid_cells1[1])) # Number of pixels per grid cell
gc1=(int(pc2/num_grid_cells2[0]),int(pc2/num_grid_cells2[1]))
# DEFINE ANCHOR BOXES
pw=((0.8,0.8,0.6,0.6,0.8,0.4,0.4,0.4, 0.3, 0.3, 0.4, 0.2,0.8,0.2,0.6,0.2,0.4,0.5,0.6,0.1,0.1,0.2,0.2),(0.8,0.8,0.6,0.6,0.8,0.4,0.4,0.4, 0.3, 0.3, 0.4, 0.2,0.8,0.2,0.6,0.2,0.4,0.5,0.6,0.1,0.1,0.2,0.2)) # Anchor box widths at each scale,
ph=((0.8,0.6,0.8,0.6,0.4,0.8,0.4,0.3, 0.4, 0.3, 0.2, 0.4,0.2,0.8,0.2,0.6,0.6,0.5,0.4,0.1,0.2,0.1,0.2),(0.8,0.6,0.8,0.6,0.4,0.8,0.4,0.3, 0.4, 0.3, 0.2, 0.4,0.2,0.8,0.2,0.6,0.6,0.5,0.4,0.1,0.2,0.1,0.2)) # Anchor box heights at each scale
num_anchor_boxes = len(pw[0]) # Number of anchor boxes at each scale
print(num_grid_cells1)

tf.test.gpu_device_name()
tf.reset_default_graph()
X, X_4_SHOW, loss_op,  cache_l1, cache_l2, cache_l3, cache_l4, cache_y1, cache_y2, cache_y3, cache_y4, cache_lambda1, cache_lambda2, cache_lambda1d, cache_lambda2d, cache_lambda1c, train_op, writer,  summ, init, saver, is_training, LR,box_loss, bl_list = build_graph(input_depth,  num_grid_cells1, num_grid_cells2, num_anchor_boxes, num_classes)
top_path=os.getcwd()+'/' 
    
image_file='Images/'
box_file='Annotations_Simple/' 
num_files=int(len(glob.glob(top_path + box_file + "*.csv")))
print(num_files)

# Training session
ct=0
loss_vec=[];
with tf.Session() as sess:
    sess.run(init)
    #Option to restore from model checkpoint to resume training
    #saver.restore(sess, '/home/will/Desktop/CV2/Models_Scratch/Model_Scratch2/Scratch_175000_res_KP')
    
    writer.add_graph(sess.graph)
    for epoch in range(num_epochs):
        file_list=random_files(int(len(glob.glob(top_path + box_file + "*.csv")))) # Shuffle files for epoch
        i=0
        
        #print("EPOCH " + "{:d}".format(epoch))
        if epoch<1:
            LR_feed=0.0001
        else:
            LR_feed=0.00001
            
        while (i+BS)<len(file_list): 
            try: 
            
                bx=np.empty((batch_size,pc1,pc2,input_depth), dtype=np.float32)
                bx_4_show=np.empty((batch_size,pc1,pc2,input_depth), dtype=np.float32)

                batch_lamb1=[]; batch_lamb2=[]; batch_lamb1d=[]; batch_lamb2d=[]; batch_lamb1c=[]; batch_lamb2c=[]; by_b1=[]; by_b2=[]; by_b3=[];by_b4=[];
                for ik in range(len(num_grid_cells1)): # Each scale
                    batch_lamb1.append(np.empty((batch_size,num_grid_cells1[ik],num_grid_cells2[ik],num_anchor_boxes), dtype=np.float32)); # Masks for objectness
                    batch_lamb2.append(np.empty((batch_size,num_grid_cells1[ik],num_grid_cells2[ik],num_anchor_boxes), dtype=np.float32)); # Masks for objectness
                    batch_lamb1d.append(np.empty((batch_size,num_grid_cells1[ik],num_grid_cells2[ik],num_anchor_boxes*2), dtype=np.float32)); # Masks for coords
                    batch_lamb2d.append(np.empty((batch_size,num_grid_cells1[ik],num_grid_cells2[ik],num_anchor_boxes*2), dtype=np.float32)); # Masks for coords
                    batch_lamb1c.append(np.empty((batch_size,num_grid_cells1[ik],num_grid_cells2[ik],num_anchor_boxes*num_classes), dtype=np.float32));  # Masks for classification
                    by_b1.append(np.empty((batch_size,num_grid_cells1[ik],num_grid_cells2[ik],num_anchor_boxes*2), dtype=np.float32))
                    by_b2.append(np.empty((batch_size,num_grid_cells1[ik],num_grid_cells2[ik],num_anchor_boxes*2), dtype=np.float32))
                    by_b3.append(np.empty((batch_size,num_grid_cells1[ik],num_grid_cells2[ik],num_anchor_boxes), dtype=np.float32))
                    by_b4.append(np.empty((batch_size,num_grid_cells1[ik],num_grid_cells2[ik],num_classes*num_anchor_boxes), dtype=np.float32))
  

                for j in range(BS):
                    im_path=(top_path + image_file + '/' + file_list[i+j] +'.jpg')
                    box_path=(top_path +  box_file + '/' + file_list[i+j] +'.csv')
				
                    im, im_4_show,ft_map1, ft_map2, ft_map3, ft_map4, temp1, temp2, cache =  get_ground(im_path, box_path, pc1, pc2, num_anchor_boxes, pw, ph, num_grid_cells1, num_grid_cells2, gc0, gc1, num_classes)
                    for ik in range(len(num_grid_cells1)): # Each scale
                        batch_lamb1[ik][j,:,:,:]=temp1[ik]; batch_lamb2[ik][j,:,:,:]=temp2[ik];
                        by_b1[ik][j,:,:,:]=ft_map1[ik]; by_b2[ik][j,:,:,:]=ft_map2[ik]; by_b3[ik][j,:,:,:]=ft_map3[ik]; by_b4[ik][j,:,:,:]=ft_map4[ik]
                        for k in range(num_anchor_boxes): # Each anchor box at each scale
                            batch_lamb1d[ik][j,:,:,2*k]=batch_lamb1[ik][j,:,:,k]; batch_lamb1d[ik][j,:,:,2*k+1]=batch_lamb1[ik][j,:,:,k]
                            batch_lamb2d[ik][j,:,:,2*k]=batch_lamb2[ik][j,:,:,k]; batch_lamb2d[ik][j,:,:,2*k+1]=batch_lamb2[ik][j,:,:,k]
                            for lmao in range(num_classes): # Assign values to classification lambda masks
                                batch_lamb1c[ik][j,:,:,num_classes*k+lmao]=batch_lamb1[ik][j,:,:,k];
                    bx[j,:,:,:]=im; bx_4_show[j,:,:,:]=im_4_show; 
		
                if np.isnan(bx).any():
                    print('NAN INPUTS')   
                FD= { X: bx, X_4_SHOW: bx_4_show, is_training: True, LR: LR_feed, 
                cache_y1[0]: by_b1[0], cache_y1[1]: by_b1[1], 
                cache_y2[0]: by_b2[0], cache_y2[1]: by_b2[1], 
                cache_y3[0]: by_b3[0], cache_y3[1]: by_b3[1], 
                cache_y4[0]: by_b4[0], cache_y4[1]: by_b4[1], 
                cache_lambda1[0]: batch_lamb1[0], cache_lambda1[1]: batch_lamb1[1],
                cache_lambda2[0]: batch_lamb2[0], cache_lambda2[1]: batch_lamb2[1],
                cache_lambda1d[0]: batch_lamb1d[0], cache_lambda1d[1]: batch_lamb1d[1],
                cache_lambda2d[0]: batch_lamb2d[0], cache_lambda2d[1]: batch_lamb2d[1], 
                cache_lambda1c[0]: batch_lamb1c[0], cache_lambda1c[1]: batch_lamb1c[1],
                }

                _, loss, summary = sess.run([train_op, loss_op, summ], feed_dict=FD)
                loss_vec.append(loss)
                if ct==0:
                    print('ITERATION1')
                if ct%(BS*10)==0 or ct<200:
                    if ct > 100:
                        writer.add_summary(summary, ct)
                        
                i=i+BS
                ct=ct+BS

            except:
                print('Import Error'); time.sleep(0.3)
                        
                i=i+BS
                ct=ct+BS
            
            #Save point   
            model_ID=2
            if (ct*BS)%5000==0 or ct==BS:
                filelist_m = [ f for f in os.listdir(('./Model/')) ]
                for f in filelist_m:
                    os.remove(os.path.join(('./Model/'), f))
                s=''
                checkpointnamelist=('./Model/Scratch_',str(ct),'_')
                checkpointname= s.join(checkpointnamelist)
                save_path = saver.save(sess,checkpointname)
                print("Model saved in file: %s" % save_path)

print("We out here")



