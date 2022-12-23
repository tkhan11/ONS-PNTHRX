
from __future__ import division, print_function
import os, re, math, cmath
from glob import glob
import numpy as np
import cv2
from sklearn.preprocessing import binarize
import tensorflow as tf
from bayesian_unet_3d_kg import BayesUNet3DKG
# import imutils
import matplotlib.patches as patches
# from imutils import perspective
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd
import argparse

SLIDING_WINDOW = False

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def isleft(p1, p2, p3): #check if p3 is to the left of the line defined by p1 and p2
    p1, p2 = p2, p1
    position = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) < 0)
    return position

def update_map(img, p1, p2, p2_1, p2_2): #rescale the map and add 0.3 to the areas inside needle cone

    x1, y1 = p1
    x2, y2 = p2

    if p2_1[0] < p2_2[0]:
        p2_1, p2_2 = p2_2, p2_1 #p2_1 is the left line (downwards)

    mid_x, mid_y = int((x2 + x1)/2), int((y2 + y1)/2)

    img = img.astype(float) / 256.
    img *= 0.7 #scale pixel values
    updated_map = img

    for i in range(updated_map.shape[0]):
        for j in range(updated_map.shape[1]):   
            if not ((isleft((mid_x, mid_y), p2_2, (i, j)) and not isleft((mid_x, mid_y), p2_1, (i, j))) or (not isleft((mid_x, mid_y), p2_2, (i, j)) and isleft((mid_x, mid_y), p2_1, (i, j)))):
                updated_map[j,i]  = 0
                # updated_map[j,i] += 0.3
    return updated_map


def get_h5_file(parent_dir):
    dir_contents = os.listdir(parent_dir)
    assert len(dir_contents) == 1, "There is more than 1 file in this model folder path. Please reduce it down to just the h5 file."
    assert dir_contents[0].endswith('.h5'), "There is no h5 file stored in this model folder path. Please check again."
    return os.path.join(parent_dir, dir_contents[0])

def create_model(model_name, num_classes, depth, dropout=0.50, batch_norm=True, type='normal'):
    print("Creating model now")

    if type == 'normal':
        if model_name == '_Bayes3DUNetKG_':
            model = BayesUNet3DKG(depth=depth,
                                num_classes=num_classes,
                                batch_norm=batch_norm,
                                dropout=dropout, 
                                dropout_type='block',
                                regularizer=tf.keras.regularizers.L1L2(l1=0.00, l2=0.001))
        else:
            raise NotImplementedError(model_name + " Not implemented")

        print("Created model")
        return model
    elif type == 'tflite':
        return None # this function may be redundant, remove when refactoring

def preprocess_img(img):
    if len(img.shape) > 2:
        img_out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_out = img
    # img_out = cv2.resize(img_out, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
    # img_out = cv2.resize(img_out[55:1025,114:798], dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
    img_out = cv2.resize(img_out[41:255, 104:350], dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
    return img_out[:,:, np.newaxis].astype(np.float32)

def calculate_distance(needle, nerve, vessel, needle_tip, shape, type = 'centroid'):
    needle_thresh, nerve_thresh, vessel_thresh = 0, 0, 0 #minimum contour area 
    edged = (needle + nerve).astype(np.uint8)
    needle = needle.astype(np.uint8)
    nerve = nerve.astype(np.uint8)
    vessel = vessel.astype(np.uint8)
    # edged = np.expand_dims(edged, axis = 0)
    # edged = cv2.cvtColor(edged, cv2.COLOR_BGR2GRAY)
    cnts_edged, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts_needle, _ = cv2.findContours(needle.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts_nerve, _ = cv2.findContours(nerve.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts_nerve = sorted(cnts_nerve, key=cv2.contourArea)
    cnts_vessel, _ = cv2.findContours(vessel.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts_vessel = sorted(cnts_vessel, key=cv2.contourArea)
    # cnts_edged = imutils.grab_contours(cnts_edged)
    img = cv2.drawContours(edged, cnts_edged, -1, (0, 255, 0), 3)
    cv2.imwrite('/home/nthumbav/Downloads/temp.png', img)

    # for c in cnts:
	# # if the contour is not sufficiently large, ignore it
    #     if cv2.contourArea(c) < 100:
    #         continue
    min_dist = np.Inf

    if type == 'centroid':
        needle_centroids, nerve_centroids, vessel_centroids = [], [], []
        for c in cnts_needle:
            if cv2.contourArea(c) < needle_thresh:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            needle_centroids.append((cX, cY))
        
        try:
            needle_centroids = sorted(cnts_needle[-1], key = lambda y:y[0][1]) #sort by depth, taking the largest needle contour
            needle_centroids = [tuple(needle_centroids[-1][0])] #term centroid is misleading, this is needle tip 
        except:
            return np.Inf
        
        if needle_tip is not None:
            needle_centroids = (needle_tip)

        for c in cnts_nerve:
            if cv2.contourArea(c) < nerve_thresh:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            nerve_centroids.append((cX, cY))
        
        for c in cnts_vessel:
            if cv2.contourArea(c) < vessel_thresh:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            vessel_centroids.append((cX, cY))
        
        #Right now i'm considering only the nerve and vessel contours with the largest area

        for cent in needle_centroids:
            cent = np.array([cent])
            # cent = np.repeat(cent, len(nerve_centroids), axis = 0)
            try:
                nerve_centroid = nerve_centroids[-1] #largest area
                distances = np.linalg.norm(cent - nerve_centroid, axis = 1)
                # distances = np.linalg.norm(cent - nerve_centroids, axis = 1)               
                min_dist = min(np.amin(distances), min_dist)
            except:
                continue
        
        # min_dist = np.Inf #uncomment if you want only needle - vessel distance
        if min_dist == np.Inf:
            for cent in needle_centroids:
                cent = np.array([cent])
                try:
                    vessel_centroid = vessel_centroids[-1] #largest area
                    # cent = np.repeat(cent, len(vessel), axis = 0)
                    distances = np.linalg.norm(cent - vessel_centroid, axis = 1)
                    # distances = np.linalg.norm(cent - vessel_centroids, axis = 1)
                    min_dist = min(np.amin(distances), min_dist)
                except:
                    continue
    
    elif type == 'pointwise':
        for cnt_needle in cnts_needle:
            for cnt_nerve in cnts_nerve:
                cnt_needle, cnt_nerve = np.squeeze(cnt_needle), np.squeeze(cnt_nerve)
                for pt in range(cnt_needle.shape[0]):
                    needle_pt = np.expand_dims(cnt_needle[pt], axis = 0)
                    needle_pt = np.repeat(needle_pt, cnts_nerve.shape[0], axis = 0)
                    distances = np.linalg.norm(needle_pt - cnt_nerve, axis = 0)
                    min_dist = min(np.amin(distances), min_dist)

        if min_dist == np.Inf:
            for cnt_needle in cnts_needle:
                for cnt_nerve in cnts_vessel:
                    cnt_needle, cnt_vessel = np.squeeze(cnt_needle), np.squeeze(cnts_vessel)
                    for pt in range(cnt_needle.shape[0]):
                        needle_pt = np.expand_dims(cnt_needle[pt], axis = 0)
                        needle_pt = np.repeat(needle_pt, cnt_vessel.shape[0], axis = 0)
                        distances = np.linalg.norm(needle_pt - cnt_vessel, axis = 0)
                        min_dist = min(np.amin(distances), min_dist)


    return min_dist

def postprocess_img(img, minLineLength = 0, maxLineGap = 100):
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    lines = cv2.HoughLinesP(gray, 5, np.pi/180, 0, minLineLength, maxLineGap)
    needle_tip = None
    updated_map = img
    if lines is not None:
        for x1,y1,x2,y2 in lines[0]:
            mid_x, mid_y = int((x2 + x1)/2), int((y2 + y1)/2)
            angle = -int(math.atan((mid_y-y2)/(x2-mid_x))*180/math.pi)
            
            length = 10000 #some large number
            angle_1, angle_2 = math.radians(angle - 20), math.radians(angle + 20)
            # x2_2 = int(mid_x + length * math.cos(math.radians(angle_2)))
            # y2_2 = int(mid_y + length * math.sin(math.radians(angle_2)))
            # x2_1 = int(mid_x + length * math.cos(math.radians(angle_1)))
            # y2_1 = int(mid_y + length * math.sin(math.radians(angle_1)))
            pt1 = cmath.rect(length, angle_1)  
            x2_1 = int(pt1.real + mid_x  )
            y2_1 = int(pt1.imag + mid_y)
            
            pt2 = cmath.rect(length, angle_2)  
            x2_2 = int(pt2.real + mid_x)  
            y2_2 = int(pt2.imag + mid_y)

            updated_map = update_map(img, (x1, y1), (x2, y2), (x2_1, y2_1),(x2_2, y2_2))
            needle_tip = (x2, y2)
        
    return updated_map, needle_tip


def load_model(input_image_shape, n_input_frames, n_classes, depth, model_name, model_epoch, filepath, dropout=0.50, batch_norm=True, type = 'normal'):
    print("Loading Model")
    # model_path = os.path.join(os.path.dirname(os.path.abspath(filepath)), "model_weights")
    if type == 'normal':
        model_path = filepath
        model = create_model(model_name, n_classes, depth, dropout, batch_norm)
        if n_input_frames == 1:
            input_shape = (None, input_image_shape[0], input_image_shape[1], input_image_shape[2])
        else:
            input_shape = (None, n_input_frames, input_image_shape[0], input_image_shape[1], input_image_shape[2])
        model.build(input_shape) # need to call first before loading weights (for H5 format)
        model.load_weights(get_h5_file(os.path.join(model_path, model_epoch)), by_name=False) # must manually set names
    
    elif type == 'tflite':
        model = tf.lite.Interpreter('model.tflite')
        model.allocate_tensors()
        # Get input and output tensors.
        # input_details = model.get_input_details()
        # output_details = model.get_output_details()

        # Test the model on random input data.
        # input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        # model.set_tensor(input_details[0]['index'], input_data)

        # model.invoke()

        # output_data = model.get_tensor(output_details[0]['index'])

    
    return model

def perform_inference(input_list = None, model = None, model_name = None, n_mc_samples = None, is_training=True, type='normal'):
    if type == 'normal':
        """Can set to is_training to false if batch statistics have converged properly, otherwise true is better"""
        print("Predicting Output")
        epi_mean, epi_var = None, None
        if model_name == '_Bayes3DUNetKG_':
            sequence_input = np.expand_dims(np.stack(input_list, axis=0), axis=0)
            # epistemic uncertainty
            all_means = None
            for i in range(n_mc_samples):
                logits_mean, _, logits_logvar, _ = model(sequence_input, is_training=is_training, dropout_training=True)
                logits_mean = tf.expand_dims(logits_mean, 0)
                if all_means is None:
                    all_means = logits_mean
                else:
                    all_means = tf.concat([all_means, logits_mean], axis=0)
            epi_mean, epi_var = tf.nn.moments(all_means, axes=[0])
        else:
            raise NotImplementedError
        
    elif type == 'tflite':
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        sequence_input = np.expand_dims(np.stack(input_list, axis=0), axis=0)
        model.set_tensor(input_details[0]['index'], sequence_input.astype(np.float32))
        model.invoke()
        all_means = model.get_tensor(output_details[0]['index'])
        epi_mean, epi_var = tf.nn.moments(all_means, axes=[0])

    return epi_mean, epi_var


def main(filepath = '../3. 238 AC_Video 2.mp4', model_folder = 'bayes_3d_kg_pnb_e_5_l_bayesian_combined_kg_f_8_model_bayes_3d_unet_kg_4-26-2022-14-36'):
    model = create_model(num_classes= 4, depth = 4, model_name = '_Bayes3DUNetKG_', type = 'tflite')
    model = load_model(input_image_shape = (256,256,1), n_input_frames = 8, n_classes = 4, depth = 4, model_name = '_Bayes3DUNetKG_', model_epoch = '2_best_bn_t', filepath = model_folder, type = 'tflite')
    video_file = filepath
    images, count, total_count, image_numpy = [], 0, 0, np.zeros((8, 256, 256, 1))
    frames, needle_maps, nerve_maps, vessel_maps, needle_tips = [], [], [], [], []
    # parse out images from video
    vid_file = cv2.VideoCapture(video_file)
    success, image = vid_file.read()

    refine_maps = True

    while success:
        if not SLIDING_WINDOW:
            if count == 8:
                epi_mean, epi_var = perform_inference(image_numpy, model, '_Bayes3DUNetKG_', 1, type='tflite')
                # model.save('models')
                # exit()
                model_output = tf.nn.softmax(epi_mean).numpy()
                if len(model_output.shape) == 5:
                    needle_softmax = model_output[0,:,:,:,1]*255.
                    nerve_softmax = model_output[0,:,:,:,2]*255.
                    vessel_softmax = model_output[0,:,:,:,3]*255.
                else:
                    needle_softmax = model_output[:,:,:,1]*255.
                    nerve_softmax = model_output[:,:,:,2]*255.
                    vessel_softmax = model_output[:,:,:,3]*255.

                needle_softmax = needle_softmax.astype(np.uint8)
                nerve_softmax = nerve_softmax.astype(np.uint8)
                vessel_softmax = vessel_softmax.astype(np.uint8)
                for i in range(8):
                    needle_frame, nerve_frame, vessel_frame = needle_softmax[i,:,:], nerve_softmax[i,:,:], vessel_softmax[i,:,:]
                    updated_map, needle_tip = postprocess_img(needle_frame)
                    needle_frame = updated_map
                    # cv2.imwrite('/home/nthumbav/Downloads/temp/{}.png'.format(total_count), updated_map)
                    # needle_maps.append(updated_map) # comment out for raw outputs
                    if refine_maps:
                        try:
                            thresh = np.percentile(needle_frame[needle_frame > 0], 25)
                            needle_frame_binary = needle_frame
                            needle_frame_binary[needle_frame_binary <= thresh] = 0
                            needle_maps.append((needle_frame_binary* 255.).astype(np.uint8) )
                        except:
                            needle_maps.append(needle_frame)
                        try:
                            thresh = np.percentile(nerve_frame[nerve_frame > 0], 25)
                            nerve_frame_binary = nerve_frame
                            nerve_frame_binary[nerve_frame_binary <= thresh] = 0
                            nerve_maps.append((nerve_frame_binary* 255.).astype(np.uint8) )
                        except:
                            nerve_maps.append(nerve_frame)
                        try:
                            thresh = np.percentile(vessel_frame[vessel_frame > 0], 25)
                            vessel_frame_binary = vessel_frame
                            vessel_frame_binary[vessel_frame_binary <= thresh] = 0
                            vessel_maps.append((vessel_frame_binary* 255.).astype(np.uint8))
                        except:
                            vessel_maps.append(vessel_frame)
                    else:
                        needle_maps.append(needle_frame)
                        nerve_maps.append(nerve_frame)
                        vessel_maps.append(vessel_frame)
                    
                    frames.append(image_numpy[i,:,:,:])
                    needle_tips.append(needle_tip)

                count = 0
                # flow_frames = dense_optical_flow(frames = image_numpy)
                # pass
        
        image_numpy[count, :, :, :] = preprocess_img(image)        
        success, image = vid_file.read()
        count += 1      
        total_count += 1
    

    return frames, needle_maps, nerve_maps, vessel_maps, needle_tips
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--run', type=str, default='positive')
    parser.add_argument('--filepath', type=str, default='US_Test_Video.mp4')
    args = parser.parse_args()

    start = time.time()
    frames, needle_maps, nerve_maps, vessel_maps, needle_tips =  main(filepath = args.filepath)
    maps_time = time.time()

    print('Time Taken to compute maps : ', maps_time - start)

    min_dist, distances = np.Inf, []
    for i in range(len(needle_maps)):
        curr_dist = calculate_distance(needle_maps[i], nerve_maps[i], vessel_maps[i], (256,256), 'centroid')
        distances.append(curr_dist)
    
    end = time.time()

    print('Time taken to compute minimum distance : ', end - maps_time)

    print('Minimum distance : ', np.amin(distances))
    print('5th percentile distance : ', np.percentile(distances, 5))

    '''
    Run model for all videos - comment out
    '''
    # run = args.run
    # distance_preference = 'min' #'min'

    # min_runtime, max_runtime  = np.Inf, 0

    # if run == 'positive':
    #     root_dir = '../Positives/'
    #     dest_csv = '../pnb_distances_positive.csv'
 
    # else:
    #     root_dir = '../Negatives/'
    #     dest_csv = '../pnb_distances_negative.csv'

    # folders, distances = [], []

    # savepath = '../final_softmax_maps'

    # df = pd.DataFrame(columns=['Folder', 'Distance'])

    # for filepath in tqdm(os.listdir(root_dir)):
        
    #     folder_path = os.path.join(root_dir, filepath)
    #     vid_file = glob(os.path.join(folder_path, "*.mp4"))

    #     assert len(vid_file) == 1

    #     start = time.time()
    #     frames, needle_maps, nerve_maps, vessel_maps, needle_tips =  main(filepath = vid_file[0])

    #     min_dist, min_dist_with_tip, all_dists = np.Inf, np.Inf, []
    #     for i in range(len(needle_maps)):

    #         if not os.path.exists(os.path.join(savepath, run, filepath)):
    #             os.makedirs(os.path.join(savepath, run, filepath))

    #         cv2.imwrite(os.path.join(savepath, '{}/{}/{}_needle.png'.format(run, filepath, str(i))), needle_maps[i])
    #         cv2.imwrite(os.path.join(savepath, '{}/{}/{}_nerve.png'.format(run, filepath, str(i))), nerve_maps[i])
    #         cv2.imwrite(os.path.join(savepath, '{}/{}/{}_vessel.png'.format(run, filepath, str(i))), vessel_maps[i])

    #         curr_dist = calculate_distance(needle_maps[i], nerve_maps[i], vessel_maps[i], needle_tips[i], (256,256), 'centroid')
    #         all_dists.append(curr_dist)
    #         min_dist = min(curr_dist, min_dist)
    #         if needle_tips[i] is not None:
    #             min_dist_with_tip = min(curr_dist, min_dist)
        
    #     if distance_preference == 'min':
    #         if min_dist < np.Inf:
    #             if min_dist_with_tip < np.Inf:
    #                 distances.append(min_dist_with_tip)
    #             else:
    #                 distances.append(min_dist)
    #             print(distances[-1])
    #             folders.append(filepath)
    #     else:
    #         if min_dist < np.Inf:
    #             distances.append(np.percentile(all_dists, 5))
    #             print(distances[-1])
    #             folders.append(filepath)
        

    #     end = time.time()
    #     min_runtime = min(min_runtime, end - start)
    #     max_runtime = max(max_runtime, end - start)

        
    # df['Folder'] = folders
    # df['Distance'] = distances
    
    # df.to_csv(dest_csv)
    # print('The min and max runtimes are {}, {}'.format(min_runtime, max_runtime))



 