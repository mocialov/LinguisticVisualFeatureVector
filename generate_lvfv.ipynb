#HamNoSys lookup documents
#http://vhg.cmp.uea.ac.uk/tech/hamnosys/An%20intro%20to%20eSignEditor%20and%20HNS.pdf
#https://www.sign-lang.uni-hamburg.de/dgs-korpus/files/inhalt_pdf/HankeLRECSLP2004_05.pdf
#https://www.sign-lang.uni-hamburg.de/dgs-korpus/files/inhalt_pdf/HamNoSys_2018.pdf
#https://www.sign-lang.uni-hamburg.de/hamnosys/input/

#hand structure reference
#https://i.pinimg.com/originals/bb/39/6d/bb396dd8684ead5717b8052d0e5ac427.jpg

#generate json keypoints and copy frames, keypoints, and the video to the /content/drive/My Drive/PhD/lvfv/t_*
#srun --partition=amd-shortq --gres=gpu ./build/examples/openpose/openpose.bin --image_dir ../tegnsprog_frames/t_129/ --face --hand --write_images ../temp_openpose2/ --write_json ../keypoints_temp/ --no_display


import os
import json
from google.colab import drive
import cv2
import numpy as np
from IPython.display import Image, display, clear_output
import matplotlib.pyplot as plt
from time import sleep
import math
import copy
from MinimumBoundingBox.MinimumBoundingBox import MinimumBoundingBox
import operator
from scipy.stats import linregress
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter
from joblib import load
from google.colab import widgets
from matplotlib import pylab
import time

#CONSTANTS
frame_size = (576, 720)
window_size = 3 #had been noticed that window size 3 captures enough details without smoothing too much
total_buckets = 10
bucket_size = 10 #to squeeze/normalise the movement distances
columns = 7 #for the generated graphs outputs
rows = 100 #for the generated graphs outputs
#CONSTANTS

#THRESHOLDS
body_part_proximity_threshold = 0.1
curve_smoothing_window_size = 5 #should be dynamic - the longer the line, the larger the window
line_smoothing_polynomial_degree = 1
minimum_angle_between_slopes = 50
minimum_both_hands_distance_travelled = 20
minimum_curvature_to_classify_curved_line = 0.5
hand_location_side_over_body_part = 0.0007 #in vicinity of 1.43% of the frame width
#THRESHOLDS


#lvfv structure
class LVFV_STRUCTURE:
    def __init__(self, dez, ori, tab, sig):#, sig_side):
        self.dez = dez
        self.ori = ori
        self.tab = tab
        self.sig = sig
      
    def __str__(self):
        return ("\nDEZ:\n" + 
                "--RIGHT: " + str(self.dez[0]) +
                "\n--LEFT: " + str(self.dez[1]) +
                
                "\n\nORI:\n" + 
                "--FINGER:\n" + 
                "------RIGHT: " + str(self.ori[0]) +
                "\n------LEFT: " + str(self.ori[1]) + 
                
                "\n--PALM:\n" +
                "------RIGHT: " + str(self.ori[2]) +
                "\n------LEFT: " + str(self.ori[3]) +
                
                "\n\nTAB:\n " + 
                "--RIGHT: " + str(self.tab[0]) +
                "\n------SIDE: " + str(self.tab[1]) +
                "\n--LEFT: " + str(self.tab[2]) +
                "\n------SIDE: " + str(self.tab[3]) +
                
                "\n\nSIG:\n" + 
                "--RIGHT: " + str(self.sig[0]) + 
                "\n------CURVATURE: " + str(self.sig[1]) +
                "\n--LEFT: " + str(self.sig[2]) +
                "\n------CURVATURE: " + str(self.sig[3]) +
                "\n")
      
class HandShape:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<Hand shape: %s>" % self
    @staticmethod
    def to_list():
        return [HandShape.ONE, HandShape.TWO, HandShape.THREE, HandShape.FIVE, HandShape.B, HandShape.B_TOMMEL, 
                HandShape.C, HandShape.NINE, HandShape.G, HandShape.PEGE, HandShape.PAEDAGOG, HandShape.O, HandShape.S]
HandShape.ONE = HandShape('1')
HandShape.TWO = HandShape('2')
HandShape.THREE = HandShape('3')
HandShape.FIVE = HandShape('5')
HandShape.B = HandShape('B')
HandShape.B_TOMMEL = HandShape('B_TOMMEL')
HandShape.C = HandShape('C')
HandShape.NINE = HandShape('9')
HandShape.G = HandShape('G')
HandShape.PEGE = HandShape('PEGE')
HandShape.PAEDAGOG = HandShape('PAEDAGOG')
HandShape.O = HandShape('O')
HandShape.S = HandShape('S')



#categories for the HamNoSys
class HandMovementSpeed: #TODO
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<Hand movmeent speed: %s>" % self
HandMovementSpeed.SLOW = HandMovementSpeed("SLOW")
HandMovementSpeed.NORMAL = HandMovementSpeed("NORMAL")
HandMovementSpeed.FAST = HandMovementSpeed("FAST")


class CurveMovementArcDirection:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<Movement arc direction side: %s>" % self
CurveMovementArcDirection.LEFT = CurveMovementArcDirection("LEFT")
CurveMovementArcDirection.RIGHT = CurveMovementArcDirection("RIGHT")
      

class HandLocationSide:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<Location side: %s>" % self
HandLocationSide.LEFT = HandLocationSide("LEFT")
HandLocationSide.CENTER = HandLocationSide("CENTER")
HandLocationSide.RIGHT = HandLocationSide("RIGHT")


#No useful way to tell whether finger is pointing towards the body or away having only 2D information
#Assuming all directions are away from the body (is there a research on that? Statistics on how often people point to themselves vs away..)
class ExtendedFingerDirection:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<Direction: %s>" % self
ExtendedFingerDirection.AWAY_NORTH = ExtendedFingerDirection("AWAY_NORTH")
ExtendedFingerDirection.AWAY_NORTH_EAST = ExtendedFingerDirection("AWAY_NORTH_EAST")
ExtendedFingerDirection.AWAY_EAST = ExtendedFingerDirection("AWAY_EAST")
ExtendedFingerDirection.AWAY_SOUTH_EAST = ExtendedFingerDirection("AWAY_SOUTH_EAST")
ExtendedFingerDirection.AWAY_SOUTH = ExtendedFingerDirection("AWAY_SOUTH")
ExtendedFingerDirection.AWAY_SOUTH_WEST = ExtendedFingerDirection("AWAY_SOUTH_WEST")
ExtendedFingerDirection.AWAY_WEST = ExtendedFingerDirection("AWAY_WEST")
ExtendedFingerDirection.AWAY_NORTH_WEST = ExtendedFingerDirection("AWAY_NORTH_WEST")

ExtendedFingerDirection.TOWARDS_NORTH = ExtendedFingerDirection("TOWARDS_NORTH")
ExtendedFingerDirection.TOWARDS_NORTH_EAST = ExtendedFingerDirection("TOWARDS_NORTH_EAST")
ExtendedFingerDirection.TOWARDS_EAST = ExtendedFingerDirection("TOWARDS_EAST")
ExtendedFingerDirection.TOWARDS_SOUTH_EAST = ExtendedFingerDirection("TOWARDS_SOUTH_EAST")
ExtendedFingerDirection.TOWARDS_SOUTH = ExtendedFingerDirection("TOWARDS_SOUTH")
ExtendedFingerDirection.TOWARDS_SOUTH_WEST = ExtendedFingerDirection("TOWARDS_SOUTH_WEST")
ExtendedFingerDirection.TOWARDS_WEST = ExtendedFingerDirection("TOWARDS_WEST")
ExtendedFingerDirection.TOWARDS_NORTH_WEST = ExtendedFingerDirection("TOWARDS_NORTH_WEST")
directions_dictionary = {
                        "NORTHERN_HEMISPHERE": 
                            {ExtendedFingerDirection.AWAY_EAST: (0.0, 22.6),
                            ExtendedFingerDirection.AWAY_NORTH_EAST: (22.6, 67.6),
                            ExtendedFingerDirection.AWAY_NORTH: (67.6, 112.6),
                            ExtendedFingerDirection.AWAY_NORTH_WEST: (112.6, 157.6),
                            ExtendedFingerDirection.AWAY_WEST: (157.6, 180.1)},
                        "SOUTHERN_HEMISPHERE":
                            {ExtendedFingerDirection.AWAY_WEST: (-180.0, -157.6),
                            ExtendedFingerDirection.AWAY_SOUTH_WEST: (-157.6, -112.6),
                            ExtendedFingerDirection.AWAY_SOUTH: (-112.6, -67.6),
                            ExtendedFingerDirection.AWAY_SOUTH_EAST: (-67.6, -22.6),
                            ExtendedFingerDirection.AWAY_EAST: (-22.6, 0.0)},
                        }


class HandMovementType:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return "<Curvature: %s>" % self
HandMovementType.STRAIGHT = HandMovementType("STRAIGHT")
HandMovementType.CURVED = HandMovementType("CURVED")
HandMovementType.CIRCULAR = HandMovementType("CIRCULAR") #not implemented yet
HandMovementType.ZIGZAG = HandMovementType("ZIGZAG") #not implemented yet
HandMovementType.WAVY = HandMovementType("WAVY") #not implemented yet
#categories for the HamNoSys


#nasty globals
right_hand_history = []
left_hand_history = []
both_hands_distances_traveled_per_sign = []
slope_changes = []
slope_changes_backtrack = []
cut_off_points = {}
last_point = None
last_slopes = []
a_slopes_angles = []
a_slopes = []
last_slope_angle = -1
right_hand_proximity_arrays = []
left_hand_proximity_arrays = []
generated_lvfvs = []
show_graphs = False
#nasty globals


openpose_keypoints_pose = ["Nose_x","Nose_y","Nose_c","Neck_x","Neck_y","Neck_c","RShoulder_x","RShoulder_y","RShoulder_c","RElbow_x","RElbow_y","RElbow_c","RWrist_x","RWrist_y","RWrist_c","LShoulder_x","LShoulder_y","LShoulder_c","LElbow_x","LElbow_y","LElbow_c","LWrist_x","LWrist_y","LWrist_c","RHip_x","RHip_y","RHip_c","RKnee_x","RKnee_y","RKnee_c","RAnkle_x","RAnkle_y","RAnkle_c","LHip_x","LHip_y","LHip_c","LKnee_x","LKnee_y","LKnee_c","LAnkle_x","LAnkle_y","LAnkle_c","REye_x","REye_y","REye_c","LEye_x","LEye_y","LEye_c","REar_x","REar_y","REar_c","LEar_x","LEar_y","LEar_c"]
openpose_keypoints_hand = ["radius_x","radius_y","radius_c","scaphoid_x","scaphoid_y","scaphoid_c","thumb_trapezium_x","thumb_trapezium_y","thumb_trapezium_c","thumb_metacarpal_x","thumb_metacarpal_y","thumb_metacarpal_c","thumb_phalange_x","thumb_phalange_y","thumb_phalange_c","index_trapezium_x","index_trapezium_y","index_trapezium_c","index_metacarpal_x","index_metacarpal_y","index_metacarpal_c","index_proximal_x","index_proximal_y","index_proximal_c","index_phalange_x","index_phalange_y","index_phalange_c","middle_trapezium_x","middle_trapezium_y","middle_trapezium_c","middle_metacarpal_x","middle_metacarpal_y","middle_metacarpal_c","middle_proximal_x","middle_proximal_y","middle_proximal_c","middle_phalange_x","middle_phalange_y","middle_phalange_c","ring_trapezium_x","ring_trapezium_y","ring_trapezium_c","ring_metacarpal_x","ring_metacarpal_y","ring_metacarpal_c","ring_proximal_x","ring_proximal_y","ring_proximal_c","ring_phalange_x","ring_phalange_y","ring_phalange_c","little_trapezium_x","little_trapezium_y","little_trapezium_c","little_metacarpal_x","little_metacarpal_y","little_metacarpal_c","little_proximal_x","little_proximal_y","little_proximal_c","little_phalange_x","little_phalange_y","little_phalange_c"]
openpose_keypoints_face = ["mouth..."] #TODO

hamnosys_locations = ["above_head","head","forehead","nose","mouth","tongue","teeth","chin","below_chin","neck","shoulder_line","breast_line","stomach_line","abdominal_line","eye_brows","eyes","ears","cheeks"]
open_pose_hamnosys_mapping = {"Nose":"nose", "Neck":"neck", "RShoulder":"shoulder_line", "RElbow":None, "RWrist":None, "LShoulder":"shoulder_line", "LElbow":None, "LWrist":None, "RHip":"abdominal_line", "RKnee":None, "RAnkle":None, "LHip":"abdominal_line", "LKnee":None, "LAnkle":None, "REye":"eyes", "LEye":"eyes", "REar":"ears", "LEar":"ears"}

#to load videos from the dataset
drive.mount('/content/drive')
dataset_dir = "/content/drive/My Drive/PhD/lvfv/collected_videos_sampled_0.04:0.08/t_120"
#to load videos from the dataset

handshape_estimator_path = "/content/drive/My Drive/PhD/lvfv/handshape_estimator.joblib"

frcnn_dataset_path = "/content/drive/My Drive/PhD/lvfv/sampled_bounding_boxes_hands.txt"

#show generated graphs
grid = widgets.Grid(1, 2)

fig=plt.figure(figsize=(20, 900)) #increase width to have bigger pictures
plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.5, wspace=0.1)
#show generated graphs

def update_frcnn_dataset(a_folder, a_file, right_finger_orientation, left_finger_orientation):
    with open(frcnn_dataset_path) as f:
        lis = [line.split() for line in f]
        for i, x in enumerate(lis):
            #print ("searching for", a_folder + "_" + a_file +" in "+ x[0].split(",")[0].split("/")[-1].split(".")[0])
            if a_folder + "_" + a_file in x[0].split(",")[0].split("/")[-1].split(".")[0]:
                
                lis[i][0] = lis[i][0] + "," + str(right_finger_orientation)
                lis[i][0] = lis[i][0] + "," + str(left_finger_orientation)
                
                with open(frcnn_dataset_path, 'w') as f2:
                    for item in lis:
                        f2.write("%s\n" % item[0])

def load_trained_handshape_model():
    return load(handshape_estimator_path)
    
def predict_handshape(model, features):
    #delete confidence value
    features = list(features)
    k=3
    del features[k-1::k]
    
    return model.predict([features])

def calculate_curvature3(xs, ys):
    last_slope = None
    
    slope_angles = []
    
    for idx, x in enumerate(xs[:-1]):
        y = ys[idx]
        
        next_x = xs[idx+1]
        next_y = ys[idx+1]
        
        find_slope = linregress([int(x), int(next_x)], [int(y), int(next_y)])
    
        if last_slope != None:
            slope_angles.append(slopes_angle(last_slope[0], find_slope[0]))
      
        last_slope = find_slope
        
    return slope_angles

def calculate_curvature2(x, y=None, error=0.1):
    
    from scipy.interpolate import UnivariateSpline

    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.

    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )

         In the second case the curve is represented as a np.array
         of complex numbers.

    error : float
        The admisible error when interpolating the splines

    Returns
    -------
    curvature: numpy.array shape (n_points, )

    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 1.5)
    return curvature

def calculate_curvature(arm, xs, ys):
    #https://stackoverflow.com/questions/50562254/curvature-of-set-of-points
    
    #first derivatives 
    dx= np.gradient(xs)
    dy = np.gradient(ys)
    
    #second derivatives 
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    
    #calculation of curvature from the typical formula
    
    #https://en.wikipedia.org/wiki/Curvature#Signed_curvature
    #positive - curved to the left?
    #negative - curved to the right?
    #print (arm, "signed curvature", (dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5) #its a list - report as a list!
    
    #curvature direction using signed curvature
    curvature_categories_list = [CurveMovementArcDirection.RIGHT if i < 0 else CurveMovementArcDirection.LEFT for i in (dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5]
    
    curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5

    return curvature_categories_list, curvature

def show_trajectory_per_interval(backtracked_frame_index, a_file, ax_traj, a_file_idx, right_hand_window, left_hand_window, right_hand_proximity_array, left_hand_proximity_array):
    global fig, frame_size, dataset_dir, right_hand_history, left_hand_history, right_hand_proximity_arrays, left_hand_proximity_arrays, curve_smoothing_window_size, line_smoothing_polynomial_degree
    
    hands_movement_type = [None, None] #right, left
    
    GENERATED_MOVEMENT_RIGHT = None
    GENERATED_MOVEMENT_RIGHT_SIDE = None
    GENERATED_MOVEMENT_LEFT = None
    GENERATED_MOVEMENT_LEFT_SIDE = None
    
    img = cv2.imread(dataset_dir+"/"+a_file.split("_")[0]+".png", 0)
    
    #if len(right_hand_proximity_arrays) > 2:
    #    #print ((len(set(right_hand_proximity_arrays[-1])) > 1), (list(set(right_hand_proximity_arrays[-1]))[0] != 1.0), (len(set(right_hand_proximity_arrays[-2])) > 1), (list(set(right_hand_proximity_arrays[-2]))[0] != 1.0), (len(set(right_hand_proximity_arrays[-3])) > 1), (list(set(right_hand_proximity_arrays[-3]))[0] != 1.0))
    #    if (len(set(right_hand_proximity_arrays[-2])) == 1) and (list(set(right_hand_proximity_arrays[-2]))[0] == 1.0) and (len(set(right_hand_proximity_arrays[-3])) == 1) and (list(set(right_hand_proximity_arrays[-3]))[0] == 1.0):
    #        return 
    #else:
    #    return 
    
    #print ("compare lengths of hand history and proximity arrays", len(right_hand_history), len(right_hand_proximity_arrays))
    
    #right_hand_history = [hist_prox[0] for hist_prox in zip(*right_hand_history, right_hand_proximity_arrays) if (len(set(hist_prox[1])) > 1 and list(set(hist_prox[1]))[0] != 1.0)]
    #left_hand_history = [hist_prox[0] for hist_prox in zip(*left_hand_history, right_hand_proximity_arrays) if (len(set(hist_prox[1])) > 1 and list(set(hist_prox[1]))[0] != 1.0)]
    
    #print ("selecting indices to remove")
    #print ("proximity arrays")
    #print (np.matrix(right_hand_proximity_arrays[backtracked_frame_index:]))
    remove_indices = []
    for idx, an_array in enumerate(right_hand_proximity_arrays[backtracked_frame_index:]):
        #print (an_array)
        if (len(set(an_array)) == 1) and (list(set(an_array))[0] == 1.0):
            remove_indices.append(idx)
            
    #print ("removing indices", remove_indices)
    
    #print ("history")
    #print(right_hand_history[backtracked_frame_index:])
    
    #return
        
    #right_hand_history_ = [i for j, i in enumerate(right_hand_history) if j not in remove_indices]
    #left_hand_history_ = [i for j, i in enumerate(left_hand_history) if j not in remove_indices]
    
    #backtracked_frame_index = backtracked_frame_index + len(remove_indices)
    
    #deviation = 0
    #for right_hand_proximity_item in right_hand_proximity_arrays[backtracked_frame_index:]:
    #    if (len(set(right_hand_proximity_item)) == 1) and (list(set(right_hand_proximity_item))[0] == 1.0):
    #        deviation += 1
                                              
    if True:#len(set(right_hand_proximity_array)) > 1 and list(set(right_hand_proximity_array))[0] != 1.0:
        #print ("before delete")
        #print (right_hand_history[backtracked_frame_index:])
        temp = []
        #print ("!!!!!!")
        #print(len(right_hand_history[backtracked_frame_index:]), remove_indices)
        for history_idx, history_item in enumerate(right_hand_history[backtracked_frame_index:]):
            if history_idx not in remove_indices:
                temp.append(history_item)
        
        #print ("modified history")
        #print(temp)
        
        #temp = np.delete(temp, remove_indices)
        #print ("after delete")
        #print (temp)
        
        if len(temp) > 1:
            xs, ys = zip(*temp) #zip(*right_hand_window)
        
            #https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
          
            if len(xs) > curve_smoothing_window_size:
                yhat = savgol_filter(ys, curve_smoothing_window_size, line_smoothing_polynomial_degree)
            else:
                yhat = ys
                
            if show_graphs:
                ax_traj.plot(np.array(xs)/float(frame_size[1]), np.array(yhat)/float(frame_size[0]), 'r-')
        
            #ax_traj.text(0, -0.02, ", ".join(map(str,right_hand_proximity_array)))
            #ax_traj.text(0, -0.025, ", ".join(map(str,np.round(calculate_curvature(np.array(xs), np.array(yhat)), 2))) + "="+str(np.sum(calculate_curvature(np.array(xs), np.array(yhat)))))
            #ax_traj.text(0, -0.01, "back " + str(backtracked_frame_index) + " in right hand history")
            #ax.text(0, 11.5, ",".join(map(str,np.array(xs)/float(frame_size[1]))) + " | " ",".join(map(str,np.array(ys)/float(frame_size[0]))))
            
            if show_graphs:
                for idx, a_point in enumerate(list(xs[:-1])):
                    cv2.line(img, (int(xs[idx]), int(yhat[idx])), (int(xs[idx+1]), int(yhat[idx+1])), (255,255,255), 3)

            curvature_categories_list, curvature = calculate_curvature("right", np.array(xs), np.array(yhat))
                
            hands_movement_type[0] = HandMovementType.CURVED if np.sum(curvature) > minimum_curvature_to_classify_curved_line else HandMovementType.STRAIGHT
    
            #ax_traj.text(0, -0.025, "right hand: " + hands_movement_type[0].name)
      
            GENERATED_MOVEMENT_RIGHT = HandMovementType.CURVED if np.sum(curvature) > minimum_curvature_to_classify_curved_line else HandMovementType.STRAIGHT
    
            GENERATED_MOVEMENT_RIGHT_SIDE = curvature_categories_list
      
            #temp = []
            #for history_idx, history_item in enumerate(left_hand_history[backtracked_frame_index:]):
            #    if history_idx not in remove_indices:
            #        temp.append(history_item)
            #temp = np.delete(temp, remove_indices)
            #xs2, ys2 = zip(*temp) #zip(*left_hand_window)
        
        remove_indices = []
        for idx, an_array in enumerate(left_hand_proximity_arrays[backtracked_frame_index:]):
            #print (an_array)
            if (len(set(an_array)) == 1) and (list(set(an_array))[0] == 1.0):
                remove_indices.append(idx)
                
        temp = []
        for history_idx, history_item in enumerate(left_hand_history[backtracked_frame_index:]):
            if history_idx not in remove_indices:
                temp.append(history_item)

        if len(temp) > 1:
            xs2, ys2 = zip(*temp)
        
            if len(xs2) > curve_smoothing_window_size:
                yhat2 = savgol_filter(ys2, curve_smoothing_window_size, line_smoothing_polynomial_degree)
            else:
                yhat2 = ys2
        
            if show_graphs:
                ax_traj.plot(np.array(xs2)/float(frame_size[1]), np.array(yhat2)/float(frame_size[0]), 'b-')
            #
            #print ("curvature", calculate_curvature(np.array(xs2)/float(frame_size[1]), np.array(ys2)/float(frame_size[0])))
            #
            #ax.text(0, 11.5, ",".join(map(str,np.array(xs2)/float(frame_size[1]))) + " | " ",".join(map(str, np.array(ys2)/float(frame_size[0]))))
            #
            #print (list(xs[:-1]), list(xs2[:-1]))
            
            #ax_traj.text(0, -0.01, "left hand curvature: "+str(np.sum(calculate_curvature(np.array(xs2), np.array(yhat2)))))
            
            if show_graphs:
                for idx, a_point in enumerate(list(xs2[:-1])):
                    cv2.line(img, (int(xs2[idx]), int(yhat2[idx])), (int(xs2[idx+1]), int(yhat2[idx+1])), (255,255,255), 3)
                
            curvature_categories_list, curvature = calculate_curvature("left", np.array(xs2), np.array(yhat2))
                
            hands_movement_type[1] = HandMovementType.CURVED if np.sum(curvature) > minimum_curvature_to_classify_curved_line else HandMovementType.STRAIGHT

            #ax_traj.text(0, -0.01, "left hand: " + hands_movement_type[1].name)
            
            GENERATED_MOVEMENT_LEFT = HandMovementType.CURVED if np.sum(curvature) > minimum_curvature_to_classify_curved_line else HandMovementType.STRAIGHT
        
            GENERATED_MOVEMENT_LEFT_SIDE = curvature_categories_list
          
        #img = cv2.imread(dataset_dir+"/"+a_file.split("_")[0]+".png", 0)
        #for idx, a_point in enumerate(list(xs[:-1])):
        #    cv2.line(img, (int(xs[idx]), int(yhat[idx])), (int(xs[idx+1]), int(yhat[idx+1])), (255,255,255), 3)
        #for idx, a_point in enumerate(list(xs2[:-1])):
        #    cv2.line(img, (int(xs2[idx]), int(yhat2[idx])), (int(xs2[idx+1]), int(yhat2[idx+1])), (255,255,255), 3)
        
        if show_graphs:
            ax___ = fig.add_subplot(rows, columns, columns*json_file_idx+5)
            ax___.axis('off')
            plt.imshow(img, interpolation='nearest')
        
        return (GENERATED_MOVEMENT_RIGHT, GENERATED_MOVEMENT_RIGHT_SIDE, GENERATED_MOVEMENT_LEFT, GENERATED_MOVEMENT_LEFT_SIDE) #hands_movement_type
    

def backtrack_slopes(slopes):
    #print ("slopes", [x[0][0] for x in slopes])
    last_slope = slopes[-1][0][0]
    idx_of_interest = 0+1
    from_to_frames = list(reversed(slopes))[0][1]
    y_axis = -1
    
    for idx, a_slope in enumerate(list(reversed(slopes))[1:]):
        #print ("index", idx)
        
        if last_slope < 0 and a_slope[0][0] < 0:
            idx_of_interest = idx+2
            from_to_frames = a_slope[1]
            #print ("!!!!!", list(reversed(slopes))[idx+1])
            y_axis = list(reversed(slopes))[idx+1][1][1] #a_slope[1][1]
            #continue 
        elif last_slope > 0 and a_slope[0][0] > 0:
            idx_of_interest = idx+2
            from_to_frames = a_slope[1]
            #print ("!!!!!", list(reversed(slopes))[idx+1])
            y_axis = list(reversed(slopes))[idx+1][1][1] #a_slope[1][1]
            #continue 
        else:
            #return idx_of_interest
            break
            
    #print ("returning index", -idx_of_interest)
    return (-idx_of_interest, from_to_frames, y_axis)

def delta_speeds(speed1, speed2):
    return abs(speed1 - speed2)

def slopes_angle(slope1, slope2):
    angle_degrees = math.degrees( abs( math.atan2(  slope1 - slope2, float(1 + slope1 * slope2) ) ) ) 
    #print ("angle between", slope1, "and", slope2, "is", angle_degrees )
    return angle_degrees
    
def detect_cut_off_points(a_file, a_file_idx, right_hand_window_centroids, left_hand_window_centroids, right_hand_proximity_array, left_hand_proximity_array):
    global right_hand_history, left_hand_history, both_hands_distances_traveled_per_sign, bucket_size, bucket_size, window_size, rows, columns, fig, last_point, last_slopes, slope_changes, slope_changes_backtrack, last_slope_angle, a_slopes_angles, right_hand_proximity_arrays, minimum_angle_between_slopes, minimum_both_hands_distance_travelled
    
    #print ("frame", a_file_idx)
    
    GENERATED_MOVEMENT = [None, None, None, None]
    
    right_hand_history.append(right_hand_window_centroids)
    left_hand_history.append(left_hand_window_centroids)
    
    hands_movement_type = None
    
    if (len(left_hand_history) % window_size == 0): #len(right_hand_window) == window_size and len(left_hand_window) == window_size:
        left_hand_window = left_hand_history[-window_size:]
        right_hand_window = right_hand_history[-window_size:]
        
        right_hand_bounding_box = None
        left_hand_bounding_box = None
        
        if right_hand_window.count((0,0)) == 0:
            right_hand_bounding_box = MinimumBoundingBox(tuple(right_hand_window))
        if left_hand_window.count((0,0)) == 0:
            left_hand_bounding_box = MinimumBoundingBox(tuple(left_hand_window))
        
        ax_traj = None
        if show_graphs:
            ax = fig.add_subplot(rows, columns, columns*a_file_idx+4)
        
            ax_traj = fig.add_subplot(rows, columns, columns*a_file_idx+6)
            ax_traj.set_ylim(ax_traj.get_ylim()[::-1])
            ax_traj.set_xlim(left=0.0, right=1.0)

        right_hand_distance = -1
        left_hand_distance = -1
        
        if right_hand_bounding_box != None:
            right_hand_distance = max(right_hand_bounding_box.length_parallel, right_hand_bounding_box.length_orthogonal)
        if left_hand_bounding_box != None:
            left_hand_distance = max(left_hand_bounding_box.length_parallel, left_hand_bounding_box.length_orthogonal)
        hands_distance_max = max(right_hand_distance, left_hand_distance)
        
        
        both_hands_distances_traveled_per_sign.append((a_file.split("frame")[1].split("_")[0], int(hands_distance_max)))
        
        if show_graphs:
            ax.plot([i[0] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)], [i[1]/bucket_size if i[1]/bucket_size < total_buckets-1 else total_buckets-1 for i in both_hands_distances_traveled_per_sign])         
        
        if last_point != None:
            find_slope = linregress([int(last_point[0]), int(a_file.split("frame")[1].split("_")[0])], [last_point[1], int(hands_distance_max)])
            
            if len(last_slopes) != 0:
                a_slopes_angles.append(slopes_angle(last_slopes[-1][0][0], find_slope[0]))
                
                if last_slope_angle != -1:
                    #ax.text(0, 10.7, "selected frames: from "+str(int([i[0] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)][backtracked_idx]))+" to "+str([i[0] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)][-1]))
                    #ax.text(0, 10.5, "angle: "+str(round(float(a_slopes_angles[-1]), 2)), fontsize=8)
                    #ax.text(0, 10, "delta speeds: "+str(delta_speeds([i[1] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)][backtracked_idx], [i[1] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)][-1])))
                    #ax.text(0, 10.8, "will backtrack? "+str((last_slopes[-1][0] > 0 and find_slope[0] < 0) or (last_slopes[-1][0] < 0 and find_slope[0] > 0))+" backtracking idx: "+str(backtracked_idx) + " out of: " +str(len([i[1] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)])))
                    
                    #slope_changes.append(([i[0] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)][-1], [i[1]/bucket_size if i[1]/bucket_size < total_buckets-1 else total_buckets-1 for i in both_hands_distances_traveled_per_sign][-1]))
                    if ((last_slopes[-1][0][0] > 0 and find_slope[0] < 0) or (last_slopes[-1][0][0] < 0 and find_slope[0] > 0)):
                        
                        #backtrack here
                        if (a_slopes_angles[-1] > minimum_angle_between_slopes and delta_speeds([i[1] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)][-2], [i[1] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)][-1]) >= minimum_both_hands_distance_travelled):
                            backtracked_idx = backtrack_slopes(last_slopes)
                            
                            slope_changes.append(([i[0] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)][-2], [i[1]/bucket_size if i[1]/bucket_size < total_buckets-1 else total_buckets-1 for i in both_hands_distances_traveled_per_sign][-2])) 
                            slope_changes_backtrack.append((backtracked_idx[1][0], [i[1]/bucket_size if i[1]/bucket_size < total_buckets-1 else total_buckets-1 for i in both_hands_distances_traveled_per_sign][backtracked_idx[0]-2]))#(([i[0] for i in copy.deepcopy(both_hands_distances_traveled_per_sign)][backtracked_idx[0]], [i[1]/bucket_size if i[1]/bucket_size < total_buckets-1 else total_buckets-1 for i in both_hands_distances_traveled_per_sign][backtracked_idx[0]])) 
                            
                            #hands_movement_type = show_trajectory_per_interval(((backtracked_idx[0]-2)*3)+3, a_file, ax_traj, a_file_idx, right_hand_window, left_hand_window, right_hand_proximity_array, left_hand_proximity_array)
                            GENERATED_MOVEMENT = show_trajectory_per_interval(((backtracked_idx[0]-2)*3)+3, a_file, ax_traj, a_file_idx, right_hand_window, left_hand_window, right_hand_proximity_array, left_hand_proximity_array)
                            
                            #ax.text(0, 10.7, str(backtracked_idx[1]) + "---" + " | ".join(["%s" % "from: "+str(k[0][0])+" to: "+str(k[1][0]) for k in zip(slope_changes_backtrack, slope_changes)]))
                            
                            if show_graphs:
                                for a_slope_change in slope_changes:
                                    center = plt.Circle(a_slope_change, 0.1, color="r", alpha=0.5)
                                    ax.add_artist(center)
                              
                                for a_slope_change_backtrack in slope_changes_backtrack:
                                    center = plt.Circle(a_slope_change_backtrack, 0.1, color="g", alpha=0.5)
                                    ax.add_artist(center)
                    
                    #temp
                    if show_graphs:
                        fig.canvas.draw()
                        labels = [item.get_text() for item in ax.get_xticklabels()]
                        new_labels = [ "%d" % int(l.lstrip("0")) if l != "" else "" for l in labels]
                        ax.set_xticklabels(new_labels, rotation='vertical')
                    #temp
                    
                last_slope_angle = a_slopes_angles[-1]
            last_slopes.append([find_slope, (last_point[0], int(hands_distance_max)/bucket_size if int(hands_distance_max)/bucket_size < total_buckets-1 else total_buckets-1)]) #a_file.split("frame")[1].split("_")[0]) ... /bucket_size if i[1]/bucket_size < total_buckets-1 else total_buckets-1
            
        last_point = (a_file.split("frame")[1].split("_")[0],int(hands_distance_max))
        
        right_hand_window = []
        left_hand_window = []
    
    return GENERATED_MOVEMENT #hands_movement_type
        

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def centeroidpython(data):
    k=3   
    del data[k-1::k]

    data = zip(data[0::2], data[1::2])

    #print data
    
    x, y = zip(*data)
    l = len(x)
    return sum(x) / l, sum(y) / l
  
def euclidean_distance(p,q):
    p1=p[0]
    p2=p[1]
    q1=q[0]
    q2=q[1]

    if (p[0]==0 and p[1]==0) or (q[0]==0 and q[1]==0) or (p[0] > 720) or (p[1] > 576) or (q[0] > 720) or (q[1] > 576) or (p[0] < 0) or (p[1] < 0) or (q[0] < 0) or (q[1] < 0):
        return 1.0
    else:
        return math.sqrt( (q1-p1)**2 + (q2-p2)**2) / math.sqrt( (0-576)**2 + (0-720)**2)
      
def angle_of_line(p,q):
    dx=q[0]-p[0]
    dy=q[1]-p[1]
    
    return math.atan2(dy, dx)
      
def normal_angle_to_line_right_hand(p,q):
    dx=q[0]-p[0]
    dy=q[1]-p[1]
    
    return math.atan2(dy, dx) + math.pi / 2.0
  
def normal_angle_to_line_left_hand(p,q):
    dx=q[0]-p[0]
    dy=q[1]-p[1]
    
    return math.atan2(dy, dx) - math.pi / 2.0
    
def relative_point_location_to_line(line_x0, line_y0, line_x1, line_y1, point_x, point_y):
    #https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
    return ((point_x - line_x0) * (line_y1 - line_y0)) - ((point_y - line_y0) * (line_x1 - line_x0))

all_json_files = []
for (dirpath, dirnames, filenames) in os.walk(dataset_dir):
    for filename in filenames:
        if filename.endswith('.json'):
            all_json_files.append(filename)
            
all_json_files.sort(key=lambda x: int(x.split("frame")[1].split("_")[0]))


trained_handshape_model = load_trained_handshape_model()


for json_file_idx, json_file in enumerate(all_json_files):
    #print ("frame", json_file_idx)
  
    #temp not to wait for all the frames to finish processing
    #if json_file_idx < 0 or json_file_idx > 40:
    #    continue
        
    with open(dataset_dir+"/"+json_file) as f:
        data = json.load(f)
        
        #to check that the size of the defined points matches the size of the received points from the json file
        if "pose_keypoints" in data["people"][0] and "hand_right_keypoints" in data["people"][0] and "hand_left_keypoints" in data["people"][0] and "face_keypoints" in data["people"][0]:
            if len(data["people"][0]["pose_keypoints"]) == len(openpose_keypoints_pose) and len(data["people"][0]["hand_right_keypoints"]) == len(openpose_keypoints_hand) and len(data["people"][0]["hand_left_keypoints"]) == len(openpose_keypoints_hand):
                #print ("all ok for", json_file)
                
                #####################BOF Location#########################
                
                #temp
                #k=3
                #del data["people"][0]["pose_keypoints"][k-1::k]
                
                #BOF plot points for visual verification
                img = np.zeros((frame_size[0],frame_size[1],3), np.uint8)
                
                if show_graphs:
                    for idx, points in enumerate(list(chunks(data["people"][0]["pose_keypoints"], 3))):
                        if idx == int(openpose_keypoints_pose.index("RHip_x") / 3) or idx == int(openpose_keypoints_pose.index("LHip_x") / 3):
                            cv2.circle(img,(int(points[0]),int(points[1])), 10, (255,0,0), -1)
                        #elif idx == int(openpose_keypoints_pose.index("RShoulder_x") / 3) or idx == int(openpose_keypoints_pose.index("LShoulder_x") / 3):
                        #    cv2.circle(img,(int(points[0]),int(points[1])), 5, (0,255,0), -1)
                        if True:#else:
                            cv2.circle(img,(int(points[0]),int(points[1])), 5, (0,0,255), -1)
                    #for idx, points in enumerate(list(chunks(data["people"][0]["hand_right_keypoints"], 3))):
                    #    cv2.circle(img,(int(points[0]),int(points[1])), 5, (255,255,255), -1)
                centroid_right_hand = centeroidpython(copy.deepcopy(data["people"][0]["hand_right_keypoints"]))
        
                if show_graphs:
                    cv2.circle(img,(int(centroid_right_hand[0]),int(centroid_right_hand[1])), 5, (255,100,100), -1)
                    #for idx, points in enumerate(list(chunks(data["people"][0]["hand_left_keypoints"], 3))):
                    #    cv2.circle(img,(int(points[0]),int(points[1])), 5, (255,255,255), -1)
                centroid_left_hand = centeroidpython(copy.deepcopy(data["people"][0]["hand_left_keypoints"]))
            
                if show_graphs:
                    cv2.circle(img,(int(centroid_left_hand[0]),int(centroid_left_hand[1])), 5, (255,100,100), -1)
                
                #clear_output()
                
                if show_graphs:
                    ax_ = fig.add_subplot(rows, columns, columns*json_file_idx+1)
                    ax_.axis('off')
                    plt.imshow(img, interpolation='nearest')
                #plt.show()
                #sleep(0.5)
                #EOF plot points for visual verification
                
                #BOF calculating Euclidean distances matrix
                distances_matrix = np.zeros((len(list(chunks(data["people"][0]["pose_keypoints"], 3))), 4))

                for pose_idx, pose_points in enumerate(list(chunks(data["people"][0]["pose_keypoints"], 3))):
                    distances_matrix[pose_idx][0] = euclidean_distance((pose_points[0],pose_points[1]), (centroid_right_hand[0], centroid_right_hand[1]))
                    distances_matrix[pose_idx][1] = euclidean_distance((pose_points[0],pose_points[1]), (centroid_left_hand[0], centroid_left_hand[1]))
                    distances_matrix[pose_idx][2] = pose_points[0]
                    distances_matrix[pose_idx][3] = pose_points[1]
                    
                #print (distances_matrix)
                #EOF calculating Euclidean distances matrix
                
                distances_matrix_ = distances_matrix[:,[0,1]]
                
                #BOF heatmap of the distance matrix
                x_axis_labels = ["right_hand_centroid", "left_hand_centroid"]
                y_axis_labels = copy.deepcopy(openpose_keypoints_pose)
                k=3
                del y_axis_labels[k-1::k]
                k=2
                del y_axis_labels[k-1::k]
                y_axis_labels = [w.replace('_x', '') for w in y_axis_labels]
                
                if show_graphs:
                    ax = fig.add_subplot(rows, columns, columns*json_file_idx+2)
                    im = plt.imshow(distances_matrix_, interpolation='nearest')
                
                    ax.set_xticks(np.arange(len(x_axis_labels)))
                    ax.set_yticks(np.arange(len(y_axis_labels)))
                
                    ax.set_xticklabels(x_axis_labels, rotation='vertical')
                    ax.set_yticklabels(y_axis_labels)
                
                    for i in range(len(y_axis_labels)):
                        for j in range(len(x_axis_labels)):
                            text = ax.text(j, i, round(float(distances_matrix_[i, j]), 2), ha="center", va="center", color="w")

                
                    fig.colorbar(im)
                #EOF heatmap of the distance matrix
                
                #BOF detecting proximity to hamnosys category
                GENERATED_PROXIMITY_RIGHT = None
                GENERATED_PROXIMITY_RIGHT_SIDE = None
                GENERATED_PROXIMITY_LEFT = None
                GENERATED_PROXIMITY_LEFT_SIDE = None
                
                found_hamnosys_proximity = False
                right_hand_proximity_array = copy.deepcopy(distances_matrix_[:,0])
                #print ("proxim array")
                #print (np.matrix(right_hand_proximity_array))
                while not found_hamnosys_proximity:
                    
                    if len(set(right_hand_proximity_array)) == 1 and list(set(right_hand_proximity_array))[0] == 1.0:
                        found_hamnosys_proximity = True
                        #print ("did not find any body part in proximity")
                        continue
                    
                    right_hand_near_idx = right_hand_proximity_array.argmin()
                    
                    for open_pose_key in open_pose_hamnosys_mapping.keys():
                        hamnosys_value = open_pose_hamnosys_mapping[open_pose_key]
                        
                        if hamnosys_value != None and y_axis_labels[right_hand_near_idx] == open_pose_key and right_hand_proximity_array[right_hand_near_idx] < body_part_proximity_threshold:
                            found_hamnosys_proximity = True
                            #print ("right hand is near", hamnosys_value)
                            
                            GENERATED_PROXIMITY_RIGHT = hamnosys_value
                            
                            #print ("location of the body part", distances_matrix[right_hand_near_idx][2], distances_matrix[right_hand_near_idx][3])
                            #cv2.circle(img,(int(distances_matrix[right_hand_near_idx][2]),int(distances_matrix[right_hand_near_idx][3])), 15, (255,255,255), -1)                            
                            proximity_side = relative_point_location_to_line(float(distances_matrix[right_hand_near_idx][2]/frame_size[0]),
                                                                             float(distances_matrix[right_hand_near_idx][3]/frame_size[1]),
                                                                             float(distances_matrix[right_hand_near_idx][2]/frame_size[0]), 
                                                                             float((distances_matrix[right_hand_near_idx][3]+10)/frame_size[1]), 
                                                                             float(centroid_right_hand[0]/frame_size[0]),
                                                                             float(centroid_right_hand[1]/frame_size[1])) # negative - left
                            
                            #print ("proximity side", HandLocationSide.CENTER if abs(proximity_side) < hand_location_side_over_body_part else HandLocationSide.RIGHT if proximity_side < 0)
                            
                            if abs(proximity_side) < hand_location_side_over_body_part:
                                #print ("right hand is in the ", HandLocationSide.CENTER)
                                
                                GENERATED_PROXIMITY_RIGHT_SIDE = HandLocationSide.CENTER
                            else:
                                if proximity_side < 0:
                                    #print ("right hand is to the ", HandLocationSide.RIGHT)
                                    
                                    GENERATED_PROXIMITY_RIGHT_SIDE = HandLocationSide.RIGHT
                                else:
                                    #print ("right hand is to the ", HandLocationSide.LEFT)
                                    
                                    GENERATED_PROXIMITY_RIGHT_SIDE = HandLocationSide.LEFT
                            
                            
                            #BOF visual representation of the proximity
                            
                            if show_graphs:
                                for idx, points in enumerate(list(chunks(data["people"][0]["pose_keypoints"], 3))):
                                    if idx == int(openpose_keypoints_pose.index(y_axis_labels[right_hand_near_idx]+"_x") / 3):
                                        cv2.circle(img,(int(points[0]),int(points[1])), 5, (255,255,255), -1)
                                ax_ = fig.add_subplot(rows, columns, columns*json_file_idx+1)
                                ax_.axis('off')
                                plt.imshow(img, interpolation='nearest')
                            #EOF visual representation of the proximity
                            
                            break
                            
                    right_hand_proximity_array[right_hand_near_idx] = 1.0
                    
                right_hand_proximity_arrays.append(copy.deepcopy(distances_matrix_[:,0]))
                #EOF detecting proximity to hamnosys category
                
                #BOF detecting proximity to hamnosys category
                found_hamnosys_proximity = False
                left_hand_proximity_array = copy.deepcopy(distances_matrix_[:,1])
                #print ("proxim array")
                #print (np.matrix(left_hand_proximity_array))
                while not found_hamnosys_proximity:
                    
                    if len(set(left_hand_proximity_array)) == 1 and list(set(left_hand_proximity_array))[0] == 1.0:
                        found_hamnosys_proximity = True
                        #print ("did not find any body part in proximity")
                        continue
                    
                    left_hand_near_idx = left_hand_proximity_array.argmin()
                    
                    for open_pose_key in open_pose_hamnosys_mapping.keys():
                        hamnosys_value = open_pose_hamnosys_mapping[open_pose_key]
                        
                        if hamnosys_value != None and y_axis_labels[left_hand_near_idx] == open_pose_key and left_hand_proximity_array[left_hand_near_idx] < body_part_proximity_threshold:
                            found_hamnosys_proximity = True
                            #print ("left hand is near", hamnosys_value)
                            
                            proximity_side = relative_point_location_to_line(float(distances_matrix[left_hand_near_idx][2]/frame_size[0]),
                                                                             float(distances_matrix[left_hand_near_idx][3]/frame_size[1]),
                                                                             float(distances_matrix[left_hand_near_idx][2]/frame_size[0]), 
                                                                             float((distances_matrix[left_hand_near_idx][3]+10)/frame_size[1]), 
                                                                             float(centroid_left_hand[0]/frame_size[0]),
                                                                             float(centroid_left_hand[1]/frame_size[1])) # negative - left
                            
                            #print ("proximity side", HandLocationSide.CENTER if abs(proximity_side) < hand_location_side_over_body_part else HandLocationSide.RIGHT if proximity_side < 0)
                            
                            if abs(proximity_side) < hand_location_side_over_body_part:
                                #print ("left hand is in the ", HandLocationSide.CENTER)
                                
                                GENERATED_PROXIMITY_LEFT_SIDE = HandLocationSide.RIGHT
                            else:
                                if proximity_side < 0:
                                    #print ("left hand is to the ", HandLocationSide.RIGHT)
                                    
                                    GENERATED_PROXIMITY_LEFT_SIDE = HandLocationSide.RIGHT
                                else:
                                    #print ("left hand is to the ", HandLocationSide.LEFT)
                                    
                                    GENERATED_PROXIMITY_LEFT_SIDE = HandLocationSide.LEFT
                            
                            
                            
                            #BOF visual representation of the proximity
                            if show_graphs:
                                for idx, points in enumerate(list(chunks(data["people"][0]["pose_keypoints"], 3))):
                                    if idx == int(openpose_keypoints_pose.index(y_axis_labels[left_hand_near_idx]+"_x") / 3):
                                        cv2.circle(img,(int(points[0]),int(points[1])), 5, (255,255,255), -1)
                                ax_ = fig.add_subplot(rows, columns, columns*json_file_idx+1)
                                ax_.axis('off')
                                plt.imshow(img, interpolation='nearest')
                            #EOF visual representation of the proximity
                            
                            break
                            
                    left_hand_proximity_array[left_hand_near_idx] = 1.0
                #EOF detecting proximity to hamnosys category
                
                #print ("prox arr what happened")
                #print (np.matrix(copy.deepcopy(distances_matrix_[:,0])))
                
                left_hand_proximity_arrays.append(copy.deepcopy(distances_matrix_[:,1]))
                #####################EOF Location#########################
                
                
                
                #####################BOF Orientation#########################
                
                img = cv2.imread(dataset_dir+"/"+json_file.split("_")[0]+".png")#[...,::-1]
                
                GENERATED_FINGER_ORIENTATION_RIGHT = None
                GENERATED_FINGER_ORIENTATION_LEFT = None
                
                GENERATED_PALM_ORIENTATION_RIGHT = None
                GENERATED_PALM_ORIENTATION_LEFT = None
                
                #BOF right hand
                start_x = (data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("radius_x")] + 
                          data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")]) / 2
                start_y = (data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("radius_y")] + 
                          data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")]) / 2
                palm_angle = normal_angle_to_line_right_hand((data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("radius_x")], data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("radius_y")]), 
                                     (data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")], data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")]))
                end_x = start_x + 50.0 * math.cos(palm_angle)
                end_y = start_y + 50.0 * math.sin(palm_angle)
                
                
                line_angle = math.degrees(angle_of_line((data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")], data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")]), (data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("radius_x")], data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("radius_y")])))
                #ax.text(2.5, -0.02, "line angle: "+str(line_angle)) #temp
                #print ("line angle", line_angle)
                
                if True:#show_graphs:
                    cv2.line(img, (int(data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("radius_x")]), int(data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("radius_y")])), (int(data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")]), int(data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")])), (0,0,255), 3)
                    cv2.circle(img,(int(data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")]), int(data["people"][0]["hand_right_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")])), 10, (255,255,255), -1)
                    cv2.line(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255,255,255), 3)
                #fig.add_subplot(rows, columns, columns*json_file_idx+3)
                #plt.imshow(img, interpolation='nearest')
                
                if line_angle >= -180 and line_angle < 0:
                    for a_direction in directions_dictionary["SOUTHERN_HEMISPHERE"]:
                        #print (a_direction)
                        if line_angle >= directions_dictionary["SOUTHERN_HEMISPHERE"][a_direction][0] and line_angle < directions_dictionary["SOUTHERN_HEMISPHERE"][a_direction][1]:
                            #print ("northern hemisphere", a_direction)
                            #ax.text(2.5, -0.02, "northern hemisphere" + str(a_direction))
                            
                            GENERATED_FINGER_ORIENTATION_RIGHT = a_direction
                else:
                    for a_direction in directions_dictionary["NORTHERN_HEMISPHERE"]:
                        #print (a_direction)
                        if line_angle >= directions_dictionary["NORTHERN_HEMISPHERE"][a_direction][0] and line_angle < directions_dictionary["NORTHERN_HEMISPHERE"][a_direction][1]:
                            #print ("southern hemisphere", a_direction)
                            #ax.text(2.5, -0.02, "southern hemisphere" + str(a_direction))
                            
                            GENERATED_FINGER_ORIENTATION_RIGHT = a_direction
                            
                #EOF right hand
                
                #BOF right palm
                if palm_angle >= -180 and palm_angle < 0:
                    for a_direction in directions_dictionary["SOUTHERN_HEMISPHERE"]:
                        #print (a_direction)
                        if palm_angle >= directions_dictionary["SOUTHERN_HEMISPHERE"][a_direction][0] and line_angle < directions_dictionary["SOUTHERN_HEMISPHERE"][a_direction][1]:
                            #print ("northern hemisphere", a_direction)
                            #ax.text(2.5, -0.02, "northern hemisphere" + str(a_direction))
                            
                            GENERATED_PALM_ORIENTATION_RIGHT = a_direction
                else:
                    for a_direction in directions_dictionary["NORTHERN_HEMISPHERE"]:
                        #print (a_direction)
                        if palm_angle >= directions_dictionary["NORTHERN_HEMISPHERE"][a_direction][0] and line_angle < directions_dictionary["NORTHERN_HEMISPHERE"][a_direction][1]:
                            #print ("southern hemisphere", a_direction)
                            #ax.text(2.5, -0.02, "southern hemisphere" + str(a_direction))
                            
                            GENERATED_PALM_ORIENTATION_RIGHT = a_direction
                #EOF right palm
                
                
                #BOF left hand
                start_x = (data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("radius_x")] + 
                          data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")]) / 2
                start_y = (data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("radius_y")] + 
                          data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")]) / 2
                palm_angle = normal_angle_to_line_left_hand((data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("radius_x")], data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("radius_y")]), 
                                     (data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")], data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")]))
                end_x = start_x + 50.0 * math.cos(palm_angle)
                end_y = start_y + 50.0 * math.sin(palm_angle)
                
                line_angle = math.degrees(angle_of_line((data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")], data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")]), (data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("radius_x")], data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("radius_y")])))
                #ax.text(2.5, -0.02, "line angle: "+str(line_angle)) #temp
                #print ("line angle", line_angle)
                
                if True:#show_graphs:
                    cv2.line(img, (int(data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("radius_x")]), int(data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("radius_y")])), (int(data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")]), int(data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")])), (0,0,255), 3)
                    cv2.circle(img,(int(data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_x")]), int(data["people"][0]["hand_left_keypoints"][openpose_keypoints_hand.index("middle_trapezium_y")])), 10, (255,255,255), -1)
                    cv2.line(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255,255,255), 3)
                #fig.add_subplot(rows, columns, columns*json_file_idx+3)
                #plt.imshow(img, interpolation='nearest')
                
                if line_angle >= -180 and line_angle < 0:
                    for a_direction in directions_dictionary["SOUTHERN_HEMISPHERE"]:
                        #print (a_direction)
                        if line_angle >= directions_dictionary["SOUTHERN_HEMISPHERE"][a_direction][0] and line_angle < directions_dictionary["SOUTHERN_HEMISPHERE"][a_direction][1]:
                            #print ("northern hemisphere", a_direction)
                            #ax.text(2.5, -0.02, "northern hemisphere" + str(a_direction))
                            
                            GENERATED_FINGER_ORIENTATION_LEFT = a_direction
                else:
                    for a_direction in directions_dictionary["NORTHERN_HEMISPHERE"]:
                        #print (a_direction)
                        if line_angle >= directions_dictionary["NORTHERN_HEMISPHERE"][a_direction][0] and line_angle < directions_dictionary["NORTHERN_HEMISPHERE"][a_direction][1]:
                            #print ("southern hemisphere", a_direction)
                            #ax.text(2.5, -0.02, "southern hemisphere" + str(a_direction))
                            
                            GENERATED_FINGER_ORIENTATION_LEFT = a_direction
                
                #EOF left hand
                
                
                #BOF left palm
                if palm_angle >= -180 and palm_angle < 0:
                    for a_direction in directions_dictionary["SOUTHERN_HEMISPHERE"]:
                        #print (a_direction)
                        if palm_angle >= directions_dictionary["SOUTHERN_HEMISPHERE"][a_direction][0] and line_angle < directions_dictionary["SOUTHERN_HEMISPHERE"][a_direction][1]:
                            #print ("northern hemisphere", a_direction)
                            #ax.text(2.5, -0.02, "northern hemisphere" + str(a_direction))
                            
                            GENERATED_PALM_ORIENTATION_LEFT = a_direction
                else:
                    for a_direction in directions_dictionary["NORTHERN_HEMISPHERE"]:
                        #print (a_direction)
                        if palm_angle >= directions_dictionary["NORTHERN_HEMISPHERE"][a_direction][0] and line_angle < directions_dictionary["NORTHERN_HEMISPHERE"][a_direction][1]:
                            #print ("southern hemisphere", a_direction)
                            #ax.text(2.5, -0.02, "southern hemisphere" + str(a_direction))
                            
                            GENERATED_PALM_ORIENTATION_LEFT = a_direction
                #EOF left palm
                
                if show_graphs:
                    ax__ = fig.add_subplot(rows, columns, columns*json_file_idx+3)
                    ax__.axis('off')
                    plt.imshow(img, interpolation='nearest')
                
                #####################EOF Orientation#########################
                
                
                #####################BOF Movement#########################
                GENERATED_MOVEMENT = detect_cut_off_points(json_file, json_file_idx, centroid_right_hand, centroid_left_hand, right_hand_proximity_array, left_hand_proximity_array)
                #####################EOF Movement#########################
                
                
                #####################BOF Handshapes#########################
                right_hand_features = data["people"][0]["hand_right_keypoints"]
                right_hand_features = np.array(right_hand_features)
                right_hand_features = (right_hand_features-right_hand_features.min())/(right_hand_features.max()-right_hand_features.min()) #values is because we are looking for min/max of the whole matrix, not min/max of a column
                
                left_hand_features = data["people"][0]["hand_left_keypoints"]
                left_hand_features = np.array(left_hand_features)
                left_hand_features = (left_hand_features-left_hand_features.min())/(left_hand_features.max()-left_hand_features.min())
                
                
                right_hand_class = left_hand_class = [None]
                if not any([math.isnan(x) for x in right_hand_features]):
                    right_hand_class = predict_handshape(trained_handshape_model, right_hand_features)
                if not any([math.isnan(x) for x in left_hand_features]):
                    left_hand_class = predict_handshape(trained_handshape_model, left_hand_features)
                #####################EOF Handshapes#########################
                
                generated_frame_lvfv = LVFV_STRUCTURE([HandShape.to_list()[right_hand_class[0]] if right_hand_class[0] != None else None, HandShape.to_list()[left_hand_class[0]] if left_hand_class[0] != None else None], 
                                                      [GENERATED_FINGER_ORIENTATION_RIGHT, GENERATED_FINGER_ORIENTATION_LEFT, GENERATED_PALM_ORIENTATION_RIGHT, GENERATED_PALM_ORIENTATION_LEFT], 
                                                      [GENERATED_PROXIMITY_RIGHT, GENERATED_PROXIMITY_RIGHT_SIDE, GENERATED_PROXIMITY_LEFT, GENERATED_PROXIMITY_LEFT_SIDE], 
                                                      GENERATED_MOVEMENT)
                
                '''ax_lvfv = fig.add_subplot(rows, columns, columns*json_file_idx+7)
                ax_lvfv.text(0, 0, str(generated_frame_lvfv))
                #fig.patch.set_visible(False)
                ax_lvfv.axis('off')
                fig.canvas.draw()
                #plt.show()
                #time.sleep(2.5)
                #clear_output()
                
                #print (generated_frame_lvfv)
                generated_lvfvs.append(generated_frame_lvfv)'''
                
                
                with grid.output_to(0, 0):
                    pylab.figure(figsize=(6, 6))
                    grid.clear_cell()
                    pylab.imshow(img, interpolation='none', extent=(0,13,0,10))
                    pylab.axis('off')
                    
                with grid.output_to(0, 1):
                    pylab.figure(figsize=(2, 2))
                    grid.clear_cell()
                    pylab.text(0,0,generated_frame_lvfv)
                    pylab.axis('off')
                
                time.sleep(0.2)
                
                update_frcnn_dataset(dataset_dir.split("/")[-1], json_file.split("_")[0], GENERATED_FINGER_ORIENTATION_RIGHT, GENERATED_FINGER_ORIENTATION_LEFT)
                
                
        else:
            print("some data is missing")
            
#plt.show()
