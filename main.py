################################################################################
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angel Martinez-Gonzalez <angel.martinez@idiap.ch>,
#
# This file is part of ResidualPose.
#
# ResidualPose is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# ResidualPose is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ResidualPose. If not, see <http://www.gnu.org/licenses/>.
################################################################################

import os
import sys
import numpy as np
import json
import argparse
import cv2

import torch
import torchvision.transforms as transforms

import ResidualPose 
import HourGlass 
import Utils

import PosePrior

from CPMDepthPoseConstruction import ItopPoseConstructor




def prepare_maps(maps, pafs, out_shape, stride=8):
    maps_= maps.data.cpu().numpy()
    pafs_= pafs.data.cpu().numpy()

    maps_= np.squeeze(maps_)
    pafs_= np.squeeze(pafs_)

    maps_= np.transpose(maps_, (1,2,0))
    pafs_= np.transpose(pafs_, (1,2,0))

    maps_= cv2.resize(maps_, (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    maps_= cv2.resize(maps_, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_CUBIC)
    pafs_= cv2.resize(pafs_, (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    pafs_= cv2.resize(pafs_, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_CUBIC)

    return maps_, pafs_



def extract_prediction_(prediction, joint_map):
    result_list_=[]
    ### include the background as part
    n_parts= len(joint_map)+1


    for pred in prediction:
        detection= {'image_id' : -1,
                    'keypoints':[[0.0,0.0,0.0,0.0]]*(n_parts-1),
                    'score':pred['score']}

        detScore = 0.0
        for point in pred['points']:
            x, y, partId, score = point[0], point[1], point[3], point[2]
            partIdx = joint_map[partId]
            detection['keypoints'][partIdx]= [x, y, 1.0, score]

        result_list_.append(detection)

    return result_list_




@torch.no_grad()
def detect_2d_pose(img, hg_model, params):
    os.makedirs(params["output_path"], exist_ok=True)

    peaksTau= params["landmark_conf"]
    connectionTau= 0.05
    pose_constructor= ItopPoseConstructor(peaksTau, connectionTau)
    
    H,W= img.shape[0], img.shape[1]
    pose_utils= Utils.SkeletonUtils()
    transform_= transforms.Compose([Utils.ResizeImage(shape=256),
                                    Utils.DepthNormalization(),
                                    Utils.ToTensor()])

    ### mainly to get a canvas
    color_img = Utils.convert_to_uchar(img,8.0)                                     
    color_img = cv2.cvtColor(color_img, cv2.COLOR_GRAY2BGR) 

    #### forward pass
    img_inputs= transform_(img)
    output, feats= hg_model(img_inputs)
    partmaps, limbmaps= output[0], output[1]
    partmaps, limbmaps= prepare_maps(partmaps[-1], limbmaps[-1], (H,W), stride=4)

    ### total of predictect parts plus background
    n_parts= params["n_landmarks"]+1

    #### extract detections
    heatmaps_mat= np.concatenate((partmaps, limbmaps), axis=2)
    heatmaps_mat= np.delete(heatmaps_mat, (n_parts-1), axis=2)

    candidates, subset= pose_constructor.part_association(heatmaps_mat)
    canvas= pose_constructor.visualize_connection(color_img, candidates, subset)
    prediction= pose_constructor.extract_keypoints(candidates, subset)

    partmaps, limbmaps= output[0], output[1]
    partmaps, limbmaps= prepare_maps(partmaps[-1], limbmaps[-1], (H,W), stride=4)

    # heatmapsImg, pafs= Utils.visualize_confmaps(colorImg, 
    #                                             partmaps[:,:,0:15], 
    #                                             limbmaps)

    result_list_= extract_prediction_(prediction, pose_utils.jointMap)




    return result_list_, canvas



def verify_enough_landmarks(keypoints, joint_map):
    must= [joint_map["Neck"], joint_map["Torso"]]
    some= [joint_map["R_Shoulder"], joint_map["L_Shoulder"], 
                joint_map["R_Hip"], joint_map["L_Hip"]]

    ### check that visibility for the trunk limb is true
    if np.sum(keypoints[must,2]) != 2:
        return False

    ### check that we have also some of the other keypoints
    if np.sum(keypoints[some, 2]) <2 :
        return False

    return True
    




if __name__ =="__main__":
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    parser= argparse.ArgumentParser(description="Test 3d human pose estimation with residual pose networks.")
    parser.add_argument("--config_file", type=str, default="config/config_file.json",
        help="Json file with configuration parameters")
    parser.add_argument("--image_sample", type=str, default="",
        help="Input image to extract pose.")
    parser.add_argument("--output_path", type=str, default="",
        help="Path where to save results of detection.")

    args= parser.parse_args()


    config= json.load(open(args.config_file))
    config["output_path"]= args.output_path
    ## calibration matrix is a single matrix for all sequence in ITOP
    ## for PANOPTIC each viewpoint has its own camera matrix 
    ## for better reading these matrices have to be input from the config file
    matrix_calibration= np.array(config["matrix_calibration"])
    lifting_fn= Utils.lift_point


    ### configure the residual pose 3d regression model
    input_size= config["n_landmarks"]*3
    output_size= config["n_landmarks"]*3
    rp_model= ResidualPose.PoseRegressor3d(input_size=input_size,
                                           output_size=output_size,
                                           n_features=config["n_features"],
                                           n_landmarks=config["n_landmarks"],
                                           dropout=True)


    rp_model.load_state_dict(torch.load(config["pretrained_respose"], 
                             map_location=device))

    rp_model.to(device)
    rp_model.eval()

    ### configure the HG 2d pose estimation model
    hg_params= HourGlass.get_hg_parameters()
    hg_model= HourGlass.PoseMachine_HG(hg_params)

    hg_model.load_state_dict(torch.load(config["pretrained_hgpose"],
                             map_location=device))


    ### read depth image in metters
    img_depth= Utils.load_depth_image(args.image_sample)
    img_color= Utils.convert_to_uchar(img_depth, 8.0)
    img_color= cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)

    #######################################################
    #### Step (1) get 2d detections on depth image 
    #######################################################
    detections, canvas2d= detect_2d_pose(img_depth, hg_model, config)
    print("[INFO] # of 2d detections: {}".format(len(detections)))
    #cv2.imshow("2d detections", canvas2d)
    #cv2.waitKey()

    cv2.imwrite(os.path.join(config["output_path"], "canvas_2d.jpg"), canvas2d)
    #######################################################
    ### Step (2) perform lifting with camera parameters
    #######################################################
    for d in detections:
        pts3d=[]
        for pts in d["keypoints"]:
            pts3d.append(lifting_fn(int(round(pts[0])), 
                                    int(round(pts[1])), 
                                    img_depth,
                                    matrix_calibration))

        d["keypoints_3d"]= pts3d
    

    #######################################################
    ### Step (3) recover missing detections with pose prior
    #######################################################
    img_color_= img_color.copy()
    skeleton_traits= Utils.SkeletonUtils()
    skeleton_limbs= skeleton_traits.limbList
    skeleton_parts= skeleton_traits.partList
    skeleton_graph= skeleton_traits.graph_list
    skeleton_names= skeleton_traits.joint_id_to_name

    skeleton_bone_graph= skeleton_traits.graph_limb_list
    skeleton_bone_names= skeleton_traits.bone_names

    #### 
    uX= np.load(config["prior_mean"])
    S=  np.load(config["prior_var"])

    torso_id= 1 ## id of torso limb

    for det in detections:
        path= "Torso"
        is_explored= [False]*len(skeleton_bone_graph)

        keypoints_2d= np.array(det["keypoints"])
        keypoints_3d= np.array(det["keypoints_3d"])

        ### skeleton without enough landmarks
        if not verify_enough_landmarks(keypoints_2d, skeleton_traits.jointMap):
            continue

        visibility= keypoints_2d[:,2].copy() 
        rescue_depth= np.mean([x[2] for x in keypoints_3d if x[2]>0.]) 

        PosePrior.depth_first_search(torso_id, skeleton_bone_graph, 
            skeleton_bone_names, is_explored, path, keypoints_3d, 
                skeleton_limbs, visibility, uX, S, keypoints_2d, img_color, 
                    img_depth, rescue_depth, lifting_fn=lifting_fn, 
                        matrix_calibration= matrix_calibration)

        det["keypoints_3d"]= keypoints_3d.tolist()

        ### plot projections of 3d pose into 2d image
        X= np.transpose(keypoints_3d)
        n_keypoints= matrix_calibration.dot(X)
        z= n_keypoints[2,:] + 0.0000000001
        n_keypoints/=z
        n_keypoints= np.transpose(np.around(n_keypoints))

        Utils.draw_keypoints(img_color_, n_keypoints)
        Utils.draw_limbs(img_color_, n_keypoints, skeleton_limbs)

    #cv2.imshow("2d with prior", img_color_)
    #cv2.waitKey()
    cv2.imwrite(os.path.join(config["output_path"], "canvas_3d_prior.jpg"), img_color_)

    #######################################################
    ### Step (4) perform 3d pose regression
    #######################################################
    ### get normalization parameters
    input_stats= json.load(open(config["reg3d_input_stats"]))
    output_stats= json.load(open(config["reg3d_output_stats"]))

    reg3d_input_mean= np.array(input_stats["train_mean"])
    reg3d_input_sd= np.array(input_stats["train_sd"])

    reg3d_output_mean= np.array(output_stats["train_mean"])
    reg3d_output_sd= np.array(output_stats["train_sd"])

    with torch.no_grad():
        for det in detections:
            keypoints_2d= np.array(det["keypoints"])
            keypoints_3d= np.array(det["keypoints_3d"])

            ### skeleton without enough landmarks
            if not verify_enough_landmarks(keypoints_2d, skeleton_traits.jointMap):
                continue

            ### normalize the input pose with mean and standard deviation
            X= (keypoints_3d-reg3d_input_mean)/reg3d_input_sd
            X= X.reshape((input_size))
            X= X[np.newaxis,:]
            X= torch.from_numpy(X).float().to(device)

            ## regress the residual pose and unnormalize output
            y= rp_model(X)
            y= y.cpu().numpy().reshape((config["n_landmarks"],3))
            y= reg3d_output_mean + y*reg3d_output_sd

            ### close the shortcut to get the final 3d pose estimated
            keypoints_3d= y+keypoints_3d

            ### set the points to be saved
            det["keypoints_3d"]= keypoints_3d.tolist()

            ### plot projections of 3d pose into 2d image
            X= np.transpose(keypoints_3d)
            n_keypoints= matrix_calibration.dot(X)
            z= n_keypoints[2,:]
            n_keypoints/=z
            n_keypoints= np.transpose(np.around(n_keypoints))

            Utils.draw_keypoints(img_color, n_keypoints)
            Utils.draw_limbs(img_color, n_keypoints, skeleton_limbs)


    with open(os.path.join(config["output_path"],"pose_results.json"), "w") as file_:
        json.dump(detections, file_, indent=4)

    cv2.imwrite(os.path.join(config["output_path"], "canvas_3d_regressed.jpg"), img_color)
    print("Saved results in {}".format(args.output_path))
    cv2.imshow("Regressed 3d pose", img_color)
    cv2.waitKey(2000)





