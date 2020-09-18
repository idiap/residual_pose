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


import sys
import os
import copy
import numpy as np
import cv2
import torch.nn.init as nninit
import torch.nn
import scipy.io

sys.path.append(os.path.dirname(__file__))


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],\
          [255, 255, 0], [170, 255, 0], [85, 255, 0],\
          [0, 255, 0], [0, 255, 85], [0, 255, 170],\
          [0, 255, 255], [0, 170, 255], [0, 85, 255],\
          [0, 0, 255], [85, 0, 255], [170, 0, 255],\
          [255, 0, 255], [255, 0, 170], [255, 0, 85]]



def draw_keypoints(img, keypoints, colors=colors, radius=3):                         
    for i in range(len(keypoints)):
        p=keypoints[i]
        x=int(round(p[0]))
        y=int(round(p[1]))
        visibility= p[2]

        ### Visibility flag has 3 posible values
        # 0 : annotation is not present
        # 1 : annotation is present and visible
        # 2 : annotation is present but not visible
        if visibility>0.:
            cv2.circle(img, (x,y), radius, colors[i], -1)


def draw_limbs(img, keypoints_, limbList_, colors=colors):
    rt=True
    for i in range(len(limbList_)):
        pair= limbList_[i]
        idx1 = pair[0]
        idx2 = pair[1]

        x1 = int(keypoints_[idx1][0])
        y1 = int(keypoints_[idx1][1])
        v1 = int(keypoints_[idx1][2])

        x2 = int(keypoints_[idx2][0])
        y2 = int(keypoints_[idx2][1])
        v2 = int(keypoints_[idx2][2])

        if v2>=1.0 and v1>=1.0:
            try:
                cv2.line(img, (x1,y1), (x2,y2), colors[i], 3)
            except OverflowError:
                print("OVERFLOW: imgshape {}, p1 {}, p2 {}".format(img.shape, (x1,y1), (x2,y2)))
                rt=False

    return rt


def load_mat_img(imgPath):                                                                     
    # Images in mat format are normally in milimeters
    try:
        img = scipy.io.loadmat(imgPath)
        img = img['depth']
                                                                                               
        if img.dtype!= np.float32:
            img= img.astype(np.float32)

        # img = img / 1000.0
        return img

    except:
        print('[ERROR] Image could not be loaded! Something went wrong:', sys.exc_info()[0])
        raise 



def convert_to_uchar(img, farPlane):                                                           
    charImg = clip_depth_image(img, farPlane)
    resFactor = 255.0/farPlane
    charImg = charImg*resFactor
    charImg = charImg.astype(np.uint8)

    return charImg


def clip_depth_image(img, farPlane=8.0):
    s = img.shape
    clipMat = img.copy()
    clipMat[clipMat < 0.0] = 0.0
    clipMat[clipMat > farPlane]=farPlane
    return clipMat


def load_depth_image(imgPath):                                                                       
    imgExt = imgPath.split('/')[-1].split('.')[-1]

    try:
        # Load blender image
        if imgExt == 'exr':
            return load_exr_img(imgPath)

        # Load matlab mat file
        elif imgExt == 'mat':
            return load_mat_img(imgPath)

        # Load color image
        elif imgExt == 'jpg' or imgExt == 'png':
            return cv2.imread(imgPath)
            # m= cv2.imread(imgPath, 0)
            m= m.astype(np.float32)/255.*8.
            return m

        # Load numpy file
        elif imgExt == 'npy':
            img = np.load(imgPath)
            img = np.array(img, dtype=np.float32)
            img = img/1000.0 # Transforming into meters                                        
            return img

        elif imgExt == 'tif' or imgExt == 'tiff':
            # print('[INFO] Loading images in tiff format')
            img = cv2.imread(imgPath, cv2.IMREAD_ANYDEPTH)
            img = np.array(img, dtype=np.float32)
            img = img / 1000.0
            return img
        else:
            return None
    except IOError:
        print('[ERROR]: Error loading file', imgPath)
        return None


def normal_init_(layer, mean_, sd_, bias):
    classname = layer.__class__.__name__
    # Only use the convolutional layers of the module
    if (classname.find('Conv') != -1 ) or (classname.find('Linear')!=-1):
        print('[INFO] (normal_init) Initializing layer {}'.format(classname))
        layer.weight.data.normal_(mean_, sd_)
        layer.bias.data.fill_(bias)


def normal_init(module, mean_=0, sd_=0.004, bias=0.0):
    moduleclass= module.__class__.__name__
    try:
        for layer in module:
            if layer.__class__.__name__ == 'Sequential':
                for l in layer:
                    normal_init_(l, mean_, sd_, bias)
            else:
                normal_init_(layer, mean_, sd_, bias)
    except TypeError:
        normal_init_(module, mean_, sd_, bias)


def xavier_init(layer):
    classname = layer.__class__.__name__
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
        print('[INFO] (xavier_init) Initializing layer {}'.format(classname))
        nninit.xavier_normal(layer.weight.data)
        # nninit.xavier_normal(layer.bias.data)
        if layer.bias is not None:
            layer.bias.data.zero_()


def layer_init(module):
    moduleclass= module.__class__.__name__
    try:
        for layer in module:
            if layer.__class__.__name__ == 'Sequential':
                for l in layer:
                    xavier_init(l)
            else:
                xavier_init(layer)
    except TypeError:
        xavier_init(module)


def ITOP_calibration_matrix():
    Cy=120
    Cx=160
    fx=1./0.0035
    fy=-1./0.0035

    mat_= np.eye(3, dtype=np.float32)
    mat_[0,0]= fx
    mat_[1,1]= fy

    mat_[0,2]= Cx
    mat_[1,2]= Cy

    return mat_


def ITOP_lift_point(x,y, depth, matrix_calibration):
    Z= depth[y,x]
    mat= matrix_calibration
    Cy= mat[1,2]
    Cx= mat[0,2]
    fx= mat[0,0]
    fy= mat[1,1]

    X=float(x-Cx)/fx * Z
    Y=float(y-Cy)/fy * Z

    return [float(X),float(Y),float(Z)]


def PAN_lift_point(x,y, depth, matrix_calibration):                                                             
    Z= depth[y,x]
    K= matrix_calibration
    fx= K[0,0]
    Cx= K[0,2]
    fy= K[1,1]
    Cy= K[1,2]

    X=float(x-Cx)/fx * Z
    Y=float(y-Cy)/fy * Z

    return [float(X),float(Y),float(Z)]

def lift_point(x, y, depth, matrix_calibration):
    # print(x,y, depth.shape)
    if isinstance(depth, np.ndarray):
        Z= depth[y,x]
    else:
        Z= depth

    K= matrix_calibration
    fx= K[0,0]
    Cx= K[0,2]
    fy= K[1,1]
    Cy= K[1,2]

    X=float(x-Cx)/fx * Z
    Y=float(y-Cy)/fy * Z

    return [float(X),float(Y),float(Z)]


class DepthNormalization:
    def __call__(self, img):
      farPlane= 8.0
      factor= 1.0/farPlane
      shift= 0.5
      trans= clip_depth_image(img, farPlane)
      trans=  trans*factor -shift
      trans= trans[np.newaxis, np.newaxis, :, :]

      trans= trans.astype(np.float32)

      return trans  



class ResizeImage:
    def __init__(self, shape=256):
        self.shape= shape

    def __call__(self, img):
        H,W= img.shape[0], img.shape[1]
        h=self.shape
        w=320
        # w= 340
        # w=int(round((h/H)*W))
        img_= cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC)

        return img_
        

class ToTensor:
    def __call__(self, np_img):
        return torch.from_numpy(np_img)

class SkeletonUtils:
    def __init__(self):
        # Definition of the joints that compose the model structure 
        # to follow for detection
        self.joint_id_to_name = {
          0: 'Head',
          1: 'Neck',
          2: 'R_Shoulder',
          3: 'L_Shoulder',
          4: 'R_Elbow',
          5: 'L_Elbow',
          6: 'R_Hand',
          7: 'L_Hand',
          8: 'Torso',
          9: 'R_Hip',
          10: 'L_Hip',
          11: 'R_Knee',
          12: 'L_Knee',
          13: 'R_Foot',
          14: 'L_Foot',
        }

        self.partList= [self.joint_id_to_name[k] for k in self.joint_id_to_name.keys()]

        # Joint map contains the index of the joint in the partList
        self.jointMap = {}
        i = 0
        for joint in self.partList:
            self.jointMap[joint] = i
            i+=1

        # Definition of the  limbs by defining what joints will compose each of them
        self.limbList  = [[self.jointMap["Neck"],        self.jointMap["Head"]],
                        [self.jointMap["Neck"],        self.jointMap["Torso"]],
                        [self.jointMap["Torso"],       self.jointMap["L_Hip"]],
                        [self.jointMap["L_Hip"],       self.jointMap["L_Knee"]],
                        [self.jointMap["L_Knee"],      self.jointMap["L_Foot"]],
                        [self.jointMap["Neck"],        self.jointMap["L_Shoulder"]],
                        [self.jointMap["L_Shoulder"],  self.jointMap["L_Elbow"]],
                        [self.jointMap["L_Elbow"],     self.jointMap["L_Hand"]],
                        [self.jointMap["Torso"],       self.jointMap["R_Hip"]],
                        [self.jointMap["R_Hip"],       self.jointMap["R_Knee"]],
                        [self.jointMap["R_Knee"],      self.jointMap["R_Foot"]],
                        [self.jointMap["Neck"],        self.jointMap["R_Shoulder"]],
                        [self.jointMap["R_Shoulder"],  self.jointMap["R_Elbow"]],
                        [self.jointMap["R_Elbow"],     self.jointMap["R_Hand"]]]


        self.parent_list= {self.jointMap["Head"]:        self.jointMap["Neck"],
                           self.jointMap["Neck"]:        self.jointMap["Torso"],
                           self.jointMap["L_Hip"]:       self.jointMap["Torso"],
                           self.jointMap["L_Knee"]:      self.jointMap["L_Hip"],
                           self.jointMap["L_Foot"]:      self.jointMap["L_Knee"],
                           self.jointMap["L_Shoulder"]:  self.jointMap["Neck"],
                           self.jointMap["L_Elbow"]:     self.jointMap["L_Shoulder"],
                           self.jointMap["L_Hand"]:      self.jointMap["L_Elbow"],
                           self.jointMap["R_Hip"]:       self.jointMap["Torso"],
                           self.jointMap["R_Knee"]:      self.jointMap["R_Hip"],
                           self.jointMap["R_Foot"]:      self.jointMap["R_Knee"],
                           self.jointMap["R_Shoulder"]:  self.jointMap["Neck"],
                           self.jointMap["R_Elbow"]:     self.jointMap["R_Shoulder"],
                           self.jointMap["R_Hand"]:      self.jointMap["R_Elbow"]}

        ##### list for bone children
        self.graph_limb_list= [[0, []],
                               [1, [0,2,8,5,11]],
                               [2, [3]],
                               [3, [4]],
                               [4, []],
                               [5, [6]],
                               [6, [7]],
                               [7, []],
                               [8, [9]],
                               [9, [10]],
                               [10,[]],
                               [11,[12]],
                               [12,[13]],
                               [13,[]]]

        self.bone_names=['Head', 'Torso',
                         'L_Hip', 'L_Leg', 'L_Chin', 'L_Shoulder', 'L_Arm', 'L_ForeArm', 
                         'R_Hip', 'R_Leg', 'R_Chin', 'R_Shoulder', 'R_Arm', 'R_ForeArm']




        self.node_names={'Head': 0, 'Neck':1, 'R_Shoulder':2,'L_Shoulder':3,
                          'R_Elbow':4, 'L_Elbow':5, 'R_Hand':6,'L_Hand':7,
                          'Torso':8,'R_Hip':9,'L_Hip':10, 'R_Knee':11,
                          'L_Knee':12,'R_Foot':13, 'L_Foot':14}


        self.children_list=[['Head'      , []], 
                            ['Neck'      , ['R_Shoulder','L_Shoulder','Head']], 
                            ['R_Shoulder', ['R_Elbow']], 
                            ['L_Shoulder', ['L_Elbow']],
                            ['R_Elbow'   , ['R_Hand']],
                            ['L_Elbow'   , ['L_Hand']],
                            ['R_Hand'    , []],
                            ['L_Hand'    , []],
                            ['Torso'     , ['R_Hip','L_Hip','Neck']], 
                            ['R_Hip'     , ['R_Knee']],
                            ['L_Hip'     , ['L_Knee']],
                            ['R_Knee'    , ['R_Foot']],
                            ['L_Knee'    , ['L_Foot']],
                            ['R_Foot'    , []],
                            ['L_Foot'    , []]]

        self.adj_list_idx=[[(i, data[0]), [self.node_names[k] for k in data[1]]]\
                                     for i, data in enumerate(self.children_list) ]

        self.graph_list=[[self.node_names[data[0]], 
                            [self.node_names[k] for k in data[1]]] 
                                for data in self.children_list]

        self.frame_defs=[\
           [],
           [self.jointMap['L_Shoulder'], self.jointMap['R_Shoulder']],
           [self.jointMap['R_Shoulder'], self.jointMap['R_Elbow']],
           [self.jointMap['L_Shoulder'], self.jointMap['L_Elbow']],
           [self.jointMap['R_Elbow'], self.jointMap['R_Hand']],
           [self.jointMap['L_Elbow'], self.jointMap['L_Hand']],
           [],
           [],
           [],
           [self.jointMap['R_Hip'], self.jointMap['R_Knee']],
           [self.jointMap['L_Hip'], self.jointMap['L_Knee']],
           [self.jointMap['R_Knee'], self.jointMap['R_Foot']],
           [self.jointMap['L_Knee'], self.jointMap['L_Foot']],
           [],
           []]








