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
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.init as nninit

sys.path.append(os.path.dirname(__file__))

import Utils


class LinearResidual(nn.Module):
    """
    Class that implements the residual module architecture with fully 
    connected layers.
    """

    def __init__(self, input_size=1024, output_size=1024, 
                 n_resmods=1, dropout=False):
        """
        input_size: int
            Dimension of the input vector.

        output_size: int
            Dimension of the ouput vector.

        n_resmods: int
            Number of inner layers before the shortcut connection.

        dropout: bool
            Flag to use or not dropout layers.
            
        """
        super(LinearResidual, self).__init__()
        thisname=self.__class__.__name__
        self.dropout_prob=0.5
        self.n_mods= n_resmods
        print('[INFO] ({}) Initializing module'.format(thisname))
        print('[INFO] ({}) Using dropout? {}'.format(thisname,dropout))
        print('[INFO] ({}) Dropout val {}'.format(thisname,self.dropout_prob))
        print('[INFO] ({}) N of inner layers {}'.format(thisname,n_resmods))

        layers=[nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(inplace=True)]

        if dropout:
            layers+=[nn.Dropout(p=self.dropout_prob)]

        
        for i in range(1, n_resmods):
            layers+= [nn.Linear(input_size, output_size),
                      nn.BatchNorm1d(output_size),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers+=[nn.Dropout(p=self.dropout_prob)]

        ### model as a sequential
        self.back= nn.Sequential(*layers)
        self.relu= nn.ReLU()

        ### initialize linear layers with a random Gaussian distribution
        Utils.normal_init(self.back)


    def forward(self, x):
        """
        Forward function

        x: torch.Tensor
            Input tensor.
        """
        y= self.back(x)
        x= self.relu(x+y)

        return x



class PoseRegressor3d(nn.Module):
    """
    Class to implement the 3d pose regressor architecture. The range of models of
    the 3d pose regerssor from any set of input vector representing 2d or 3d
    keypoints.
    """

    def __init__(self,input_size=15*2, output_size=15*3, 
                 n_features=1024, n_landmarks=15, 
                 n_resblocks= 3, ### this is equivalent as to have a triple_residual
                 dropout=False):
        """
        input_size: int
            Dimension of the input vector.

        output_size: int
            Dimension of the output vector.

        n_features: int
            Dimension of the feature that every residual module produce.

        n_landmarks: int
            Number of body landmarks that represents the ouput and input pose.

        n_resblocks: int
            Number of residual blocks contained in the regressor architecture.

        dropout: bool
            Flag defining if dropout layers are including in the architecture.
        """
        super(PoseRegressor3d, self).__init__()

        self.n_landmarks= n_landmarks
        self.input_size=  input_size
        self.output_size= output_size 
        self.high_features=n_features
        self.dropout_prob=0.5
        self.dropout= dropout
        self.n_resblocks= n_resblocks

        thisname=self.__class__.__name__
        print('[INFO] ({}) Model 2d lifting to 3d'.format(thisname))
        print('[INFO] ({}) Using dropout? {}'.format(thisname,dropout))
        print('[INFO] ({}) Input size {}'.format(thisname,self.input_size))
        print('[INFO] ({}) Output size {}'.format(thisname,self.output_size))
        print('[INFO] ({}) High features {}'.format(thisname,self.high_features))
        print("[INFO] ({}) # residual blocks {}".format(thisname, self.n_resblocks))
        
        ### init architecture and initialize with Gaussian distribution
        self.init_modules()
        Utils.normal_init(self)



    def init_modules(self):
        """
        Initialize the different components of the architecture of the
        residual module.
        """
        print('[INFO] ({}) Init model'.format(self.__class__.__name__))
        thisname= self.__class__.__name__

        #### define the front architecture
        self.front= self.get_linear_module(self.input_size, 
                                           self.high_features, 
                                           dropout=self.dropout)
        ### Initialize main body of the regressor
        self.init_backbone()
        ### pose regressor should always be linear
        self.back= nn.Linear(self.high_features, self.output_size)


        ### initialize values with a random normal distribution
        Utils.normal_init(self.front)
        Utils.normal_init(self.back)
        


    def init_backbone(self):
        """
        Initialize the main body of the regressor architecture that can contain
        different and multiple inner residual blocks 
        """

        ### build the architecture by stacking residual modules
        the_modules=[]
        for i in range(self.n_resblocks):
            the_modules.append(LinearResidual(dropout=self.dropout, 
                                              input_size=self.high_features, 
                                              output_size=self.high_features))


        self.middle= nn.Sequential(*the_modules)


    def get_linear_module(self, input_features, output_features, dropout=False):
        """
        Generates the architecture of the first lifting module architecture.

        input_features: int
            Dimension of the input vector.

        output_features: int
            Dimension of the output vector.

        dropout: bool
            Flag to determine if use dropout layers or not
        """
        layers=[nn.Linear(input_features, output_features),
                nn.BatchNorm1d(output_features),
                nn.ReLU(inplace=True)]
        if dropout:
            layers+=[nn.Dropout(p=self.dropout_prob)]
        return nn.Sequential(*layers)


    def forward(self, x):
        """
        Forward pass of the architecture.

        x: torch.Tensor
            Input vector
        """
        x= self.front(x)
        x= self.middle(x)
        x= self.back(x)

        return x



if __name__ == "__main__":
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser= argparse.ArgumentParser()
    parser.add_argument("--n_landmarks", type=int, default=15,
         help="Number of body landmarks")
    parser.add_argument("--n_features", type=int, default=1024,
        help="Number of features in regressor.")
    parser.add_argument("--backbone_type", type=str, default="triple_residual",
        help="Architecture of NN regressor")
    parser.add_argument("--n_resblocks", type=int, default=3,
        help="Number of residual blocks in the architecture.")
    parser.add_argument("--regressor", type=str, default="Linear",
        help="Regressor output layer architecture.")
    parser.add_argument("--pretrained_model", type=str, default="",
        help="Pretrained model")
    parser.add_argument("--config_file", type=str, default="",
        help="Configuration file with all required parameters.")



    args= parser.parse_args()
    input_size= args.n_landmarks*3
    output_size= args.n_landmarks*3


    pm_model= PoseRegressor3d(input_size=input_size,
                              output_size=output_size,
                              n_features=args.n_features,
                              n_landmarks=args.n_landmarks,
                              regressor=args.regressor,
                              dropout=True)

    thisname="Main"
    model_parameters = filter(lambda p: p.requires_grad, pm_model.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    print('[INFO] ({}) Total number of trainable parameters: {}'.format(thisname, nparams))


    pm_model.to(device)
    pm_model.eval()


    import tqdm

    with torch.no_grad():
        for i in tqdm.tqdm(range(1000)):
            a= torch.FloatTensor(1, 45).normal_()
            a= a.to(device)


            x= pm_model(a)




