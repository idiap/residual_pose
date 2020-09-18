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
import cv2
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()

        self.mid_channels= input_channels//2 if input_channels > 1 else output_channels 

        layers=[nn.Conv2d(input_channels, self.mid_channels, kernel_size=1, stride=1, padding=0),
          nn.BatchNorm2d(self.mid_channels),
          nn.ReLU(inplace=True),

          nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(self.mid_channels),
          nn.ReLU(inplace=True),

          nn.Conv2d(self.mid_channels, output_channels, kernel_size=1, stride=1, padding=0),
          nn.BatchNorm2d(output_channels)]

        self.bn= nn.BatchNorm2d(output_channels)
        self.relu= nn.ReLU(inplace=True)

        self.body=nn.Sequential(*layers)

        self.shortcut = None                                                                                               
        if input_channels != output_channels:
            self.shortcut = nn.Conv2d(in_channels=input_channels,\
                                      out_channels=output_channels,\
                                      kernel_size=1, \
                                      stride=1, \
                                      padding=0)



    def forward(self, x):
        res=x
        out= self.body(res)

        if self.shortcut is not None:
            res = self.shortcut(res)

        y= out+res

        y= self.bn(y)
        y= self.relu(y)

        return y



class HourGlass(nn.Module):
    def __init__(self, params):
        super(HourGlass, self).__init__()

        self.params= params
        thisname=self.__class__.__name__ 
        for k, v in  self.params.items():
            print('[INFO] ({}) {}: {}'.format(thisname, k, v))

        self.residual_block= ResidualBlock

        self.latent= nn.Sequential(*[self.residual_block(self.params['hg_across_channels'], self.params['hg_across_channels']),
                                     self.residual_block(self.params['hg_across_channels'], self.params['hg_across_channels']),
                                     self.residual_block(self.params['hg_across_channels'], self.params['hg_across_channels'])])

        self.front = nn.ModuleList()
        self.maxpools= nn.ModuleList()
        self.skip_connections= nn.ModuleList()

        self.make_maxpoolings()
        self.make_fronts()
        self.make_skip_connections()



    def make_maxpoolings(self):
        for i in range(4):
            self.maxpools.append(nn.MaxPool2d(kernel_size=2, stride=2))



    def make_front_(self, n_input, n_output):
        layers= [self.residual_block(n_input, n_output),
                 self.residual_block(n_output, n_output),
                 self.residual_block(n_output, n_output)]

        return nn.Sequential(*layers)


    def make_fronts(self):
        n_output= self.params['hg_across_channels']
        for i in range(4):
            n_input= self.params['hg_input_channels'] if i==0 else self.params['hg_across_channels']
            self.front.append(self.make_front_(n_input, n_output))
            n_input*=2


    def make_skip_connection_(self, n_input):
        layers= [self.residual_block(n_input, n_input),
                 self.residual_block(n_input, n_input),
                 self.residual_block(n_input, n_input)]

        return nn.Sequential(*layers)


    def make_skip_connections(self):
        n_input= self.params['hg_across_channels']

        for i in range(4):
            # print('[INFO] Input size skip 1', n_input)
            self.skip_connections.append(self.make_skip_connection_(n_input))


    def forward(self, x):
        #### downsamplings
        skip_inputs=[]
        out= x
        for i in range(4):
            out= self.front[i](out)
            skip_inputs.append(out)
            out= self.maxpools[i](out)
            # print('Down {} {}'.format(i,out.size()))

        ### them lowest resolution
        Z= self.latent(out)
        # print('Latent {}'.format(Z.size()))

        skip_outputs= []
        for i in range(4):
            skip= self.skip_connections[i](skip_inputs[i])
            skip_outputs.append(skip)
            # print('skip {} {}'.format(i,skip.size()))

        up1= F.interpolate(Z, scale_factor=2)
        # print(up1.size(), skip_outputs[-1].size())
        up= up1+skip_outputs[-1]

        j=2
        for i in range(3):
            up_= F.interpolate(up, scale_factor=2)
            up__= skip_outputs[j]
            # print("{} {} ".format(up_.size(), up__.size()))
            up=up_+up__
            j-=1


        return up




class PoseMachine_HG(nn.Module):
    def __init__(self, params):
        super(PoseMachine_HG,self).__init__()
        thisname= self.__class__.__name__
        self.params=params

        for k, v in  self.params.items():
            print('[INFO] ({}) {}: {}'.format(thisname, k, v))

        ####
        self.residual_block= ResidualBlock

        self.front= [nn.Conv2d(self.params['input_channels'], self.params['front_channels'], kernel_size=7, stride=2, padding=3),
                     nn.BatchNorm2d(self.params['front_channels']),
                     nn.ReLU(inplace=True),
                     nn.MaxPool2d(kernel_size=2, stride=2),
                     self.residual_block(self.params['front_channels'], self.params['front_channels']),
                     self.residual_block(self.params['front_channels'], self.params['front_channels']),
                     self.residual_block(self.params['front_channels'], self.params['hg_input_channels'])]

        self.front= nn.Sequential(*self.front)

        self.hg= nn.ModuleList()
        self.scores= nn.ModuleList()
        self.scores_= nn.ModuleList()

        self.refinement= nn.ModuleList()
        self.refinement_= nn.ModuleList()

        if self.params['predict_vectors']:
            self.n_limbs= self.params['n_limbs']
            self.v_scores= nn.ModuleList()
            self.v_scores_= nn.ModuleList()
            self.v_refinement= nn.ModuleList()
            self.v_refinement_= nn.ModuleList()


        for i in range(self.params['n_stages']):
            self.hg.append(HourGlass(params))

        self.make_scores()
        self.make_refinement()


        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nparams = sum([np.prod(p.size()) for p in model_parameters])
        print('[INFO] ({}) This module has {} parameters!'.format(thisname, nparams))





    def make_scores(self):
        for i in range(self.params['n_stages']):
            conv= nn.Conv2d(self.params['hg_across_channels'], self.params['n_parts'], kernel_size=1, stride=1, padding=0)
            conv_= nn.Conv2d(self.params['n_parts'], self.params['hg_input_channels'], kernel_size=1, stride=1, padding=0)

            self.scores.append(conv)
            self.scores_.append(conv_)

            if self.params['predict_vectors']:
                conv= nn.Conv2d(self.params['hg_across_channels'], self.params['n_limbs'], kernel_size=1, stride=1, padding=0)
                conv_= nn.Conv2d(self.params['n_limbs'], self.params['hg_input_channels'], kernel_size=1, stride=1, padding=0)

                self.v_scores.append(conv)
                self.v_scores_.append(conv_)



    def make_refinement(self):
        for i in range(self.params['n_stages']):
            block= [self.residual_block(self.params['hg_across_channels'], self.params['hg_across_channels']),
                    self.residual_block(self.params['hg_across_channels'], self.params['hg_across_channels']),
                    self.residual_block(self.params['hg_across_channels'], self.params['hg_across_channels']),
                    nn.Conv2d(self.params['hg_across_channels'], self.params['hg_across_channels'], kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(self.params['hg_across_channels']),
                    nn.ReLU(inplace=True)]

            block_= nn.Conv2d(self.params['hg_across_channels'], self.params['hg_input_channels'], kernel_size=1, stride=1, padding=0)

            self.refinement.append(nn.Sequential(*block))
            self.refinement_.append(block_)



    def forward(self, x):
        #print('[INFO] Size 1 ', x.size())
        in_features= self.front(x)

        score_list=[None]*self.params['n_stages']
        v_score_list=[None]*self.params['n_stages']

        for i in range(self.params['n_stages']):
            # print('[INFO] Processing stage {}'.format(i))
            out= self.hg[i](in_features)
            out= self.refinement[i](out)

            ### compute score of part maps
            score= self.scores[i](out)
            score_list[i]= score
            score_=self.scores_[i](score)

            ### if the model predict vectors
            if self.params['predict_vectors']:
                v_score= self.v_scores[i](out)
                v_score_list[i]= v_score
                v_score_= self.v_scores_[i](v_score)
                score_= score_+v_score_


            ref_ = self.refinement_[i](out)
            in_features= score_ + ref_ + in_features

        ## predict vectors and maps
        if self.params['predict_vectors']:
            return (score_list, v_score_list), None

        ## predict only maps
        return score_list



    def init_fine_tuning(self, pretrained_state_dict):
        print('[INFO] (%s) Doing fine tune from pretrained model' % (self.__class__.__name__))
        self_dict = self.state_dict()
        match_dict = {k: v for k, v in pretrained_state_dict.items() if (k in self_dict)}

        for k in match_dict.keys():
            print('[INFO] (%s) Matched layer to finetune: %s' % (self.__class__.__name__, k))

        if len(match_dict.keys())==0:
            print('[INFO] (%s) No matched keys for finetune!!!!' % (self.__class__.__name__))

        self_dict.update(match_dict)
        self.load_state_dict(self_dict)




def get_hg_parameters():
    """
    Returns the default parameters used to generate the HG architecture.
    """
    return {### channels produced by the first layers before HG
            'front_channels': 64,
            ### channels input to the HG blocks
            'hg_input_channels': 128,
            ### channels produced across the HG
            'hg_across_channels': 256,
            ### how many hourglass modules
            'n_stages': 2,
            ### how many landmarks including background
            'n_parts': 16,   
            ### Number of limb components (n_parts-1)*2
            'n_limbs': 28,
            ### channels of the image
            'input_channels':1,
            ### Predict limb vectors 
            'predict_vectors':True}


        
if __name__ == "__main__":

    params= get_hg_parameters()
    PoseMachine_HG(params)



