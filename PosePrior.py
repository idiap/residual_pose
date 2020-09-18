################################################################################
#Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#Written by Angel Martinez-Gonzalez <angel.martinez@idiap.ch>
#
#This file is part of ResidualPose.
#
#ResidualPose is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License version 3 as
#published by the Free Software Foundation.
#
#ResidualPose is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with ResidualPose. If not, see <http://www.gnu.org/licenses/>.
################################################################################



import os
import numpy as np
import cv2



def compute_conditional(x1_idx, x2_idx, x2_a, mean, S):                                                                                                                           
    start1=x1_idx*4
    ux1= mean[start1:(start1+4)]
    S11= S[start1:(start1+4), start1:(start1+4)]

    start2=x2_idx*4
    ux2= mean[start2:(start2+4)]
    S22= S[start2:(start2+4), start2:(start2+4)]

    S12= S[start1:(start1+4), start2:(start2+4)]
    S21= np.transpose(S12)

    S22inv= np.linalg.inv(S22)

    u= ux1 + (S12.dot(S22inv)).dot(x2_a-ux2)

    return u


def depth_first_search(start_node,adj_list, names, explored, path, keypoints_3d, 
                       limb_list, visibility, mean, S, keypoints_2d, img, depth, 
                       rescue_depth, lifting_fn=None, matrix_calibration=None):
    #################################
    explored[start_node]=True
    children= adj_list[start_node][1]
    pa1= limb_list[start_node][0]
    pa2= limb_list[start_node][1]
    if visibility[pa1]<1. or visibility[pa2]<1.:
        # print('Parent is not visible!!!', names[start_node])
        return

    #### get instance of the parent
    pa= keypoints_3d[pa2,:] - keypoints_3d[pa1,:]
    pa_l= np.linalg.norm(pa)
    pa_l= pa_l if pa_l>0. else 1.
    pa= pa/pa_l
    pa= np.array([a for a in pa]+[pa_l])
    #print(path)

    thisname= "DFS"

    # print("[INFO] ({}) depth size {} color size {}".format(thisname, depth.shape, img.shape))
    for c in children:
        img_cpy= img.copy()
        if not explored[c]:
            path_= path+'->'+names[c]
            cp1= limb_list[c][0]
            cp2= limb_list[c][1]

            #print(path_)
            #### cp2 is the extra joint, thus this is the one not visible
            #print(pa1, pa2, cp1, cp2)
            #print(names[c])

            if visibility[cp2]<1.:
                # print('[INFO] ({}) CHILD not visible! {}'.format(thisname, names[c]))
                ##### compute conditional
                u= compute_conditional(c, start_node, pa, mean, S)
                x= np.array(u[0:3])*u[3]
                nx= keypoints_3d[pa2,:] + x
                visibility[cp2]= 2.
                keypoints_3d[cp2,:]= nx
                # print('The child is', cp2, nx)
            if keypoints_3d[cp2, 2] < 0.001:
                # print('CHILD with Z equal to 0!', keypoints_3d[cp2, :], keypoints_2d[cp2,:])
                a,b= keypoints_2d[cp1,0:2],keypoints_2d[cp2,0:2]
                #### the vector that goes from b to a
                v= a-b
                dist2d=np.linalg.norm(v)
                # print("The distance is", dist2d)
                #if dist2d == 0.0:
                #    #### set the same depth as parent
                #    # print("Set same depth as parent")
                #    keypoints_3d[cp2,2]=keypoints_3d[cp1,2]
                #    continue
                space=dist2d/10.
                v/=dist2d
                z= rescue_depth
                # cv2.line(img_copy, (int(a[0]),int(a[1])), (int(b[0]),int(b[1])), [0,255,0])
                visibility[cp2]=3.
                keypoints_3d[cp2,:]= lifting_fn(int(round(b[0])), int(round(b[1])), z, matrix_calibration) 
                #for i in range(10):
                #    newp= np.round(v*space*(i+1) + b)
                #    x= int(newp[0])
                #    y= int(newp[1])
                #    # cv2.circle(img_copy, (x,y), 2, [0,255,0],-1)
                #    z= depth[y,x]
                #    if z>0.0:
                #        #keypoints_3d[cp2,:]= EVAL_lift_point(int(round(b[0])),int(round(b[1])),z)
                #        # keypoints_3d[cp2,:]= ITOP_lift_point(int(round(b[0])),int(round(b[1])),z)
                #        keypoints_3d[cp2,:]= lifting_fn(int(round(b[0])),int(round(b[1])),z, matrix_calibration)
                #        visibility[cp2]= 3.
                #        #print("NEW Z", z, keypoints_3d[cp2,:])
                #        #cv2.imshow('debug', img_copy)
                #        #cv2.waitKey()
                #        break


            depth_first_search(c,adj_list, names, explored, path_, keypoints_3d, 
                               limb_list, visibility, mean, S, keypoints_2d, 
                               img, depth, rescue_depth, lifting_fn, matrix_calibration)




