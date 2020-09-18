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
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
import copy
import math
import scipy.io
# from sets import Set
import sys

sys.path.append('./../')
import Utils as CpmUtils



##
# @brief Base class to perform the pose construction. The class
# implements the search by computing the confidence of each of 
# the body parts and body limbs provided the part confidence map
# and the part affinity fields.
# The implementation was made an interface class to be able to
# use the same code for different types of body structures. To 
# use different body structure a child class has to inherit from
# 
# The methods implemented in this class were directly translated 
# from the original matlab implementation provided by the author.
# 
#
class PoseConstructor:
    def __init__(self, peaksTau, connectionTau):
        self.params = None

        self.peaksTau       = peaksTau
        self.connectionTau  = connectionTau
        self.doVerbose      = False
        self.doOldWay       = False
        self.theSearchIndex = 0
        self.min_n_parts    = 3 ## for comprising with openpose

        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],\
                       [255, 255, 0], [170, 255, 0], [85, 255, 0],\
                       [0, 255, 0], [0, 255, 85], [0, 255, 170],\
                       [0, 255, 255], [0, 170, 255], [0, 85, 255],\
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],\
                       [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 170], [255, 0, 85]]


        # To be defined by the child class
        self.partList = None
        self.limbSeq  = None
        self.init_limbs_map()

        self.numParts = len(self.partList)
        self.numLimbs = len(self.limbSeq)

    def verbose(self, caller=None):
        if caller==None:
            caller = self.__class__.__name__

        print('[INFO] (%s)' % (caller))
        print('[INFO] (%s) Peaks tau %f' % (caller, self.peaksTau))
        print('[INFO] (%s) Connection tau %f' % (caller, self.connectionTau))
        print('[INFO] (%s) Number of parts %d' % (caller, self.numParts))
        print('[INFO] (%s) Number of limbs %d' % (caller, self.numLimbs))

    def get_num_parts(self):
        return self.numParts

    def get_numb_limbs(self):
        return self.numLimbs

    @abstractmethod
    def init_limbs_map(self):
        pass

    # TODO Is there a way to localize  non zero coordinates?
    def find_nonzero(self, mat):
        s = mat.shape
        X = []
        Y = []

        for i in range(s[0]):
          for j in range(s[1]):
            if mat[i, j]> 0.0:
              X.append(j)
              Y.append(i)

        return X, Y


    # TODO What's the best way to do the non maximum supression?
    def find_peaks(self, heatmap, thre):
        # print('[INFO]({}) find_peaks with threshold {}'.format(self.__class__.__name__, thre))
        # May use a preprocess step to get rid of noise before thresholding
        ret, map_smooth = cv2.threshold(heatmap, thre, 255, cv2.THRESH_TOZERO)

        # map_aug = -1*zeros(size(map_smooth,1)+2, size(map_smooth,2)+2);
        map_aug = np.zeros(shape=(heatmap.shape[0]+2, heatmap.shape[1]+2))
        map_aug1 = np.zeros(shape=(heatmap.shape[0]+2, heatmap.shape[1]+2))
        map_aug2 = np.zeros(shape=(heatmap.shape[0]+2, heatmap.shape[1]+2))
        map_aug3 = np.zeros(shape=(heatmap.shape[0]+2, heatmap.shape[1]+2))
        map_aug4 = np.zeros(shape=(heatmap.shape[0]+2, heatmap.shape[1]+2))
        
        s = map_aug.shape
        map_aug[1:(s[0]-1),  1:(s[1]-1)] = map_smooth
        map_aug1[1:(s[0]-1), 0:(s[1]-2)] = map_smooth
        map_aug2[1:(s[0]-1), 2:s[1]]   = map_smooth
        map_aug3[0:(s[0]-2), 1:(s[1]-1)] = map_smooth
        map_aug4[2:s[0],   1:(s[1]-1)] = map_smooth
        
        peakMap = (map_aug > map_aug1) & (map_aug > map_aug2) & (map_aug > map_aug3) & (map_aug > map_aug4)
        s = peakMap.shape
        peakMap = peakMap[1:(s[0]-1), 1:(s[1]-1)]
        # X, Y = self.find_nonzero(peakMap)
        zeroWhere = np.where(peakMap > 0.0)
        X, Y = list(zeroWhere[1]), list(zeroWhere[0])

        if len(X) == 0:
            return None, None, None

        score = np.zeros(shape=(len(X),1))

        for i in range(len(X)):
            score[i,0] = heatmap[Y[i],X[i]]

        flag = np.ones(shape=(1, len(X)))

        delIdx = [];
        if self.doOldWay:
            # Find peaks very close to each other and remove them
            for i in range(len(X)):
                if flag[0,i] > 0:
                    for j in range(i+1, len(X)):
                        norm = (X[i]-X[j])*(X[i]-X[j]) + (Y[i]-Y[j])*(Y[i]-Y[j])
                        # TODO WTF with this distance?
                        if math.sqrt(norm) <= 6: 
                            flag[0,j] = 0.0
                            delIdx.append(j)
        else:
            #delIdx = Set()
            delIdx= set()
            match = np.zeros(shape=(len(X), len(X)))
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    norm = (X[i]-X[j])*(X[i]-X[j]) + (Y[i]-Y[j])*(Y[i]-Y[j])
                    match[i,j] = math.sqrt(norm)
                    if match[i,j] <= 6:
                        delIdx.add(j)

            delIdx = list(delIdx)

        # Transform list into float arrays
        X = np.asarray(X, dtype=np.float64, order='F').reshape((len(X),1))
        Y = np.asarray(Y, dtype=np.float64, order='F').reshape((len(Y),1))
        # score = np.asarray(score, dtype=np.float64, order='F').reshape((score.shape[0],1))

        X = np.delete(X, delIdx, axis=0)
        Y = np.delete(Y, delIdx, axis=0)
        score = np.delete(score, delIdx, axis=0)

        return X, Y, score



    def visualize_peaks(self, img, candidates):
        # Visualize peaks
        # print('[INFO] size of concatenation:', candidates.shape)
        # print(candidates) 

        for i in range(candidates.shape[0]):
            cv2.circle(img, (int(candidates[i, 0]), int(candidates[i,1])), 5, [0,255,255], -1)

        # To compare against matlab peaks candidates
        lmat = np.loadtxt('./../process_images/other_test/matlab_output/peaks.txt')
        # print('[INFO] Shape of loaded mat', lmat.shape)

        for i in range(lmat.shape[0]):
            cv2.circle(img, (int(lmat[i, 0]), int(lmat[i,1])), 2, [255,255,0], -1)

        cv2.imshow('Peaks.jpg', img)
        cv2.waitKey()


    def get_candidates(self, heatmaps, peaksTau):
        candidates = None
        maximum = []
        count = 0
        totalParts = self.numParts

        # Compute part candidates from all heatmaps
        for i in range(totalParts):
            # print 'The index ', i
            self.theSearchIndex = i
            X, Y, score = self.find_peaks(heatmaps[:,:,i], peaksTau)
            # if i == 1:
            #  print 'Printing the index'
            #  print zip(X, Y)

            if X is None or Y is None or score is None:
                maximum.append(np.array(()))
                continue

            c = np.concatenate((X,Y,score, np.ones(shape=(X.shape[0],1))*i), axis=1)
            if candidates is None:
                candidates = copy.deepcopy(c)
            else:
                candidates = np.concatenate((candidates, c), axis=0)

            temp = range(count, len(X)+count)
            temp = np.asarray(temp, dtype=np.float64, order='C').reshape((len(X), 1))
            maximum.append(np.concatenate((X, Y, score, temp), axis=1))
            count = count + len(X)

        # visualize_peaks(img, candidates)
        # print('[INFO] The size of maximum is', len(maximum))

        return candidates, maximum

    def get_weighted_connection(self, score_mid, candA, candB, height, connectionTau, inv=False):
        nA = candA.shape[0]
        nB = candB.shape[0]

        temp     = np.array(())
        midPoint = np.zeros(shape=(2,2))

        # Weight the connections between the candidates
        for i in range(nA): #  = 1:nA
            for j in range(nB): # = 1:nB
                #print('[INFO] Testing the conection of canditate A', candA[i,[0,1]])
                #print('[INFO] Testing the conection of canditate B', candB[j,[0,1]])

                # Calculate the mid point between the candidates
                midPoint[0,:] = np.around(candA[i,[0,1]]*0.5 + candB[j,[0,1]]*0.5);
                midPoint[1,:] = midPoint[0,:];
               
                #print('[INFO] The mid point is', midPoint)
              
                # The vector that joints both candidates 
                vec = candB[j,[0,1]] - candA[i,[0,1]]
                norm_vec = math.sqrt(vec[0]**2+vec[1]**2);

                if inv == True:
        #            print('INVERTING THE VECTOR')
                    vec = vec *-1.0

                # Avoid zero division
                if norm_vec > 0:
                  vec = vec/norm_vec;
                  
                  score = vec[0]*score_mid[int(midPoint[0,1]), int(midPoint[0,0]),0] +\
                          vec[1]*score_mid[int(midPoint[1,1]), int(midPoint[1,0]),1];
                else:
                  score = -1e10
                
                height_n = height;

                suc_ratio = 0;
                mid_score = np.zeros(shape=(1))
                mid_num = 10; 

        #        print('[INFO] The score of such connection', score)
        #        print('[INFO] The vector is ', vec)

                if score > -100.0: #&& norm_vec < height_n #0.01
                  p_sum = 0
                  p_count = 0
                  
                  x = np.linspace(candA[i,0], candB[j,0], num=mid_num)
                  y = np.linspace(candA[i,1], candB[j,1], num=mid_num)

                  for lm in range(mid_num): #= 1:mid_num
                    mx = int(round(x[lm]))
                    my = int(round(y[lm]))
                    pred = np.squeeze(score_mid[my, mx, [0,1]]);
                    #print('[INFO] The pred is', pred)
                    score = vec[1]*pred[1] + vec[0]*pred[0];
                    #print('[INFO] The score %f and the thresh2 %f' %(score, tau2))

                    if score > connectionTau: # norm(pred) > 0.01
                      p_sum = p_sum + score
                      p_count = p_count+1
                      #print('[INFO] ===== The new values', p_sum, p_count)
                  
                  suc_ratio = float(p_count)/float(mid_num);

                  # Avoid zero division (MATLAB does something weird) 
                  if p_count > 0:
                    mid_score[0] = p_sum/p_count + min(height_n/norm_vec-1.0, 0)
                  else:
                    mid_score[0] = -1e10

                #print('[INFO] The mid_score is %f and suc_ratio %f' %(mid_score[0], suc_ratio))
                if mid_score[0] > 0 and suc_ratio > 0.8: #0.7 #second threshold
                  # score = sum(mid_score);
                  score = mid_score[0]
                  # parts score + connection score
                  score_all = score + candA[i, 2] + candB[j, 2]
                  l = [i,j,score, score_all]
                  if temp.shape[0] == 0:
                    temp = np.asarray(l).reshape((1,4))
                  else:
                    temp = np.concatenate((temp, np.asarray(l).reshape(1,4)), axis=0)

        ### Finishing weighting the connections between the candidates
        ## select the top num connection, assuming that each part occur only once
        # sort rows in descending order 
        if temp.shape[0] > 0:
            # temp = sortrows(temp,-3); #based on connection score
            #temp = sortrows(temp,-4); #based on parts + connection score
            temp = temp[(-temp[:,2]).argsort(axis=0)] # Based on connection score

        return temp


    def part_association(self, heatmaps):
        peaksTau = self.peaksTau
        connectionTau = self.connectionTau

        height = heatmaps.shape[0]/2;
        width  = heatmaps.shape[1];
        kpt_num = len(self.partList) + 2;# Why summing 2?

        subset = np.array(())
        connection = [None]*self.mapIdx.shape[0]

        # Get part candidates by applying non maximum supression
        candidates, maximum = self.get_candidates(heatmaps, peaksTau)

        # find the parts connection and cluster them into different subset
        for k in range(self.mapIdx.shape[0]):
            # get the paf maps
#            print(self.mapIdx[k,:])
            score_mid = heatmaps[:,:,self.mapIdx[k,:]]

            # Take the candidates that make a limb
            candA = maximum[self.limbSeq[k, 0]];
            candB = maximum[self.limbSeq[k, 1]];

            nA = candA.shape[0]
            nB = candB.shape[0]
            indexA = self.limbSeq[k,0]
            indexB = self.limbSeq[k,1]

            if self.doVerbose:
              print('[INFO] ====== Associating parts %d and %d ================ ' % (indexA, indexB))
              print('[INFO] ====== Associating parts %s  and %s =============== ' % (self.partList[indexA], self.partList[indexB]))
              print('[INFO] CandA')
              print(candA)
              print('[INFO] CandB')
              print(candB)
              print('[INFO] The k is', k)
              try:
                input('[INPUT] Press enter to continue...')
              except SyntaxError:
                pass
              print('[INFO] Number of A candidates %d and B candidates %d' % (nA, nB))
            
            # add parts into the subset in special case
            if nA ==0 and nB ==0:
              continue
            elif nA == 0:
              for i in range(nB):#    i = 1:nB
                num = 0;

                s = subset.shape[0] if subset is not None else 0
                for j in range(s): # 1:size(subset,1)
                  if subset[j, indexB] == candB[i,3]:
                    num = num+1;
                    continue;

                # if find no partB in the subset, create a new subset
                if num==0:
                  if subset.shape[0] == 0:
                    subset = np.ones(shape=(1, kpt_num))*-1.0
                  else:
                    subset = np.concatenate((subset, np.ones(shape=(1, kpt_num))*-1.0 ), axis=0)

                  subset[subset.shape[0]-1, indexB] = candB[i, 3]
                  subset[subset.shape[0]-1, subset.shape[1]-1] = 1
                  subset[subset.shape[0]-1, subset.shape[1]-2] = candB[i,2]

              continue
            elif nB == 0:       
              for i in range(nA): # = 1:nA
                num = 0
                s = subset.shape[0] if subset is not None  else 0
                for j in range(s): # 1:size(subset,1)
                  if subset[j, indexA] == candA[i,3]:
                    num = num+1
                    continue

                # if find no partA in the subset, create a new subset
                if num == 0:
                  if subset.shape[0] == 0:
                    subset = np.ones(shape=(1, kpt_num))*-1.0
                  else:
                    subset = np.concatenate((subset, np.ones(shape=(1, kpt_num))*-1.0), axis=0)

                  subset[subset.shape[0]-1, indexA] = candA[i, 3]
                  subset[subset.shape[0]-1, subset.shape[1]-1] = 1
                  subset[subset.shape[0]-1, subset.shape[1]-2] = candA[i,2]

              continue
         
            inv = True if indexB == 15 or indexB == 16 else False
            inv = False
                
   
            temp = self.get_weighted_connection(score_mid, candA, candB, height, connectionTau, inv)

            if temp.shape[0] == 0:
              continue

            if self.doVerbose:
              print('[INFO] The temp size', temp.shape)
              print('[INFO] The temp is')
              print(temp)

            # set the connection number as the samller parts set number
            num = min(nA, nB)
            cnt = 0
            occurA = np.zeros(shape=(nA))
            occurB = np.zeros(shape=(nB))
            
            # Set occurence flags for A and B candidates
            for row in range(temp.shape[0]):
              if cnt == num:
                break
              else:
                i = int(temp[row,0])
                j = int(temp[row,1])
                score = temp[row,2];

                if occurA[i] == 0 and occurB[j] == 0: #&& score> (1+thre)
                  # Score of candidate a, score of candidate b and score of connecting a<->b
                  l = [candA[i,3], candB[j,3], score]
                  if connection[k] is None:
                    connection[k] = np.asarray(l).reshape((1,3))
                  else:
                    connection[k] = np.concatenate((connection[k], np.asarray(l).reshape((1,3))),axis=0) 
                  cnt = cnt+1;
                  occurA[i] = 1;
                  occurB[j] = 1;
            
            # cluster all the joints candidates into subset based on the part connection
            temp = connection[k];

            if self.doVerbose:
                print('[INFO] The connection is')
                print(temp)
            
            # initialize first body part connection 15&16 
            if k==0:
              subset = np.ones(shape=(temp.shape[0],kpt_num))*-1.0 #last number in each row is the parts number of that person
              for i in range(temp.shape[0]):
                subset[i, self.limbSeq[0, [0,1]]] = temp[i,[0,1]]
                subset[i, subset.shape[1]-1] = 2
                # add the score of parts and the connection
                subset[i, subset.shape[1]-2] = np.sum(candidates[temp[i,[0,1]].astype(int),2]) + temp[i,2]

            
            elif k==17 or k==18:# Why these limbs need specific treatment? These are the connections between ears and shoulders
              # add 15 16 connection
              partA = temp[:,0]
              partB = temp[:,1]
              indexA = self.limbSeq[k,0]
              indexB = self.limbSeq[k,1]

              for i in range(temp.shape[0]):# = 1:size(temp,1)
                for j in range(subset.shape[0]): # = 1:size(subset,1)
      #            print('[INFO] val of subset in A %f val of part %f val of subset in B %f' %(subset[j, indexA], partA[i], subset[j, indexB]))
                  if subset[j, indexA] == partA[i] and subset[j, indexB] == -1:
                    subset[j, indexB] = partB[i]
      #              print('[INFO] Entering to the first')
                  elif subset[j, indexB] == partB[i] and subset[j, indexA] == -1:
                    subset[j, indexA] = partA[i]
      #              print('[INFO] Entering to the second')

              continue
            else:
                
              # partA is already in the subset, find its connection partB
              partA = temp[:,0]
              partB = temp[:,1]
#              print(partA)
#              print(partB)

              indexA = self.limbSeq[k,0]
              indexB = self.limbSeq[k,1]

              for i in range(temp.shape[0]):# = 1:size(temp,1)
                num = 0
                for j in range(subset.shape[0]): # = 1:size(subset,1)
                  if subset[j, indexA] == partA[i]:
                    subset[j, indexB] = partB[i]
                    num = num+1
                    subset[j, subset.shape[1]-1] = subset[j, subset.shape[1]-1]+1
                    subset[j, subset.shape[1]-2] = subset[j, subset.shape[1]-2]+ candidates[int(partB[i]),2] + temp[i,2]

                # if find no partA in the subset, create a new subset
                if num==0:
#                  print('[INFO] Creating new subset for part', self.partList[indexA] )
                  if subset.shape[0] > 0:
                    subset = np.concatenate((subset, np.ones(shape=(1,kpt_num))*-1.0), axis=0)
                  else:
                    subset = np.ones(shape=(1,kpt_num))*-1.0

                  s = subset.shape
                  subset[s[0]-1, indexA] = partA[i]
                  subset[s[0]-1, indexB] = partB[i]
                  subset[s[0]-1, s[1]-1] = 2
                  subset[s[0]-1, s[1]-2] = np.sum(candidates[temp[i,[0,1]].astype(int),2]) + temp[i,2]

    #    if indexA == 2 and indexB == 16:
    #      print('[INFO] The subset before')
    #      print(subset)


        if self.doVerbose:
            print('[INFO] The subset is')
            print(subset)

        if subset is not None: 
          deleIdx = []
          s = subset.shape
          for i in range(s[0]): #=1:size(subset,1)
            # if(subset(i,end)<5)
            if subset[i,s[1]-1] < self.min_n_parts or (subset[i,s[1]-2]/subset[i,s[1]-1]) < 0.2:
              deleIdx.append(i)

          subset = np.delete(subset, deleIdx, axis=0) 
    #    if indexA == 2 and indexB == 16:
    #      print('[INFO] The subset after')
    #      print(subset)


        if self.doVerbose:
            print('[INFO] The subset is ')
            print(subset)

        return candidates, subset

    def visualize_connection(self, img, candidates, subset):
        if candidates is None or subset is None:
            return img

        facealpha = 0.6;
        stickwidth = 4;
        canvas = img.copy()

        for num in range(subset.shape[0]): #= 1:size(subset,1)
            for i in range(self.numParts):
                index = int(subset[num,i])
                if index == -1:
                  continue

                X = int(candidates[index,0])
                Y = int(candidates[index,1])
                cv2.circle(canvas, (X, Y), 5, self.colors[i], -1)
                # image = insertShape(image, 'FilledCircle', [X Y 5], 'Color', joint_color(i,:)); 


        for i in range(self.numLimbs):
            for num in range(subset.shape[0]):
                index = subset[num, self.limbSeq[i,[0,1]]]

                ## Check if any of the joint's limb was not detected
                if (index==-1).sum() > 0:
                    continue

                cur_canvas = canvas.copy()

                Y = candidates[index.astype(int), 0]
                X = candidates[index.astype(int), 1]

                if np.isnan(X).sum() == 0:
                    mx = np.mean(X)
                    my = np.mean(Y)

                    length = ((X[0]-X[1])** 2 + (Y[0]-Y[1])** 2)** 0.5
                    angle = math.degrees(math.atan2(X[0]-X[1], Y[0]-Y[1]))
                    polygon = cv2.ellipse2Poly((int(my),int(mx)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
      
        return canvas

    @abstractmethod
    def extract_keypoints(self, candidates, subset):
        pass

    def keypoints_to_coco_order(self, pred, imgId, catId):
        cocoList = []

        for i in range(len(pred)):
            d = {'image_id':imgId, 'category_id':catId, 'score':0.99, 'keypoints':np.zeros(shape=(17,3))}
            points = pred[i]['points']

            for j in range(len(points)):
                p = points[j]
                d['keypoints'][p[3],0] = p[0] - 0.5
                d['keypoints'][p[3],1] = p[1] - 0.5
                d['keypoints'][p[3],2] = 1.0

            d['score'] = pred[i]['score']*len(points)
            d['keypoints'] = d['keypoints'].reshape((51)).tolist()
            cocoList.append(d)

        return cocoList





class RGBPoseConstructor(PoseConstructor):
    def __init__(self, peaksTau, connectionTau):
        PoseConstructor.__init__(self, peaksTau, connectionTau)
        self.verbose(self.__class__.__name__)

    def init_limbs_map(self):
        # Matlab based indexed
        self.mapIdx = [31, 32, 39, 40, 33, 34, 35, 36, 41, 42,\
                       43, 44, 19, 20, 21, 22, 23, 24, 25, 26,\
                       27, 28, 29, 30, 47, 48, 49, 50, 53, 54,\
                       51, 52, 55, 56, 37, 38, 45, 46]

        self.limbSeq = [2, 3, 2, 6, 3, 4, 4, 5, 6, 7, 7, 8, 2,\
                        9, 9, 10, 10, 11, 2, 12, 12, 13, 13, 14,\
                        2, 1, 1, 15, 15, 17, 1, 16, 16, 18, 3, 17, 6, 18]


        # Matlab index
        self.orderCOCO = [1,0, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4]

        self.partList = ["Nose", "Neck", "RShoulder", "RElbow",\
                        "RWrist", "LShoulder", "LElbow",  "LWrist",\
                        "RHip", "RKnee", "RAnkle", "LHip",\
                        "LKnee" , "LAnkle", "REye",\
                        "LEye", "REar", "LEar"] 


        # Convert into array type and to zero based index
        self.mapIdx  = np.asarray(self.mapIdx).reshape((len(self.mapIdx)/2, 2)) - 1
        self.limbSeq = np.asarray(self.limbSeq).reshape((len(self.limbSeq)/2,2)) - 1
        self.orderCOCO = np.asarray(self.orderCOCO).reshape((len(self.orderCOCO))) - 1




    def extract_keypoints_coco(self, candidates, subset):
        if candidates is None or subset is None:
            return []

        point_cnt = 0
        pred = [None]*subset.shape[0]

        for ridxPred in range(subset.shape[0]): #= 1:size(subset,1)
            points = []
            part_cnt = 0
            for part in range(18):#= 1:18
                # Part 1 of CPM contains neck which is not in COCO
                if part == 1:
                   continue

                index = int(subset[ridxPred, part])
                if index >= 0:
                    part_cnt = part_cnt + 1
                    x = candidates[index,0]
                    y = candidates[index,1]
                    score = candidates[index,2]
                    _id = self.orderCOCO[part]
                    l = [x, y, score, _id]
                    points.append(l)


            point_cnt = point_cnt +1;
            pred[ridxPred] = {'points':points, 'score': subset[ridxPred, subset.shape[1]-2]}

        return pred





    def extract_keypoints(self, candidates, subset):
        if candidates is None or subset is None:
            return []

        point_cnt = 0
        pred = [None]*subset.shape[0]

        for ridxPred in range(subset.shape[0]): #= 1:size(subset,1)
            points = []
            part_cnt = 0
            for part in range(18):#= 1:18
                index = int(subset[ridxPred, part])
                if index >= 0:
                    part_cnt = part_cnt + 1
                    x = candidates[index,0]
                    y = candidates[index,1]
                    score = candidates[index,2]
                    l = [x, y, score, part]
                    points.append(l)


            point_cnt = point_cnt +1;
            pred[ridxPred] = {'points':points, 'score': subset[ridxPred, subset.shape[1]-2]}

        return pred





class DepthPoseConstructor(PoseConstructor):
    def __init__(self, peaksTau, connectionTau):
        PoseConstructor.__init__(self, peaksTau, connectionTau)

        self.verbose(self.__class__.__name__)

    def init_limbs_map(self):
        depthCpmUtils = CpmUtils.CpmDepthUtils()
        self.partList = depthCpmUtils.partList
        self.limbSeq  = depthCpmUtils.limbList

        print('[INFO] The list of parts is')
        print(self.partList)
        print(len(self.limbSeq))

        jointMap = {}
        i = 0

        nParts = len(self.partList)

        self.mapIdx = [x for x in range(nParts, nParts+len(self.limbSeq)*2)]
        self.mapIdx = np.asarray(self.mapIdx).reshape((len(self.mapIdx)/2,2))
        self.limbSeq = np.asarray(self.limbSeq)
        # print(self.mapIdx.shape)





    def extract_keypoints(self, candidates, subset):
        if candidates is None or subset is None:
            return []

        point_cnt = 0
        pred = [None]*subset.shape[0]

        for ridxPred in range(subset.shape[0]): 
            points = []
            part_cnt = 0

            for part in range(len(self.partList)):
                index = int(subset[ridxPred, part])

                if index >= 0:
                    part_cnt = part_cnt + 1
                    x = candidates[index,0]
                    y = candidates[index,1]
                    score = candidates[index,2]
                    _id = self.partList[part]
                    l = [x, y, score, _id]
                    points.append(l)

            point_cnt = point_cnt +1;
            pred[ridxPred] = {'points':points,\
                              'score': subset[ridxPred, subset.shape[1]-2]}

        return pred



class ItopPoseConstructor(PoseConstructor):
    def __init__(self, peaksTau, connectionTau):
        PoseConstructor.__init__(self, peaksTau, connectionTau)

        self.verbose(self.__class__.__name__)

    def init_limbs_map(self):
        depthCpmUtils = CpmUtils.SkeletonUtils()
        self.partList = depthCpmUtils.partList
        self.limbSeq  = depthCpmUtils.limbList

        print('[INFO] The list of parts is')
        print(self.partList)
        print(len(self.limbSeq))

        jointMap = {}
        i = 0

        nParts = len(self.partList)

        self.mapIdx = [x for x in range(nParts, nParts+len(self.limbSeq)*2)]
        self.mapIdx = np.asarray(self.mapIdx).reshape((len(self.mapIdx)//2,2))
        self.limbSeq = np.asarray(self.limbSeq)
        # print(self.mapIdx.shape)





    def extract_keypoints(self, candidates, subset):
        if candidates is None or subset is None:
            return []

        point_cnt = 0
        pred = [None]*subset.shape[0]

        for ridxPred in range(subset.shape[0]): 
            points = []
            part_cnt = 0

            for part in range(len(self.partList)):
                index = int(subset[ridxPred, part])

                if index >= 0:
                    part_cnt = part_cnt + 1
                    x = candidates[index,0]
                    y = candidates[index,1]
                    score = candidates[index,2]
                    _id = self.partList[part]
                    l = [x, y, score, _id]
                    points.append(l)

            point_cnt = point_cnt +1;
            pred[ridxPred] = {'points':points,\
                              'score': subset[ridxPred, subset.shape[1]-2]}

        return pred


def extract_keypoints_(candidates, subset):
    if candidates is None or subset is None or len(subset.shape)==0:
        return []

    depthCpmUtils = CpmUtils.CpmDepthUtils()
    partList = depthCpmUtils.partList

    point_cnt = 0
    pred = [None]*subset.shape[0]

    for ridxPred in range(subset.shape[0]): 
        points = []
        part_cnt = 0

        for part in range(len(partList)):
            index = int(subset[ridxPred, part])

            if index >= 0:
                part_cnt = part_cnt + 1
                x = candidates[index,0]
                y = candidates[index,1]
                score = candidates[index,2]
                _id = partList[part]
                l = [x, y, score, _id]
                points.append(l)

        point_cnt = point_cnt +1;
        pred[ridxPred] = {'points':points,\
                          'score': subset[ridxPred, subset.shape[1]-2]}

    return pred





def extract_depth_keypoints(candidates, subset, imgId):
    utils      = CpmUtils.CpmDepthUtils()
    jointMap   = utils.jointMap
    numParts=17 # without the background

    prediction = extract_keypoints_(candidates, subset)
    totalScore = 0.0
    resultList=[]
    for pred in prediction:
        detection = { 'image_id' : imgId,\
                      'keypoints' : [[0.0,0.0,0.0,0.0]]*(numParts),\
                      'score':pred['score']}

        detScore = 0.0
        for point in pred['points']:
            x, y, partId, score = float(point[0]), float(point[1]), point[3], float(point[2])
            partIdx = jointMap[partId]
            detection['keypoints'][partIdx] = ((x, y, 1.0, score))
            totalScore += score
            detScore += score

        detection['score'] = detScore
        resultList.append(detection)

    return resultList, totalScore



