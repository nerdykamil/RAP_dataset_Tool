#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 16:55:51 2018

@author: lfd
"""

##this script extract relevant data from .mat file in RAP dataset
import scipy.io  
import numpy as np
#import csv
#import datetime
#from datetime import datetime
from datetime import timedelta
#import os
import pandas as pd


def loadmat_and_extract(file, root_dir):
    ##load the .mat file
    #mat = scipy.io.loadmat('./RAP_annotation/RAP_annotation.mat') #we have desired objects in mat now
    mat = scipy.io.loadmat(file)
    ##there are key value pairs in mat of which we want wiki key and its values
    print(mat.keys())
    data = mat['RAP_annotation']
    images = data['imagesname']
    labels = data['label']
    eng_attr = data['attribute_eng']
    pos = data['position']
   
    ## Extracting required labels only
    # 0 -> Gender Pr1
    # 1-3 -> Age
    # 15-23 -> Upper Body
    # 24-29 -> Lower Body
    # 35-42 -> attachments/accessories
    # 51-54 -> face direction
    # 55-62 -> occlusion
    # 63-74 -> upper color
    # 75-82 -> lower color
    
    ## putting wordy attributes in place of 1's in labels
    req_labels = labels[0][0].astype(str)
    for imgnum in range(0,len(req_labels)):
        for lblnum in range(0,len(req_labels[imgnum])):
            if req_labels[imgnum][lblnum] == '1':
                req_labels[imgnum][lblnum] = eng_attr[0][0][lblnum][0][0]
    
    # for now taking gender, upper body, lower body, face direction, upper color and lower colr
    #req_labels2 = np.ndarray((41585,1))
    #from set import Set
    req_labels2 = []
    lbl_idx = [0] + list(range(15,23+1)) + list(range(24,29+1)) + list(range(51,54+1)) + list(range(63,74+1)) + list(range(75,82+1)) 
    for imgnum in range(0,len(req_labels)):
        temp_lbl = []
        for i in range(0,92):
            if i == 0 and req_labels[imgnum][i] == '0':
                temp_lbl.append("Male")
            elif i == 0 and req_labels[imgnum][i] == '2':
                temp_lbl.append("Unknown")
            elif i in lbl_idx:
                temp_lbl.append(req_labels[imgnum][i])
            
        req_labels2.append(np.asarray(temp_lbl).reshape(-1,1)) 
    
#    req_labels2 = np.asarray(req_labels2)
    
    img_names = []
    for i in range(0,len(images[0][0])):
        renamed = str(images[0][0][i][0][0][:-4]).replace('-','_')
        img_names.append(renamed)
    #img_names[0][:-4]
    
    ##finding size of images
    import cv2
#    root_dir = "./RAP_dataset/"
    print("extracting images from root dir %s to get image sizes" % root_dir)
    
    width = []
    height = []
    for l in range(0,len(img_names)):
    #    print(img_names[l])
        file_loc = root_dir + str(img_names[l] + ".png")
    #    print(file_loc)
        img = cv2.imread(file_loc,0)
        height.append(img.shape[0])
        width.append(img.shape[1])
    
    ## Finding top right, topleft, bottomright, bottomleft
    ## fb = fullbody, hs = head-shoulder, ub = upperbody, lb = lowerbody
         
    bbox = list(pos[0][0])
    fb_xmin = []
    fb_ymin = []
    fb_xmax = []
    fb_ymax = []
    hs_xmin = []
    hs_ymin = []
    hs_xmax = []
    hs_ymax = []
    ub_xmin = []
    ub_ymin = []
    ub_xmax = []
    ub_ymax = []
    lb_xmin = []
    lb_ymin = []
    lb_xmax = []
    lb_ymax = []
    for i in range(0,len(bbox)):
        fb_xmin.append(bbox[i][0])
        fb_ymin.append(bbox[i][1])
        fb_xmax.append(bbox[i][2] + bbox[i][0])
        fb_ymax.append(bbox[i][3] + bbox[i][1])
        hs_xmin.append(bbox[i][4])
        hs_ymin.append(bbox[i][5])
        hs_xmax.append(bbox[i][6] + bbox[i][4])
        hs_ymax.append(bbox[i][7] + bbox[i][5])
        ub_xmin.append(bbox[i][8])
        ub_ymin.append(bbox[i][9])
        ub_xmax.append(bbox[i][10] + bbox[i][8])
        ub_ymax.append(bbox[i][11] + bbox[i][9])
        lb_xmin.append(bbox[i][12])
        lb_ymin.append(bbox[i][13])
        lb_xmax.append(bbox[i][14] + bbox[i][12])
        lb_ymax.append(bbox[i][15] + bbox[i][13])
    
    ## Saving attribute list
    attr=[]
    for i in lbl_idx:
        attr.append(eng_attr[0][0][i][0][0])


    data3 = {'labels':attr}
    df3 = pd.DataFrame(data=data3,index=lbl_idx)
    df3.to_csv("attributes.csv")


    ## Putting all data in dataframe
    data2 = {'images':img_names,'labels':req_labels2,'width':width,'height':height,
            'fb_xmin':fb_xmin,'fb_xmax':fb_xmax,'fb_ymin':fb_ymin,'fb_ymax':fb_ymax,
            'ub_xmin':ub_xmin,'ub_xmax':ub_xmax,'ub_ymin':ub_ymin,'ub_ymax':ub_ymax,
            'hs_xmin':hs_xmin,'hs_xmax':hs_xmax,'hs_ymin':hs_ymin,'hs_ymax':hs_ymax,
            'lb_xmin':lb_xmin,'lb_xmax':lb_xmax,'lb_ymin':lb_ymin,'lb_ymax':lb_ymax}
    df = pd.DataFrame(data=data2)
    return df
    


def annotate(df):
    #df = pd.read_csv(csvfile)
    for row in df.itertuples():
        xmlData = open("annotations/"+str(row.images)+".xml", 'w')
        xmlData.write('<?xml version="1.0"?>' + "\n")
        xmlData.write('<annotation>' + "\n")
        xmlData.write('    ' + '<folder>RAP_dataset/</folder>' + "\n")
        xmlData.write('    ' + '<filename>' \
                          + str(str(row.images) + '.png') + '</filename>' + "\n")
        xmlData.write('    ' + '<size>' + "\n")
        xmlData.write('        ' + '<width>' \
                          + str(row.width) + '</width>' + "\n")
        xmlData.write('        ' + '<height>' \
                          + str(row.height) + '</height>' + "\n")
        xmlData.write('        ' + '<depth>3</depth>' + "\n")
        xmlData.write('    ' + '</size>' + "\n")
        
        for i in range(0,len(row.labels)):
#            if row.labels[i] != "0" or row.labels[i] == "['2']":
#            if row.labels[i] != "0" or row.labels[i] != "2":
            ext_lbl = str(row.labels[i]).replace("[","").replace("]","").replace("'","")
            if ext_lbl != "0" or ext_lbl == "2":                
                xmlData.write('    ' + '<object>' + "\n")
                xmlData.write('        ' + '<name>' \
                              + str(ext_lbl) + '</name>' + "\n")
                xmlData.write('        ' + '<pose>Unknown</pose>' + "\n")
                xmlData.write('        ' + '<truncated>0</truncated>' + "\n")
                xmlData.write('        ' + '<difficult>0</difficult>' + "\n")
                if row.labels[i][0][:2] == 'Ma' or row.labels[i][0][:2] == 'Fe':
                    xmlData.write('        ' + '<bndbox>' + "\n")
                    xmlData.write('            ' + '<xmin>' \
                              + str(row.fb_xmin) + '</xmin>' + "\n")
                    xmlData.write('            ' + '<ymin>' \
                                  + str(row.fb_ymin) + '</ymin>' + "\n")            
                    xmlData.write('            ' + '<xmax>' \
                                  + str(row.fb_xmax) + '</xmax>' + "\n")
                    xmlData.write('            ' + '<ymax>' \
                                  + str(row.fb_ymax) + '</ymax>' + "\n")
                    xmlData.write('        ' + '</bndbox>' + "\n")
                if row.labels[i][0][:2] == 'up' or row.labels[i][0][:2] == 'ub':
                    xmlData.write('        ' + '<bndbox>' + "\n")
                    xmlData.write('            ' + '<xmin>' \
                              + str(row.ub_xmin) + '</xmin>' + "\n")
                    xmlData.write('            ' + '<ymin>' \
                                  + str(row.ub_ymin) + '</ymin>' + "\n")            
                    xmlData.write('            ' + '<xmax>' \
                                  + str(row.ub_xmax) + '</xmax>' + "\n")
                    xmlData.write('            ' + '<ymax>' \
                                  + str(row.ub_ymax) + '</ymax>' + "\n")
                
                    xmlData.write('        ' + '</bndbox>' + "\n")
                if row.labels[i][0][:3] == 'low' or row.labels[i][0][:2] == 'lb':
                    xmlData.write('        ' + '<bndbox>' + "\n")
                    xmlData.write('            ' + '<xmin>' \
                              + str(row.lb_xmin) + '</xmin>' + "\n")
                    xmlData.write('            ' + '<ymin>' \
                                  + str(row.lb_ymin) + '</ymin>' + "\n")            
                    xmlData.write('            ' + '<xmax>' \
                                  + str(row.lb_xmax) + '</xmax>' + "\n")
                    xmlData.write('            ' + '<ymax>' \
                                  + str(row.lb_ymax) + '</ymax>' + "\n")
                
                    xmlData.write('        ' + '</bndbox>' + "\n")
                if row.labels[i][0][:2] == 'fa':
                    xmlData.write('        ' + '<bndbox>' + "\n")
                    xmlData.write('            ' + '<xmin>' \
                              + str(row.hs_xmin) + '</xmin>' + "\n")
                    xmlData.write('            ' + '<ymin>' \
                                  + str(row.hs_ymin) + '</ymin>' + "\n")            
                    xmlData.write('            ' + '<xmax>' \
                                  + str(row.hs_xmax) + '</xmax>' + "\n")
                    xmlData.write('            ' + '<ymax>' \
                                  + str(row.hs_ymax) + '</ymax>' + "\n")
                
                    xmlData.write('        ' + '</bndbox>' + "\n")
                xmlData.write('    ' + '</object>' + "\n")
        xmlData.write('</annotation>' + "\n")
        xmlData.close()

file = './RAP_annotation/RAP_annotation.mat'    
root_dir = './RAP_dataset/'
RAP_anno = loadmat_and_extract(file, root_dir)
RAP_anno.to_csv('RAP_attributes.csv')
annotate(df)


