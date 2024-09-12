import numpy as np
import torch as torch
import shutil,os
import copy
import random
import nibabel as nib
import torch.nn as nn
import math
from numpy import linalg as la
from torchvision import transforms
from scipy.optimize import nnls,curve_fit,leastsq
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
from sklearn.preprocessing import StandardScaler

def make_ball_coord(array):
    new_array = np.zeros([array.shape[0],array.shape[1]])
    for i in range(array.shape[0]):
        
        for k in range(array.shape[1]):
            if k == 0:
                r = (array[i,:].dot(array[i,:]))**0.5
                new_array[i,k] = r
            else:
                y = array[i,k]
                x = (array[i,:k].dot(array[i,:k]))**0.5
                a = np.arctan(y/x)
                if k == 1:
                    x = array[i,k-1]
                    if a > 0:
                        if x < 0:
                            a = a + np.pi/2
                    if a < 0:
                        if x < 0:
                            a = a + 3*np.pi/2
                        if x >= 0:
                            a = a + 2*np.pi
                    if a == 0:
                        if x < 0:
                            a = a + np.pi 
                    new_array[i,k] = a
                else:
                    new_array[i,k] = a
    return new_array    


def traversalDir_FirstDir(path):
    list1 = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            list1.append(m)
    return(list1)

def make_array1(ROIpath,imagepath,newpath):
    ROI = nib.load(ROIpath)
    ROI_data = ROI.get_fdata()
    ROI_affine = ROI.affine.copy()
    ROI_hdr = ROI.header.copy()
    x,y,z=np.nonzero(ROI_data)
    image =['Enhancement at first postcontrast time point.nii.gz',
            'maximum_enhancement.nii.gz',
            'curve_shape_index.nii.gz',
            'time_to_peak.nii.gz',
            'uptake_rate.nii.gz',
            'washout_rate.nii.gz',
            'SER.nii.gz']
    #image = ['f','dslow','dfast','md','mk','ddc','a']      #conventional images
    new_array = np.zeros([len(x),len(image)])
    for i in range(len(image)):
        data_image = nib.load(imagepath+"\\"+image[i])
        data_affine = data_image.affine.copy()
        data_hdr = data_image.header.copy()
        data = data_image.get_fdata()
        for j in range(len(x)):
            new_array[j,i]=data[x[j],y[j],z[j]]
        a_list = new_array[:,i]
        mean1 = np.mean(a_list)
        std1 = np.std(a_list)
        for k in range(len(x)):
            if new_array[k,i] < (mean1-3*std1):
                new_array[k,i] = mean1-3*std1
            if new_array[k,i] > (mean1+3*std1):
                new_array[k,i] = mean1+3*std1
            data[x[k],y[k],z[k]] = new_array[k,i]
        nib.Nifti1Image(data,data_affine,data_hdr).to_filename(newpath+'\\'+ image[i])
        
    max1=np.max(new_array,axis=0)
    min1=np.min(new_array,axis=0)
    for x0 in range(new_array.shape[1]):
        for y0 in range(new_array.shape[0]):
            new_array[y0,x0] = (new_array[y0,x0]-min1[x0])/(max1[x0]-min1[x0]+1e-10)
    
    return x,y,z,ROI_data,ROI_affine,ROI_hdr,new_array

def make_array2(ROIpath,imagepath,newpath):
    ROI = nib.load(ROIpath)
    ROI_data = ROI.get_fdata()
    ROI_affine = ROI.affine.copy()
    ROI_hdr = ROI.header.copy()
    x,y,z=np.nonzero(ROI_data)
    image =['Enhancement at first postcontrast time point.nii.gz',
            'maximum_enhancement.nii.gz',
            'curve_shape_index.nii.gz',
            'time_to_peak.nii.gz',
            'uptake_rate.nii.gz',
            'washout_rate.nii.gz',
            'SER.nii.gz']
    #image = ['f','dslow','dfast','md','mk','ddc','a']
    new_array = np.zeros([len(x),len(image)])
    for i in range(len(image)):
        data_image = nib.load(imagepath+"\\"+image[i])
        data_affine = data_image.affine.copy()
        data_hdr = data_image.header.copy()
        data = data_image.get_fdata()
        for j in range(len(x)):
            new_array[j,i]=data[x[j],y[j],z[j]]
        a_list = new_array[:,i]
        mean1 = np.mean(a_list)
        std1 = np.std(a_list)
        for k in range(len(x)):
            if new_array[k,i] < (mean1-3*std1):
                new_array[k,i] = mean1-3*std1
            if new_array[k,i] > (mean1+3*std1):
                new_array[k,i] = mean1+3*std1
            data[x[k],y[k],z[k]] = new_array[k,i]
    
        nib.Nifti1Image(data,data_affine,data_hdr).to_filename(newpath+'\\'+ 'circle_'+image[i])
        
        
    max1=np.max(new_array,axis=0)
    min1=np.min(new_array,axis=0)
    for x0 in range(new_array.shape[1]):
        for y0 in range(new_array.shape[0]):
            new_array[y0,x0] = (new_array[y0,x0]-min1[x0])/(max1[x0]-min1[x0]+1e-10)
    
    return x,y,z,ROI_data,ROI_affine,ROI_hdr,new_array


def make_imgs1_2(path0,goalpath,roi1,roi2):
    patientlist = traversalDir_FirstDir(path0)
    for patient in patientlist:
        patient_path = patient
        print(patient_path)
        finalpath = goalpath+'\\'+patient.split('\\')[-1]
        if not os.path.exists(finalpath):
            os.makedirs(finalpath)
           
        imgs_name=['r.nii.gz','a1.nii.gz','a2.nii.gz','a3.nii.gz','a4.nii.gz','a5.nii.gz']
        
        
        ROI1_list = os.path.join(roi1,patient.split('\\')[-1])
        ROI1_file = traversalDir_FirstDir(ROI1_list)
        for ROI1_data in ROI1_file:
            print(ROI1_data)
            x1,y1,z1,ROI_data1,ROI_affine1,ROI_hdr1,new_array1 = make_array1(ROI1_data,patient_path,finalpath)
            newimgs1 = make_ball_coord(StandardScaler().fit_transform(new_array1))
            for i in range(len(newimgs1)):
                for j in range(len(x1)):
                    ROI_data1[x1[j],y1[j],z1[j]] = newimgs1[i][j]
                nib.Nifti1Image(ROI_data1,ROI_affine1,ROI_hdr1).to_filename(finalpath+'\\'+ imgs_name[i])
            
        ROI2_list = os.path.join(roi2,patient.split('\\')[-1])
        ROI2_file = traversalDir_FirstDir(ROI2_list)
        for ROI2_data in ROI2_file:
            x2,y2,z2,ROI_data2,ROI_affine2,ROI_hdr2,new_array2 = make_array2(ROI2_data,patient_path,finalpath)
            newimgs2 = make_ball_coord(StandardScaler().fit_transform(new_array2))
            for i in range(len(newimgs2)):
                for j in range(len(x2)):
                    ROI_data2[x2[j],y2[j],z2[j]] = newimgs2[i][j]
                nib.Nifti1Image(ROI_data2,ROI_affine2,ROI_hdr2).to_filename(finalpath+'\\'+ 'circle_'+imgs_name[i])
                
path0 = r'C:\...\image'     #Conventional images path
goalpath = r'C:\...\result'    #Vector images path
roi1_path = r'C:\...\roi'      #Intratumoral Region
roi2_path = r'C:\...\roi-circle'   #Peritumoral Region
make_imgs1_2(path0,goalpath,roi1_path,roi2_path)
        
        
           

    
    