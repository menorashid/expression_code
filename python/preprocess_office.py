import numpy as np;
import cv2;
import util;
import os;
import visualize;
import scipy;
import scipy.io;
import multiprocessing;
import subprocess;
import sys;
import random;
import re;
import csv;
dir_server='/home/SSD3/maheen-data/';
click_str='http://vision1.idav.ucdavis.edu:1000/';
import urllib
import time;
import preprocess_data;

def resizeAndSave((in_file,out_file,out_size,num_arg)):
    if num_arg%1000==0:
        print num_arg;
    out_dir=os.path.split(out_file)[0];
    util.makedirs(out_dir);
    im=scipy.misc.imread(in_file);
    im=scipy.misc.imresize(im,out_size);
    scipy.misc.imsave(out_file,im);

def makeResizeFiles(in_folder,out_folder,ext='.jpg',out_size=(227,227)):
    util.mkdir(out_folder);
    in_files_all=[os.path.join(root,file_curr) for root,dirs,files in os.walk(in_folder) for file_curr in files if file_curr.endswith(ext)]
    out_files_all=[file_curr.replace(in_folder,out_folder) for file_curr in in_files_all];
    out_size_all=[out_size]*len(out_files_all)
    args=zip(in_files_all,out_files_all,out_size_all,list(range(len(out_files_all))));
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    p.map(resizeAndSave,args);

def script_resize():
    in_folder='../data/office/domain_adaptation_images'
    out_folder='../data/office/domain_adaptation_images_227'
    makeResizeFiles(in_folder,out_folder);

def makeFolds(folder,num_per_cat,num_folds,ext='.jpg'):
    files_all=[os.path.join(root,file_curr) for root,dirs,files in os.walk(folder) for file_curr in files if file_curr.endswith(ext)]
    files_all=np.array(files_all);
    categories=np.array([file_curr.split('/')[-2] for file_curr in files_all]);
    categories_uni=np.unique(categories);
    
    train_all=[[] for num in range(num_folds)];
    test_all=[[] for num in range(num_folds)];

    for cat_num,cat_curr in enumerate(categories_uni):
        files_rel=files_all[categories==cat_curr];
        for fold_num in range(num_folds):
            np.random.shuffle(files_rel);
            start_idx=0;
            end_idx=min(start_idx+num_per_cat,files_rel.shape[0]);
            idx_keep=np.zeros((files_rel.shape[0],))
            idx_keep[start_idx:end_idx]=1;
            train_files=list(files_rel[idx_keep>0]);
            test_files=list(files_rel[idx_keep<=0]);
            train_files=[file_curr+' '+str(cat_num) for file_curr in train_files]
            test_files=[file_curr+' '+str(cat_num) for file_curr in test_files]
            train_all[fold_num]=train_all[fold_num]+train_files;
            test_all[fold_num]=test_all[fold_num]+test_files;
        
    return train_all,test_all;

def saveTrainTestFiles():
    in_folder='../data/office/domain_adaptation_images_227'
    dir_files='../data/office/domain_adaptation_images_227/train_test_files'
    util.mkdir(dir_files);
    domains=['amazon','dslr','webcam'];
    num_per_cats=[20,8,8];
    num_folds=5;
    for domain,num_per_cat in zip(domains,num_per_cats):
        folder=os.path.join(in_folder,domain);
        train_all,test_all=makeFolds(folder,num_per_cat,num_folds,ext='.jpg')
        for fold_num in range(len(train_all)):
            train_file_curr=os.path.join(dir_files,'train_'+domain+'_'+str(fold_num)+'.txt');
            test_file_curr=os.path.join(dir_files,'test_'+domain+'_'+str(fold_num)+'.txt');
            print train_file_curr,test_file_curr,len(train_all[fold_num]),len(test_all[fold_num]);
            random.shuffle(train_all[fold_num]);
            random.shuffle(test_all[fold_num]);
            util.writeFile(train_file_curr,train_all[fold_num]);
            util.writeFile(test_file_curr,test_all[fold_num]);


def main():
    saveTrainTestFiles()
    

if __name__=='__main__':
    main();