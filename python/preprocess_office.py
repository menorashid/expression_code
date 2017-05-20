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
import scripts_and_viz

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
    out_folder='../data/office/domain_adaptation_images_256'
    # out_folder='../data/office/domain_adaptation_images_227'
    makeResizeFiles(in_folder,out_folder,out_size=(256,256));

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
    # in_folder='../data/office/domain_adaptation_images_227'
    # dir_files='../data/office/domain_adaptation_images_227/train_test_files'
    in_folder='../data/office/domain_adaptation_images_256'
    dir_files='../data/office/domain_adaptation_images_256/train_test_files'
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
            weights_file_curr=os.path.join(dir_files,'weights_'+domain+'_'+str(fold_num)+'.npy');
            print train_file_curr,test_file_curr,len(train_all[fold_num]),len(test_all[fold_num]);
            random.shuffle(train_all[fold_num]);
            random.shuffle(test_all[fold_num]);
            util.writeFile(train_file_curr,train_all[fold_num]);
            util.writeFile(test_file_curr,test_all[fold_num]);
            preprocess_data.saveWeightsFile(train_file_curr,weights_file_curr);


def saveFullTestFiles():
    dir_files='../data/office/domain_adaptation_images_256/train_test_files'
    domains=['amazon','dslr','webcam'];
    for domain in domains:
        files=[os.path.join(dir_files,'_'.join([file_pre,domain,str(0)+'.txt'])) for file_pre in ['train','test']];
        out_file=os.path.join(dir_files,'test_'+domain+'_all.txt');
        lines=[util.readLinesFromFile(file_curr) for file_curr in files];
        lines=lines[0]+lines[1];
        print len(lines);
        lines=list(set(lines));
        print len(lines);
        util.writeFile(out_file,lines);

def writeScriptForBaseline():
    domain_train='amazon';
    domain_test='webcam';
    num_folds=1;
    path_to_th='train_alexnet_withBlur.th';
    experiment_name='office_bl_'+domain_train+'_less_256';
    out_dir_meta=os.path.join('../experiments',experiment_name);
    out_script=os.path.join('../scripts','train_'+experiment_name);
    num_scripts=1;
    dir_files='../data/office/domain_adaptation_images_256/train_test_files'
    util.mkdir(out_dir_meta);
    model_file='../models/alexnet_31.dat';
    learningRate=0.001;
    testAfter=1;
    saveAfter=5;
    iterations=50;
    lower=True;
    commands=[];
    for num_fold in range(num_folds):
        fold_num=domain_train+'_'+str(num_fold);
        # val_data_path=os.path.join(dir_files,'test_'+domain_test+'_all.txt');
        # modelTest=os.path.join(out_dir_meta,fold_num,'intermediate','model_all_40.dat');
        command=scripts_and_viz.writeBlurScript(\
                path_to_th,
                out_dir_meta,
                dir_files,
                fold_num,
                model_file=model_file,
                learningRate=learningRate,
                # val_data_path=val_data_path,
                mean_im_path='',
                std_im_path='',
                iterations=iterations,
                testAfter=testAfter,
                saveAfter=saveAfter,
                weights=True,
                lower=lower
                # ,
                # modelTest=modelTest
                );
        commands.append(command);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_script+'_'+str(idx_commands)+'.sh';
        print out_file_script_curr,len(commands);
        util.writeFile(out_file_script_curr,commands);


def main():

    writeScriptForBaseline()
    # saveFullTestFiles()

    # saveTrainTestFiles()

    

if __name__=='__main__':
    main();