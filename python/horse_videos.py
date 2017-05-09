import util;
import ffmpy
import visualize;
import os;
import numpy as np
from PIL import Image
import scipy.misc;
import caffe
import re;
import subprocess;
import time;
import multiprocessing;
import math;
import cv2;
import csv
import random;
import scripts_and_viz
import preprocess_data
dir_server='/home/SSD3/maheen-data/';
click_str='http://vision1.idav.ucdavis.edu:1000/';

def testVOC(net,in_dir,out_dir,batchSize=15,small_side=500):

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe

    in_files_all=util.getFilesInFolder(in_dir,'.jpg')
    in_files_split=np.array(in_files_all);
    in_files_split=np.array_split(in_files_split,len(in_files_all)/batchSize+1);
    print 'num_batches',len(in_files_split)
    
    for num_batch,in_files in enumerate(in_files_split):
        print num_batch,len(in_files_split),len(in_files)

        num_im=len(in_files);
        out_files=[os.path.join(out_dir,os.path.split(file_curr)[1]) for file_curr in in_files]
        # in_=np.array(Image.open(in_files[0]),dtype=np.float32)
        # new_size=(small_side,int(in_.shape[1]*(float(small_side)/in_.shape[0])))
        # print new_size,in_.shape
        # in_=scipy.misc.imresize(in_,new_size);
        # in_.shape[1]
        new_size=(small_side,small_side);

        in_arr=np.zeros((num_im,3,small_side,small_side),dtype=np.float32)
        
        for num_file,in_file in enumerate(in_files):
            in_ = scipy.misc.imread(in_file);
            in_=scipy.misc.imresize(in_,new_size);
            in_ = np.array(in_, dtype=np.float32)
            
            in_ = in_[:,:,::-1]
            in_ -= np.array((104.00698793,116.66876762,122.67891434))
            in_ = in_.transpose((2,0,1))
            in_arr[num_file]=in_;

        net.blobs['data'].reshape(*in_arr.shape)
        net.blobs['data'].data[...]=in_arr
        
        net.forward()
        out_arr = net.blobs['score'].data

        for out_num in range(out_arr.shape[0]):
            out=out_arr[out_num].argmax(axis=0)
            out[out==13]=255;
            out[out<255]=0;
            out_file=out_files[out_num];
            scipy.misc.imsave(out_file,out)

    visualize.writeHTMLForFolder(out_dir);
    

def getNet(deploy_path,model_path,gpu_num=0):
    caffe.set_mode_gpu()
    caffe.set_device(gpu_num)
    net = caffe.Net(deploy_path,model_path, caffe.TEST)
    return net;
    

def saveFrames((in_file,out_dir_curr,num_secs)):
    print out_dir_curr
    util.mkdir(out_dir_curr);
    vid_length=util.getVideoLength(in_file);
    # vid_length=int(vid_length);
    vid_points=np.arange(num_secs,vid_length,num_secs)

    for idx_vid_point,vid_point in enumerate(vid_points):
        out_file=os.path.join(out_dir_curr,str(idx_vid_point)+'.jpg');
        command=[];
        command.extend(['ffmpeg','-y','-hide_banner','-loglevel','panic']);
        command.extend(['-accurate_seek', '-ss', str(vid_point)])
        command.extend(['-i',util.escapeString(in_file)]);
        command.extend(['-frames:v','1',out_file]);
        command=' '.join(command);
        os.system(command)
        # , shell=True)
        # -frames:v 1 period_down_$i.bmp

        # command.extend(['-i',in_file]);
        # command.extend(['-filter:v','fps=fps=1/'+str(num_secs)]);
        # command.append(os.path.join(out_dir_curr,'%0d.jpg'));
        # command=' '.join(command);
        # print command;
        
    # 
    visualize.writeHTMLForFolder(out_dir_curr);


def extractFramesMultiProc():
    in_dir='../../../Dropbox/horse_videos/Experimental pain';
    out_dir='../data/karina_vids/data_unprocessed';
    util.makedirs(out_dir);

    dirs_videos=[os.path.join(in_dir,dir_a,dir_b) for dir_a in ['Observer present','Observer not present'] for dir_b in ['Pain','No pain']]; 
    files_all=[];
    for dir_curr in dirs_videos:
        files_all=files_all+util.getFilesInFolder(dir_curr,'.mts');

    just_names=[os.path.split(file_curr)[1] for file_curr in files_all];
    print len(just_names),len(set(just_names));
    print just_names[0];
    num_secs=10;

    args=[];
    for in_file in files_all:
        just_name=os.path.split(in_file)[1];
        out_dir_curr=os.path.join(out_dir,just_name[1:just_name.rindex('.')]);
        args.append((in_file,out_dir_curr,num_secs));

    # args=args[33:];
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    t=time.time();
    p.map(saveFrames,args);
    print (time.time()-t);


def saveMasks():
    deploy_path='../../fcn.berkeleyvision.org/voc-fcn8s/deploy.prototxt'
    model_path='../../fcn.berkeleyvision.org/voc-fcn8s/fcn8s-heavy-pascal.caffemodel'
    net=getNet(deploy_path,model_path);
    
    in_dir_meta='../data/karina_vids/data_unprocessed';
    in_dirs=[os.path.join(in_dir_meta,dir_curr) for dir_curr in os.listdir(in_dir_meta) if os.path.isdir(os.path.join(in_dir_meta,dir_curr)) and not dir_curr.endswith('mask')]
    in_dirs=in_dirs[33:];
    # print in_dirs;
    print len(in_dirs);
    out_dirs=[dir_curr+'_mask' for dir_curr in in_dirs]
    print in_dirs[0]
    # print 
    for idx_dir,(in_dir,out_dir) in enumerate(zip(in_dirs,out_dirs)):
        print idx_dir,in_dir,out_dir
        util.mkdir(out_dir);
        testVOC(net,in_dir,out_dir)

def getImAnnoList(row_curr,out_dir,num_secs=10):
    # videos,lengths,
    vid_name=row_curr[0];
    # vids=[os.path.split(file_curr)[1] for file_curr in videos];
    # idx_vid=vids.index(vid_name)
    # vid_file=videos[idx_vid]
    # length=lengths[idx_vid];

    out_dir_curr=os.path.join(out_dir,vid_name.strip('#'));


    if row_curr[1]=='np':
        annos_all=util.getFilesInFolder(out_dir_curr,'.jpg');
        annos_all=[file_curr+' -1' for file_curr in annos_all];
    elif row_curr[1]=='p' and row_curr[2].lower()=='all':
        annos_all=util.getFilesInFolder(out_dir_curr,'.jpg');
        annos_all=[file_curr+' 1' for file_curr in annos_all];
    elif row_curr[1]=='p' and row_curr[2].lower()=='none':
        annos_all=[];
    else:
        # times=row_curr[3];
        annos_all=[];
        for time_range in row_curr[2].split(';'):
            times=[];
            for t in time_range.split('-'):
                t=t.split(':');
                t_sec=int(t[0])*60+int(t[1][:2]);
                times.append(t_sec);
            im_start=max(round(times[0]/float(num_secs))-1,0);
            im_end=round(times[1]/float(num_secs));
            print row_curr[2],times,im_start,im_end
            for im_num in range(int(im_start),int(im_end)):
                im_curr=os.path.join(out_dir_curr,str(im_num)+'.jpg')
                if os.path.exists(im_curr):
                    print im_curr;
                    # raw_input();
                    annos_all.append(im_curr+' 1');

    return annos_all;


def labelImages():
    out_dir='../data/karina_vids/data_unprocessed';
    out_file_anno=os.path.join(out_dir,'annos_all.txt');
    excel_file='Videos_overview_06072016.csv';
    
    annos_all=[];
    with open(excel_file,'rb') as f:
        spamreader = csv.reader(f);
        for row in spamreader:
            if row[1].startswith('#'):
                anno_curr=[];
                anno_curr.append(row[1]);
                if row[5]=='x':
                    anno_curr.append('np');
                elif row[6]=='x':
                    anno_curr.append('p');
                    anno_curr.append(row[-3]);
                else:
                    print row;
                    continue;

                annos_all.append(anno_curr);

    im_annos_all=[];
    print len(annos_all);
    rows_to_keep=[];
    for row_curr in annos_all:
        vid_name=row_curr[0].strip('#');
        if os.path.exists(os.path.join(out_dir,vid_name)):
            im_anno_list=getImAnnoList(row_curr,out_dir);
            im_annos_all.extend(im_anno_list);

    print len(im_annos_all);
    util.writeFile(out_file_anno,im_annos_all);
    
    # print len(rows_to_keep);
def saveResizeImage(real_im,mask_im,out_file,border=200,size_out=(96,96),justMask=False):

    im=scipy.misc.imread(real_im);
    mask_curr=scipy.misc.imread(mask_im);
    mask_curr=scipy.misc.imresize(mask_curr,(im.shape[0],im.shape[1]),'nearest');
    mask_curr[mask_curr>0]=1;
    im_out,contours=cv2.findContours(np.array(mask_curr),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours_sizes=[im_curr.shape[0] for im_curr in im_out];
    max_contour=im_out[np.argmax(contours_sizes)];
    contour_curr=max_contour
    contour_curr=contour_curr[:,0,:];
    min_max=list(np.min(contour_curr,axis=0))+list(np.max(contour_curr,axis=0));
    min_max=[max(min_max[0]-border,0),max(min_max[1]-border,0),min(min_max[2]+border,im.shape[1]),min(min_max[3]+border,im.shape[0])]
    im = cv2.cvtColor(im[:,:,::-1], cv2.COLOR_BGR2GRAY)

    if justMask:
        im = im*mask_curr;

    im=im[min_max[1]:min_max[3],min_max[0]:min_max[2]]
    im = scipy.misc.imresize(im,size_out);

    scipy.misc.imsave(out_file,im);
        # '../scratch/test.jpg',im);

def makeResizeImAndFile():

    anno_file='../data/karina_vids/data_unprocessed/annos_all.txt'
    in_dir='../data/karina_vids/data_unprocessed'
    out_dir_files=os.path.join(in_dir,'test_train_images');
    util.mkdir(out_dir_files);
    lines=util.readLinesFromFile(anno_file);
    lines_new=[];
    out_file_new=os.path.join(out_dir_files,'annos_all.txt');

    for idx_line_curr,line_curr in enumerate(lines):
        if idx_line_curr%100==0:
            print idx_line_curr,len(lines);
        anno=line_curr.split(' ')[1];
        file_curr=line_curr.split(' ')[0];
        dir_curr=os.path.split(file_curr)[0];
        mask_dir_curr=dir_curr+'_mask';
        mask_file=os.path.join(mask_dir_curr,os.path.split(file_curr)[1]);
        out_dir_curr=dir_curr+'_rsMask';
        util.mkdir(out_dir_curr);
        out_file=os.path.join(out_dir_curr,os.path.split(file_curr)[1]);
        # print file_curr,mask_file,out_file
        try:
            saveResizeImage(file_curr,mask_file,out_file,justMask=True);
            lines_new.append(out_file+' '+anno);
            visualize.writeHTMLForFolder(out_dir_curr);
        except:
            print 'ERROR',line_curr
            continue

    util.writeFile(out_file_new,lines_new);

def getEqualList(train_list):
    anno_train=np.array([int(line_curr.split(' ')[1]) for line_curr in train_list]);
    bin_pos=anno_train[anno_train>0];
    bin_neg=anno_train[anno_train<0];
    pos_lines=list(train_list[bin_pos]);
    neg_lines=list(train_list[bin_neg]);
    random.shuffle(pos_lines);
    random.shuffle(neg_lines);
    num_keep=min(len(pos_lines),len(neg_lines));
    pos_lines=pos_lines[:num_keep];
    neg_lines=neg_lines[:num_keep];
    train_list=pos_lines+neg_lines;
    random.shuffle(train_list);
    return train_list

def writeTrainTestFiles():
    in_dir='../data/karina_vids/data_unprocessed';
    out_dir_files=os.path.join(in_dir,'test_train_images');  
    anno_file=os.path.join(out_dir_files,'annos_all.txt');

    lines_all=util.readLinesFromFile(anno_file);
    range_horses=range(1,7)
    for idx_fold,test_horse in enumerate(range_horses):
        train_horse=[str(num) for num in range_horses if num!= test_horse];
        test_horse=[str(test_horse)]
        out_file_train=os.path.join(out_dir_files,'train_'+str(idx_fold)+'.txt');
        out_file_test=os.path.join(out_dir_files,'test_'+str(idx_fold)+'.txt');
        horse_num=[line_curr.split('/')[-2][0] for line_curr in lines_all];
        anno_num=[line_curr.split(' ')[1] for line_curr in lines_all];
        
        idx_rows=np.array(list(range(len(lines_all))));
        horse_num=np.array(horse_num);
        anno_num=np.array(anno_num);

        train_bin=np.in1d(np.array(horse_num),np.array(train_horse));
        test_bin=np.in1d(np.array(horse_num),np.array(test_horse));
        print train_bin.shape,np.sum(train_bin);
        print test_bin.shape,np.sum(test_bin);

        train_list=np.array(lines_all)[train_bin];
        test_list=np.array(lines_all)[test_bin];
        train_list=getEqualList(train_list);
        print len(train_list);
        test_list=getEqualList(test_list);
        print len(test_list);
        util.writeFile(out_file_train,train_list);
        util.writeFile(out_file_test,test_list);

def writeScriptForTraining():
    path_to_th='train_khorrami_withBlur.th';

    # dir_files='../data/karina_vids/data_unprocessed/test_train_images';
    # twoClass=True;
    # model_file='../models/base_khorrami_model_1.dat'
    # experiment_name='horses_twoClass'
    
    # dir_files='../data/karina_vids/train_test_files_01';
    dir_files='../data/karina_vids/train_test_files_mask';
    twoClass=False;
    # model_file='../models/base_khorrami_model_2.dat'
    # experiment_name='horses_twoClass_01'
    mean_im_path=None;
    # onlyLast=False;

    model_file='../models/ck_khorrami_model_forFT_2.dat'
    # mean_im_path='../data/ck_96/train_test_files/train_0_mean.png';
    # std_im_path='../data/ck_96/train_test_files/train_0_std.png'
    # # experiment_name='horses_twoClass_01_ft_slr'

    experiment_name='horses_twoClass_01_ft_higherLast_mask'
    experiment_name='horses_twoClass_01_ft_slr_mask'
    experiment_name='horses_twoClass_01_ft_onlyLast_mask'
    onlyLast=True
    lower=False

    num_scripts=1;
    out_dir_meta=os.path.join('../experiments',experiment_name);
    out_script='../scripts/train_'+experiment_name;
    
    util.mkdir(out_dir_meta);
    
    iterations=50;
    saveAfter=10;
    testAfter=1;
    learningRate=0.01
    
    commands_all=[];
    
    for fold_num in range(6):
        if mean_im_path is not None:
            command = scripts_and_viz.writeBlurScript(path_to_th,out_dir_meta,dir_files,fold_num,model_file=model_file,twoClass=twoClass,iterations=iterations,saveAfter=saveAfter,learningRate=learningRate,testAfter=testAfter,mean_im_path=mean_im_path,std_im_path=std_im_path, onlyLast=onlyLast,lower=lower);
        else:
            command = scripts_and_viz.writeBlurScript(path_to_th,out_dir_meta,dir_files,fold_num,model_file=model_file,twoClass=twoClass,iterations=iterations,saveAfter=saveAfter,learningRate=learningRate,testAfter=testAfter,onlyLast=onlyLast,lower=lower);
        commands_all.append(command);

    commands=np.array(commands_all);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_script+'_'+str(idx_commands)+'.sh';
        print idx_commands,len(commands)
        print out_file_script_curr
        # print commands;
        util.writeFile(out_file_script_curr,commands);

    # util.writeFile(out_file_script,commands_all);


def createComparativeHtml():
    experiment_name='test_mean_blur';
    out_dir_meta_meta=os.path.join(dir_server,'expression_project','experiments');

    out_dir_meta_pre='horses_twoClass_01_ft_';
    out_dir_meta_posts=['slr_mask','onlyLast_mask','higherLast_mask'];
    out_dir_metas=[os.path.join(out_dir_meta_meta,out_dir_meta_pre+out_dir_meta_post) for out_dir_meta_post in out_dir_meta_posts];

    dir_files='../data/karina_vids/train_test_files_01'
    for out_dir_meta in out_dir_metas:
        dir_comps=[out_dir_meta]
        range_folds=[0];
        # range(6);
        
        expressions=[None]
        right=True;

        npy_file_pred='1_pred_labels.npy';
        npy_file_gt='1_gt_labels.npy';
        posts=['_org','_gb_gcam','_gb_gcam_pred','_hm','_hm_pred','_gb_gcam_org','_gb_gcam_org_pred']
        
        batchSizeTest=128;
        for expression_curr in expressions:
        
            out_file_html=os.path.join(out_dir_meta,str(expression_curr)+'.html');
            ims_all=[];
            captions_all=[];
            for fold_num in range_folds:    
                data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
                gt_labels_file=os.path.join(dir_comps[0],str(fold_num),'test_images',npy_file_gt);
                gt_labels=np.load(gt_labels_file);    
                num_batches=int(math.ceil(len(gt_labels)/128.0))
                im_pre=np.array([str(batch_num)+'_'+str(im_num) for batch_num in range(1,num_batches+1) for im_num in range(1,batchSizeTest+1)]);
                im_pre=im_pre[:len(gt_labels)];
                if expression_curr is not None:
                    bin_keep=gt_labels==expression_curr;
                else:
                    bin_keep=gt_labels==gt_labels
                gt_rel=gt_labels[bin_keep]
                # print bin_keep.shape;
                # print im_pre.shape
                im_pre_rel=im_pre[bin_keep];

                dir_ims=[os.path.join(dir_curr,str(fold_num),'test_images') for dir_curr in dir_comps];
                pred_rels=[np.load(os.path.join(dir_curr,npy_file_pred))[bin_keep] for dir_curr in dir_ims];

                for idx_im_pre_curr,im_pre_curr in enumerate(im_pre_rel):
                    for idx_epoch in range(len(dir_ims)):
                        im_row=[util.getRelPath(os.path.join(dir_ims[idx_epoch],im_pre_curr+post_curr+'.jpg'),dir_server) for post_curr in posts];
                        caption_pre='right' if pred_rels[idx_epoch][idx_im_pre_curr]==gt_rel[idx_im_pre_curr] else 'wrong'
                        caption_row=[str(idx_im_pre_curr)+' '+caption_pre+' '+str(int(gt_rel[idx_im_pre_curr]))+' '+str(int(pred_rels[idx_epoch][idx_im_pre_curr]))]*len(im_row);
                        ims_all.append(im_row[:]);
                        captions_all.append(caption_row[:])

            visualize.writeHTML(out_file_html,ims_all,captions_all,100,100);
            print out_file_html.replace(dir_server,click_str);

def writeTrainTestFiles01():
    train_test_dir='../data/karina_vids/train_test_files';
    out_dir='../data/karina_vids/train_test_files_01';
    util.mkdir(out_dir);
    num_folds=6;
    in_file_pres=['train_','test_'];
    range_folds=range(num_folds);

    for num_fold in range_folds:
        for in_file_pre in in_file_pres:
            in_file=os.path.join(train_test_dir,in_file_pre+str(num_fold)+'.txt');
            out_file=os.path.join(out_dir,in_file_pre+str(num_fold)+'.txt');
            lines=util.readLinesFromFile(in_file);
            # lines=[line_curr.split(' ')[0]+' 0' if line_curr.split(' ')[1]=='-1' else 
            lines_new=[];
            for line_curr in lines:
                line_split=line_curr.split(' ')
                line_split[1]='0' if line_split[1]=='-1' else '1'
                lines_new.append(' '.join(line_split));

            util.writeFile(out_file,lines_new);

def writeTrainTestFilesMaskIm():
    train_test_dir='../data/karina_vids/train_test_files_01';
    out_dir='../data/karina_vids/train_test_files_mask';
    
    util.mkdir(out_dir);
    num_folds=6;
    in_file_pres=['train_','test_'];
    train_pre=in_file_pres[0]
    range_folds=range(num_folds);

    for num_fold in range_folds:
        for in_file_pre in in_file_pres:
            in_file=os.path.join(train_test_dir,in_file_pre+str(num_fold)+'.txt');
            out_file=os.path.join(out_dir,in_file_pre+str(num_fold)+'.txt');
            lines=util.readLinesFromFile(in_file);
            lines_new=[line_curr.replace('_rs/','_rsMask/') for line_curr in lines];
            util.writeFile(out_file,lines_new);
    
    preprocess_data.saveCKMeanSTDImages(out_dir,train_pre,resize_size=None,num_folds=num_folds)



def main():

    # writeTrainTestFilesMaskIm();
    # makeResizeImAndFile()
    createComparativeHtml();
    # writeScriptForTraining()

                


    


    






if __name__=='__main__':
    main();