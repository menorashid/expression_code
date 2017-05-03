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

    # args=args[10:];
    p=multiprocessing.Pool(multiprocessing.cpu_count());
    t=time.time();
    p.map(saveFrames,args);
    print (time.time()-t);


def saveMasks():
    deploy_path='../../fcn.berkeleyvision.org/voc-fcn8s/deploy.prototxt'
    model_path='../../fcn.berkeleyvision.org/voc-fcn8s/fcn8s-heavy-pascal.caffemodel'
    net=getNet(deploy_path,model_path);
    
    in_dir_meta='../data/karina_vids/data_unprocessed';
    in_dirs=[os.path.join(in_dir_meta,dir_curr) for dir_curr in os.listdir(in_dir_meta) if os.path.isdir(os.path.join(in_dir_meta,dir_curr))]
    # print in_dirs;
    # print len(in_dirs);
    out_dirs=[dir_curr+'_mask' for dir_curr in in_dirs]

    for idx_dir,(in_dir,out_dir) in enumerate(zip(in_dirs,out_dirs)):
        print idx_dir,in_dir,out_dir
        util.mkdir(out_dir);
        testVOC(net,in_dir,out_dir)

def main():

    pass;


if __name__=='__main__':
    main();