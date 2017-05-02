import util;
import ffmpy
import visualize;
import os;
import numpy as np
from PIL import Image
import scipy.misc;
import caffe
import re;

def testVOC(net,in_file,out_file):

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(in_file)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    print in_.shape

    # load net
    # net = caffe.Net(deploy_path,model_path, caffe.TEST)
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    # out = out.argmax(axis=1) # Get the labels at each pixel
    print out.shape
    out[out==13]=255;
    out[out<255]=0;
    # out = out.transpose(1, 2, 0) # Reshape the output into an image
    # out = np.tile(out, (1,3))
    # print np.min(out),np.max(out);
    # out=np.array((out-np.min(out))/np.max(out)*255,dtype=np.uint8);
    scipy.misc.imsave(out_file,out)
    print out_file;

def getNet(deploy_path,model_path,gpu_num=0):
    caffe.set_mode_gpu()
    caffe.set_device(gpu_num)
    net = caffe.Net(deploy_path,model_path, caffe.TEST)
    return net;
    

def main():

    # string_to_match='data-filename="';
    # file_name='untitled.html';
    # with open(file_name,'rb') as f:
    #     lines=f.read();

    # idx_match=[m.start() for m in re.finditer(string_to_match, lines)];
    # idx_match.append(len(lines));
    # for idx_idx_curr,idx_match_curr in enumerate(idx_match[:-1]):
    #     idx_start=idx_match_curr+len(string_to_match);
    #     idx_end=lines[idx_start:idx_match[idx_idx_curr+1]].find('"');
    #     idx_end=idx_start+idx_end;
    #     file_curr=lines[idx_start:idx_end];
    #     file_curr='Dropbox/'+file_curr;
    #     # print file_curr;
    #     file_curr=util.escapeString(file_curr);
    #     file_curr=file_curr.replace(' ','\ ');
    #     print file_curr+' ',
    #     # break;
    
    # return
    in_dir='../scratch/2_3b';
    out_dir=os.path.join(in_dir+'_pred');
    util.mkdir(out_dir);
    im_name='2_3b_3.jpg';
    
    out_file=os.path.join(out_dir,im_name);
    in_file=os.path.join(in_dir,im_name);

    deploy_path='../../fcn.berkeleyvision.org/voc-fcn8s/deploy.prototxt'
    model_path='../../fcn.berkeleyvision.org/voc-fcn8s/fcn8s-heavy-pascal.caffemodel'
    net=getNet(deploy_path,model_path);

    testVOC(deploy_path,model_path,in_file,out_file);
    




    return
    video_path='../scratch/#2_3b.mts';
    video_name=os.path.split(video_path)[1];
    video_name=video_name[video_name.index('#')+1:video_name.rindex('.')];
    out_dir=os.path.join('../scratch',video_name);
    util.mkdir(out_dir);

    command=[];
    command.extend(['ffmpeg','-i']);
    command.append(video_path);
    command.extend(['-filter:v','fps=fps=1/60']);
    command.append(os.path.join(out_dir,video_name+'_%0d.jpg'));
    command=' '.join(command);
    print command;
    os.system(command);
    # ffmpeg -i input.mp4 -filter:v fps=fps=1/60 ffmpeg_%0d.bmp


if __name__=='__main__':
    main();