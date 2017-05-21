import sys
import os;
import numpy as np;
import math;
import visualize;
import util;

def makeHtml(test_folder,test_file,batchSizeTest):
    out_file_html=os.path.join(test_folder,'results.html');

    
    npy_file_pred='1_pred_labels.npy';
    npy_file_gt='1_gt_labels.npy';
    posts=['_org','_gb_gcam','_gb_gcam_pred','_gb_gcam_org','_gb_gcam_org_pred','_hm','_hm_pred'];
    
    ims_all=[];
    captions_all=[];
    gt_labels_file=os.path.join(test_folder,npy_file_gt);
    gt_labels=np.load(gt_labels_file);    
    pred_labels=np.load(os.path.join(test_folder,npy_file_pred));

    num_batches=int(math.ceil(len(gt_labels)/float(batchSizeTest)))
    im_pre=np.array([str(batch_num)+'_'+str(im_num) for batch_num in range(1,num_batches+1) for im_num in range(1,batchSizeTest+1)]);
    
    im_pre=im_pre[:len(gt_labels)];
    
    for idx_im_pre_curr,im_pre_curr in enumerate(im_pre):
        im_row=[os.path.join('./',im_pre_curr+post_curr+'.jpg') for post_curr in posts];
        caption_pre='right' if pred_labels[idx_im_pre_curr]==gt_labels[idx_im_pre_curr] else 'wrong'
        caption_row=[str(idx_im_pre_curr)+' '+caption_pre+' '+str(int(gt_labels[idx_im_pre_curr]))+' '+str(int(pred_labels[idx_im_pre_curr]))]*len(im_row);
        ims_all.append(im_row[:]);
        captions_all.append(caption_row[:])

    visualize.writeHTML(out_file_html,ims_all,captions_all,224,224);
    print out_file_html

def main(args):
    test_folder=args[1];
    test_file=args[2];
    batchSizeTest=int(args[3])
    makeHtml(test_folder,test_file,batchSizeTest);
    

if __name__=='__main__':
    main(sys.argv);