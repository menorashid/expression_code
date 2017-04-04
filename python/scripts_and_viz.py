import numpy as np;
import cv2;
import util;
import os;
import visualize;
import scipy;
import math;
dir_server='/home/SSD3/maheen-data/';
click_str='http://vision1.idav.ucdavis.edu:1000/';

def writeCKScripts():
    # experiment_name='khorrami_basic_aug_fix_resume_again';
    # out_dir_meta='../experiments/'+experiment_name;
    # util.mkdir(out_dir_meta);
    # out_script='../scripts/train_'+experiment_name+'.sh';
    # path_to_th='train_khorrami_basic.th';

    # dir_files='../data/ck_96/train_test_files';
    # resume_dir_meta='../experiments/khorrami_basic_aug_fix_resume'
    # num_folds=10;


    experiment_name='khorrami_basic_tfd_withBias';
    out_dir_meta='../experiments/'+experiment_name;
    util.mkdir(out_dir_meta);
    out_script='../scripts/train_'+experiment_name+'.sh';
    path_to_th='train_khorrami_basic.th';

    dir_files='../data/tfd/train_test_files';
    resume_dir_meta=None
    # '../experiments/khorrami_basic_tfd_resume'
    noBias=False;
    num_folds=5;

    learningRate = 0.01;
    batchSizeTest=128;
    batchSize=128;

    commands=[];
    # print '{',
    for fold_num in range(num_folds):
        
        data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
        val_data_path=os.path.join(dir_files,'test_'+str(fold_num)+'.txt');
        mean_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_mean.png');
        std_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_std.png');

        epoch_size=len(util.readLinesFromFile(data_path))/batchSize;
        totalSizeTest=len(util.readLinesFromFile(val_data_path));
        # print str(batchSizeTest)+',',
        iterationsTest=int(math.ceil(totalSizeTest/float(batchSizeTest)));
        outDir=os.path.join(out_dir_meta,str(fold_num));

        command=['th',path_to_th];
        if resume_dir_meta is not None:
            model_path_resume=os.path.join(resume_dir_meta,str(fold_num),'final','model_all_final.dat');
            command = command+['-model',model_path_resume];            

        command = command+['-mean_im_path',mean_im_path];
        command = command+['-std_im_path',std_im_path];
        command = command+['-batchSize',batchSize];
        
        command = command+['learningRate',learningRate];
        if noBias:
            command = command+['-noBias'];

        command = command+['-iterations',500*epoch_size];
        command = command+['-saveAfter',50*epoch_size];
        command = command+['-testAfter',10*epoch_size];
        command = command+['-dispAfter',1*epoch_size];
        command = command+['-dispPlotAfter',10*epoch_size];

        command = command+['-val_data_path',val_data_path];
        command = command+['-data_path',data_path];
        
        command = command+['-iterationsTest',iterationsTest];
        command = command+['-batchSizeTest',batchSizeTest]
        
        command = command+['-outDir',outDir];
        command = command+['-modelTest',os.path.join(outDir,'final','model_all_final.dat')];
        command = [str(c_curr) for c_curr in command];
        command=' '.join(command);
        print command;
        commands.append(command);
    # print '}';
    print out_script
    util.writeFile(out_script,commands);


def writeTFDSchemeScripts():
    
    experiment_name='khorrami_basic_tfd_schemes_fullBlur';
    out_dir_meta='../experiments/'+experiment_name;
    util.mkdir(out_dir_meta);
    out_script='../scripts/train_'+experiment_name;
    # +'.sh';
    path_to_th='train_khorrami_withBlur.th';

    dir_files='../data/tfd/train_test_files';
    resume_dir_meta=None
    # '../experiments/khorrami_basic_tfd_resume'
    num_folds=5;

    learningRate = 0.01;
    batchSizeTest=128;
    batchSize=128;
    ratioBlur=1.0;
    schemes=['mix','mixcl'];
    
    num_scripts=1;
    # print '{',
    commands=[];
    for fold_num in range(num_folds):
        for scheme in schemes:
            out_dir_meta_curr=os.path.join(out_dir_meta,scheme);
            util.mkdir(out_dir_meta_curr)
            data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
            val_data_path=os.path.join(dir_files,'test_'+str(fold_num)+'.txt');
            mean_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_mean.png');
            std_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_std.png');

            
            epoch_size=len(util.readLinesFromFile(data_path))/batchSize;
            totalSizeTest=len(util.readLinesFromFile(val_data_path));
            # print str(batchSizeTest)+',',
            
            iterationsTest=int(math.ceil(totalSizeTest/float(batchSizeTest)));
            outDir=os.path.join(out_dir_meta_curr,str(fold_num));

            
            command=['th',path_to_th];
            if resume_dir_meta is not None:
                model_path_resume=os.path.join(resume_dir_meta,str(fold_num),'final','model_all_final.dat');
                command = command+['-model',model_path_resume];            

            command = command+['-mean_im_path',mean_im_path];
            command = command+['-std_im_path',std_im_path];
            command = command+['-batchSize',batchSize];

            command = command+['-ratioBlur',ratioBlur];
            command = command+['-incrementDifficultyAfter',10*epoch_size];
            command = command+['-scheme',scheme];
            
            command = command+['learningRate',learningRate];

            command = command+['-iterations',1000*epoch_size];
            command = command+['-saveAfter',100*epoch_size];
            command = command+['-testAfter',10*epoch_size];
            command = command+['-dispAfter',1*epoch_size];
            command = command+['-dispPlotAfter',10*epoch_size];

            command = command+['-val_data_path',val_data_path];
            command = command+['-data_path',data_path];
            
            command = command+['-iterationsTest',iterationsTest];
            command = command+['-batchSizeTest',batchSizeTest]


            
            command = command+['-outDir',outDir];
            command = command+['-modelTest',os.path.join(outDir,'final','model_all_final.dat')];
            command = [str(c_curr) for c_curr in command];
            command=' '.join(command);
            # print command;
            commands.append(command);
        # print '}';
    print out_script

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        # print commands;
        util.writeFile(out_file_script_curr,commands);


    # util.writeFile(out_script,commands);


def makeHTML_withPosts(out_file_html,im_dirs,num_ims_all,im_posts,gt_labels_all,pred_labels_all):
    im_list=[];
    caption_list=[];

    for im_dir,num_ims,gt_labels,pred_labels in zip(im_dirs,num_ims_all,gt_labels_all,pred_labels_all):

        for num_im in num_ims:
            ims=['1_'+str(num_im)+im_post_curr for im_post_curr in im_posts]
            im_list_curr=[util.getRelPath(os.path.join(im_dir,im_curr),dir_server) for im_curr in ims];
            gt=gt_labels[num_im-1];
            pred=pred_labels[num_im-1];
            caption_curr='right' if gt==pred else 'wrong';
            caption_curr=caption_curr+' '+str(gt)+' '+str(pred);
            caption_list_curr=[caption_curr+' '+im_curr[:im_curr.rindex('.')].lstrip('_') for im_curr in im_posts];
            caption_list.append(caption_list_curr);
            im_list.append(im_list_curr);

    visualize.writeHTML(out_file_html,im_list,caption_list);

def script_vizTestGradCam():
    dir_meta_pre=os.path.join(dir_server,'expression_project');
    out_dir_meta=os.path.join(dir_meta_pre,'scratch/test_grad_cam');
    test_pre='../data/ck_96/train_test_files/test_';
    # expressions=['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'];
    # expressions=['pred', 'gt']
    expressions=['']
    npy_file='1_pred_labels.npy';

    # , 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'];
    pres=['_gb_gcam_org_','_hm_'];
    pres=['_gb_gcam_th_','_gb_gcam_','_gaussian_','_blur_'];
    # pres=['_gb_gcam_th_','_gb_gcam_','_gaussian_','_blur_'];
    pres=['_org','_gb_gcam','_hm','_gb_gcam_org','_gb_gcam_th','_gaussian','_blur']

    num_folds=10;
    im_posts=['_org.jpg']+[pre+exp_curr+'.jpg' for pre in pres for exp_curr in expressions]; 
            # ['_gb_gcam_org_'+exp_curr+'.jpg' for exp_curr in expressions]+\
            # ['_hm_'+exp_curr+'.jpg' for exp_curr in expressions];
            # ['_gb_gcam_'+exp_curr+'.jpg' for exp_curr in expressions]+\
            
    out_file_html=os.path.join(out_dir_meta,'test_buildBlurryBatch.html');
    # out_file_html=os.path.join(out_dir_meta,'viz_all.html');
    im_dirs=[];
    num_ims_all=[];
    gt_labels_all=[];
    pred_labels_all=[];
    
    for fold_num in range(10,10+1):

        out_dir=os.path.join(out_dir_meta,str(fold_num));
        # out_dir=out_dir_meta;
        # out_file_html=out_dir+'.html';
        # im_dir=os.path.join(out_dir,'images');
        im_dir=os.path.join(out_dir,'test_buildBlurryBatch');
        # out_file_html=os.path.join(out_dir,os.path.split(out_dir)[1]+'.html');
        fold_num=0
        # fold_num%10;

        # pred_labels=np.load(os.path.join(out_dir,npy_file)).astype(int);
        gt_data=util.readLinesFromFile(test_pre+str(fold_num)+'.txt')
        gt_labels=np.array([int(line_curr.split(' ')[1]) for line_curr in gt_data])+1;
        pred_labels = gt_labels;


        num_ims=list(range(1,len(gt_data)+1));
        im_dirs.append(im_dir)
        num_ims_all.append(num_ims);
        gt_labels_all.append(gt_labels);
        pred_labels_all.append(pred_labels);
        
    makeHTML_withPosts(out_file_html,im_dirs,num_ims_all,im_posts,gt_labels_all,pred_labels_all);
    print out_file_html.replace(dir_server,click_str);

def writeTestGradCamScript():
    num_folds=10;

    experiment_name='test_grad_cam';
    dir_files='../data/ck_96/train_test_files';
    out_dir_meta=os.path.join('../scratch',experiment_name);
    util.mkdir(out_dir_meta);
    out_script=os.path.join('../scripts',experiment_name+'.sh');
    path_to_th='grad_cam.th';
    dir_models='../experiments/khorrami_basic_aug_fix_resume_again';

    commands=[];
    for fold_num in range(num_folds):
        val_data_path=os.path.join(dir_files,'test_'+str(fold_num)+'.txt');
        mean_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_mean.png');
        std_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_std.png');
        batchSizeTest=len(util.readLinesFromFile(val_data_path));

        command=['th',path_to_th];
        command = command+['-model',os.path.join(dir_models,str(fold_num),'final','model_all_final.dat')];            

        command = command+['-mean_im_path',mean_im_path];
        command = command+['-std_im_path',std_im_path];
        
        command = command+['-val_data_path',val_data_path];
        
        command = command+['-iterationsTest',1];
        command = command+['-batchSizeTest',batchSizeTest];
        
        command = command+['-outDir',os.path.join(out_dir_meta,str(fold_num))];
        command = [str(c_curr) for c_curr in command];
        command=' '.join(command);
        print command;
        commands.append(command);

    util.writeFile(out_script,commands);

def main():
    writeTFDSchemeScripts();
    # writeCKScripts();
    # writeTestGradCamScript()
    # script_vizTestGradCam();

if __name__=='__main__':
    main();