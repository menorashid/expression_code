import numpy as np;
import cv2;
import util;
import os;
import visualize;
import scipy;
import math;
dir_server='/home/SSD3/maheen-data/';
click_str='http://vision1.idav.ucdavis.edu:1000/';


def writeBlurScript(\
    path_to_th,
    out_dir_meta,
    dir_files,
    fold_num,
    model_file=None,
    batchSize=128,
    learningRate=0.01,
    scheme='ncl',
    ratioBlur=0,
    incrementDifficultyAfter=-1,
    startingActivation=0,
    fixThresh=-1,
    activationThreshMax=0.5,
    iterations=1000,
    saveAfter=100,
    testAfter=10,
    dispAfter=1,
    dispPlotAfter=10,
    batchSizeTest=128,
    modelTest=None,
    resume_dir_meta=None,
    twoClass=False,
    val_data_path=None,
    mean_im_path=None,
    std_im_path=None,
    onlyLast=False,
    lower=False,
    input_size=None,
    withAnno=False,
    weights=False
    ):
    data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
    if val_data_path is None:
        if withAnno:
            val_data_path=os.path.join(dir_files,'test_'+str(fold_num)+'_withAnno.txt');
        else:
            val_data_path=os.path.join(dir_files,'test_'+str(fold_num)+'.txt');
    
    if mean_im_path is None:
        mean_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_mean.png');
    if std_im_path is None:
        std_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_std.png');

    epoch_size=len(util.readLinesFromFile(data_path))/batchSize;
    totalSizeTest=len(util.readLinesFromFile(val_data_path));
    # print str(batchSizeTest)+',',
    iterationsTest=int(math.ceil(totalSizeTest/float(batchSizeTest)));
    outDir=os.path.join(out_dir_meta,str(fold_num));

    command=['th',path_to_th];
    if model_file is not None:
        command = command+['-model',model_file];            
    elif resume_dir_meta is not None:
        model_path_resume=os.path.join(resume_dir_meta,str(fold_num),'final','model_all_final.dat');
        command = command+['-model',model_path_resume];            
    if mean_im_path!='':
        command = command+['-mean_im_path',mean_im_path];
    if std_im_path!='':
        command = command+['-std_im_path',std_im_path];
    command = command+['-batchSize',batchSize];
    
    command = command+['learningRate',learningRate];

    command = command+['-scheme',scheme];
    command = command+['-ratioBlur',ratioBlur];
    if incrementDifficultyAfter>=0:
        command = command+['-incrementDifficultyAfter',incrementDifficultyAfter*epoch_size];
    else:
        command = command+['-incrementDifficultyAfter',-1];
    command = command+['-startingActivation',startingActivation];
    command = command+['-fixThresh',fixThresh];
    command = command+['-activationThreshMax',activationThreshMax];
    
    command = command+['-iterations',iterations*epoch_size];
    command = command+['-saveAfter',saveAfter*epoch_size];
    command = command+['-testAfter',testAfter*epoch_size];
    command = command+['-dispAfter',dispAfter*epoch_size];
    command = command+['-dispPlotAfter',dispPlotAfter*epoch_size];

    command = command+['-val_data_path',val_data_path];
    command = command+['-data_path',data_path];
    
    command = command+['-iterationsTest',iterationsTest];
    command = command+['-batchSizeTest',batchSizeTest]
    
    command = command+['-outDir',outDir];
    if modelTest is None:
        command = command+['-modelTest',os.path.join(outDir,'final','model_all_final.dat')];
    else:
        command = command+['-modelTest',modelTest];

    if twoClass:
        command = command+['-twoClass'];

    if onlyLast:
        command = command+['-onlyLast'];

    if lower:
        command = command+['-lower'];

    if input_size is not None:
        command = command+['-input_size',input_size];        

    if weights:
        weights_file=os.path.join(dir_files,'weights_'+str(fold_num)+'.npy'); 
        command = command+['-weights_file',weights_file];                   

    command = [str(c_curr) for c_curr in command];
    command=' '.join(command);
    return command;

def writeCKScripts():
    experiment_name='khorrami_ck_rerun';
    out_dir_meta='../experiments/'+experiment_name;
    util.mkdir(out_dir_meta);
    out_script='../scripts/train_'+experiment_name+'.sh';
    # path_to_th='train_khorrami_basic.th';
    path_to_th='train_khorrami_withBlur.th';

    dir_files='../data/ck_96/train_test_files';
    # resume_dir_meta='../experiments/khorrami_basic_aug_fix_resume'
    num_folds=10;

    resume_dir_meta=None
    noBias=False;
    learningRate = 0.01;
    batchSizeTest=128;
    batchSize=128;
    ratioBlur=0.0;

    # experiment_name='khorrami_basic_tfd_withBias';
    # out_dir_meta='../experiments/'+experiment_name;
    # util.mkdir(out_dir_meta);
    # out_script='../scripts/train_'+experiment_name+'.sh';
    # path_to_th='train_khorrami_basic.th';

    # dir_files='../data/tfd/train_test_files';
    # resume_dir_meta=None
    # # '../experiments/khorrami_basic_tfd_resume'
    # noBias=False;
    # num_folds=5;

    # learningRate = 0.01;
    # batchSizeTest=128;
    # batchSize=128;

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

        command = command+['-ratioBlur',ratioBlur];
        command = command+['-incrementDifficultyAfter',0];
            

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
        print command;
        commands.append(command);
    # print '}';
    print out_script
    util.writeFile(out_script,commands);


def writeTFDSchemeScripts():
    
    # experiment_name='khorrami_basic_tfd_schemes_fullBlur';
    experiment_name='noBlur_meanFirst_pixel_augment';
    out_dir_meta='../experiments/'+experiment_name;
    util.mkdir(out_dir_meta);
    out_script='../scripts/train_'+experiment_name;
    # +'.sh';
    path_to_th='train_khorrami_withBlur.th';

    dir_files='../data/tfd/train_test_files';
    resume_dir_meta='../experiments/noBlur_meanFirst_pixel_augment/noBlur_meanFirst_7out'
    # '../experiments/khorrami_basic_tfd_resume'
    num_folds=5;

    learningRate = 0.01;
    batchSizeTest=128;
    batchSize=128;
    ratioBlur=0.0;
    schemes=['mix','mixcl'];
    # incrementDifficultyAfter=10*epoch_size;
    incrementDifficultyAfter=0;
    schemes=['noBlur_meanFirst_7out_resume'];
    # model='../models/base_khorrami_model_7.dat';
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
            else:
                command = command+['-model',model];            

            command = command+['-mean_im_path',mean_im_path];
            command = command+['-std_im_path',std_im_path];
            command = command+['-batchSize',batchSize];

            command = command+['-ratioBlur',ratioBlur];
            command = command+['-incrementDifficultyAfter',0];
            # 1*epoch_size];
            command = command+['-scheme',scheme];
            
            command = command+['learningRate',learningRate];

            command = command+['-iterations',300*epoch_size];
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


def makeHTML_withPosts(out_file_html,im_dirs,num_ims_all,im_posts,gt_labels_all,pred_labels_all,num_batches_all=[1]):
    im_list=[];
    caption_list=[];

    for im_dir,num_ims,gt_labels,pred_labels,num_batches in zip(im_dirs,num_ims_all,gt_labels_all,pred_labels_all,num_batches_all):
        idx_curr=-1;
        
        for num_batch in num_batches:
            for num_im in num_ims:
                
                idx_curr=idx_curr+1;
                if idx_curr==len(gt_labels):
                    break;

                ims=[str(num_batch)+'_'+str(num_im)+im_post_curr for im_post_curr in im_posts]
                im_list_curr=[util.getRelPath(os.path.join(im_dir,im_curr),dir_server) for im_curr in ims];    
                gt=gt_labels[idx_curr];
                pred=pred_labels[idx_curr];
                caption_curr='right' if gt==pred else 'wrong';
                caption_curr=caption_curr+' '+str(gt)+' '+str(pred);
                caption_list_curr=[caption_curr+' '+im_curr[:im_curr.rindex('.')].lstrip('_') for im_curr in im_posts];
                caption_list.append(caption_list_curr);
                im_list.append(im_list_curr);

    visualize.writeHTML(out_file_html,im_list,caption_list);

def script_vizTestGradCam():
    dir_meta_pre=os.path.join(dir_server,'expression_project');
    out_dir_meta=os.path.join(dir_meta_pre,'experiments/khorrami_ck_rerun');
    test_pre='../data/ck_96/train_test_files/test_';
    # expressions=['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'];
    # expressions=['pred', 'gt']
    expressions=['']
    npy_file='1_pred_labels.npy';
    npy_file_gt='1_gt_labels.npy';

    # , 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'];
    pres=['_gb_gcam_org_','_hm_'];
    pres=['_gb_gcam_th_','_gb_gcam_','_gaussian_','_blur_'];
    # pres=['_gb_gcam_th_','_gb_gcam_','_gaussian_','_blur_'];
    pres=['_org','_gb_gcam','_hm','_gb_gcam_org','_gb_gcam_th','_gaussian','_blur']
    pres=['_org','_gb_gcam','_hm','_gb_gcam_org']
    # ,'_gaussian','_blur']

    num_folds=2;
    im_posts=[pre+exp_curr+'.jpg' for pre in pres for exp_curr in expressions]; 
            # ['_gb_gcam_org_'+exp_curr+'.jpg' for exp_curr in expressions]+\
            # ['_hm_'+exp_curr+'.jpg' for exp_curr in expressions];
            # ['_gb_gcam_'+exp_curr+'.jpg' for exp_curr in expressions]+\
            
    out_file_html=os.path.join(out_dir_meta,'test_results.html');
    # out_file_html=os.path.join(out_dir_meta,'viz_all.html');
    im_dirs=[];
    num_ims_all=[];
    gt_labels_all=[];
    pred_labels_all=[];
    num_batches_all=[];
    for fold_num in range(1,num_folds):

        out_dir=os.path.join(out_dir_meta,str(fold_num));
        # out_dir=out_dir_meta;
        # out_file_html=out_dir+'.html';
        im_dir=os.path.join(out_dir,'test_images');
        # im_dir=os.path.join(out_dir,'test_buildBlurryBatch');
        # out_file_html=os.path.join(out_dir,os.path.split(out_dir)[1]+'.html');
        # fold_num=0
        # fold_num%10;

        pred_labels=np.load(os.path.join(im_dir,npy_file)).astype(int);
        gt_labels=np.load(os.path.join(im_dir,npy_file_gt)).astype(int);
        # gt_data=util.readLinesFromFile(test_pre+str(fold_num)+'.txt')
        # gt_labels=np.array([int(line_curr.split(' ')[1]) for line_curr in gt_data])+1;
        # pred_labels = gt_labels;


        num_ims=list(range(1,128+1));
        num_batches_all.append(range(1,int(math.ceil(len(gt_labels)/128.0))+1));
        im_dirs.append(im_dir)
        num_ims_all.append(num_ims);
        gt_labels_all.append(gt_labels);
        pred_labels_all.append(pred_labels);
        
    makeHTML_withPosts(out_file_html,im_dirs,num_ims_all,im_posts,gt_labels_all,pred_labels_all,num_batches_all);
    print out_file_html.replace(dir_server,click_str);

def script_vizCompareActivations():
    dir_meta_pre=os.path.join(dir_server,'expression_project','experiments');
    out_dir_metas=[os.path.join(dir_meta_pre,'khorrami_basic_tfd_schemes_halfBlur/ncl'),os.path.join(dir_meta_pre,'khorrami_basic_tfd_resume_again')];
    test_pre='../data/tfd/train_test_files/test_';
    
    pres=['_gb_gcam_org_','_hm_'];
    pres=['_gb_gcam_th_','_gb_gcam_','_gaussian_','_blur_'];
    # pres=['_gb_gcam_th_','_gb_gcam_','_gaussian_','_blur_'];
    pres=['_gb_gcam','_hm','_gb_gcam_org','_gb_gcam_th','_gaussian','_blur']

    im_posts=['_org.jpg']+[pre+'.jpg' for pre in pres];
    # [pre+exp_curr+'.jpg' for pre in pres for exp_curr in expressions]; 
    out_file_html=os.path.join(out_dir_metas[0],'comparison.html');
    
    num_batches=range(1,7+1);

    im_dirs=[];
    num_ims_all=[];
    gt_labels_all=[];
    pred_labels_all=[];
    
    fold_num=0;

    for out_dir_meta in out_dir_metas:
        out_dir=os.path.join(out_dir_meta,str(fold_num));
        # out_dir=out_dir_meta;
        # out_file_html=out_dir+'.html';
        # im_dir=os.path.join(out_dir,'images');
        im_dir=os.path.join(out_dir,'test_images');
        # out_file_html=os.path.join(out_dir,os.path.split(out_dir)[1]+'.html');
        
        # pred_labels=np.load(os.path.join(out_dir,npy_file)).astype(int);
        gt_data=util.readLinesFromFile(test_pre+str(fold_num)+'.txt')
        gt_labels=np.array([int(line_curr.split(' ')[1]) for line_curr in gt_data])+1;
        pred_labels = gt_labels;

        num_ims=list(range(1,128+1));
        im_dirs.append(im_dir)
        num_ims_all.append(num_ims);
        gt_labels_all.append(gt_labels);
        pred_labels_all.append(os.path.split(out_dir_meta)[1][:3]);
    

    im_list=[];
    caption_list=[];

    for num_batch in num_batches:
        for num_im in num_ims:
            for im_dir,gt_labels,type_scheme in zip(im_dirs,gt_labels_all,pred_labels_all):
                ims=[str(num_batch)+'_'+str(num_im)+im_post_curr for im_post_curr in im_posts]
                im_list_curr=[util.getRelPath(os.path.join(im_dir,im_curr),dir_server) for im_curr in ims];
                gt_idx=128*(num_batch-1)+(num_im-1);
                # print gt_idx;
                if gt_idx>=len(gt_labels):
                    break;
                    # print num_batch,num_im
                    # raw_input();
                gt=gt_labels[gt_idx];
                # pred=pred_labels[num_im-1];
                caption_curr=str(num_batch)+'_'+str(num_im)+' '+str(gt)+' '+type_scheme;
                # 'right' if gt==pred else 'wrong';
                # caption_curr=caption_curr+' '+str(gt)+' '+str(pred);
                caption_list_curr=[caption_curr+' '+im_curr[:im_curr.rindex('.')].lstrip('_') for im_curr in im_posts];
                caption_list.append(caption_list_curr);
                im_list.append(im_list_curr);

    visualize.writeHTML(out_file_html,im_list,caption_list);

    # makeHTML_withPosts(out_file_html,im_dirs,num_ims_all,im_posts,gt_labels_all,pred_labels_all);
    print out_file_html.replace(dir_server,click_str);

def writeCKScripts_viz_inc():
    experiment_name='test_mean_blur';
    # out_dir_meta='../experiments/'+experiment_name;
    out_dir_meta='../scratch/'+experiment_name;
    util.mkdir(out_dir_meta);
    out_script='../scripts/test_'+experiment_name;
    # +'.sh';
    # path_to_th='train_khorrami_basic.th';
    path_to_th='train_khorrami_withBlur.th';

    dir_files='../data/ck_96/train_test_files';
    # resume_dir_meta='../experiments/khorrami_basic_aug_fix_resume'
    num_folds=1;

    resume_dir_meta=None
    noBias=False;
    learningRate = 0.01;
    batchSizeTest=128;
    batchSize=128;
    ratioBlur=0.0;

    out_dir_model='../experiments/khorrami_ck_rerun';
    
    # print '{',
    # out_script_curr=out_script+'_'+str(fold_num)+'.sh';
    commands=[];
    for fold_num in range(num_folds):
        for epoch_num in range(100,1100,100):
            out_dir_meta_curr=os.path.join(out_dir_meta,str(epoch_num));
            util.mkdir(out_dir_meta_curr);
            out_dir_model_curr=os.path.join(out_dir_model,str(fold_num));
            data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
            epoch_size=len(util.readLinesFromFile(data_path))/batchSize;
            inc_num=epoch_size*epoch_num;
            modelTest=os.path.join(out_dir_model_curr,'intermediate','model_all_'+str(inc_num)+'.dat');
            assert os.path.exists(modelTest);
            command=writeBlurScript(path_to_th,out_dir_meta_curr,dir_files,fold_num,modelTest=modelTest);
            # print command;
            commands.append(command);
    
    print out_script+'.sh'
    print len(commands);
    util.writeFile(out_script+'.sh',commands);


def writeHTMLs_viz_inc():
    # experiment_name='ck_inc_viz';
    # out_dir_meta=os.path.join(dir_server,'expression_project','experiments/'+experiment_name);
    # dir_files='../data/ck_96/train_test_files';
    # num_folds=10;
    # epoch_range=list(range(100,1100,100));

    experiment_name='test_mean_blur';
    out_dir_meta=os.path.join(dir_server,'expression_project','scratch/'+experiment_name);
    dir_files='../data/ck_96/train_test_files';
    num_folds=1;
    epoch_range=list(range(100,200,100));


    expressions=[None]
    # range(1,9);
    right=True;

    npy_file_pred='1_pred_labels.npy';
    npy_file_gt='1_gt_labels.npy';
    posts=['_org','_gb_gcam','_gb_gcam_pred','_hm','_hm_pred','_gb_gcam_org','_gb_gcam_org_pred',
            '_gb_gcam_th','_gb_gcam_th_pred',
            '_blur','_blur_pred']
    
    # num_folds=1;
    # im_posts=[pre+exp_curr+'.jpg' for pre in pres for exp_curr in expressions]; 
    

    
    batchSizeTest=128;
    for expression_curr in expressions:
    
        out_file_html=os.path.join(out_dir_meta,str(expression_curr)+'.html');
        ims_all=[];
        captions_all=[];
        for fold_num in range(num_folds):    
            data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
            gt_labels_file=os.path.join(out_dir_meta,str(epoch_range[0]),str(fold_num),'test_images',npy_file_gt);
            gt_labels=np.load(gt_labels_file);    
            num_batches=int(math.ceil(len(gt_labels)/128.0))
            im_pre=np.array([str(batch_num)+'_'+str(im_num) for batch_num in range(1,num_batches+1) for im_num in range(1,batchSizeTest+1)]);
            if expression_curr is not None:
                bin_keep=gt_labels==expression_curr;
            else:
                bin_keep=gt_labels==gt_labels
            gt_rel=gt_labels[bin_keep]
            im_pre_rel=im_pre[bin_keep];

            dir_ims=[os.path.join(out_dir_meta,str(epoch_num),str(fold_num),'test_images') for epoch_num in epoch_range];
            pred_rels=[np.load(os.path.join(dir_curr,npy_file_pred))[bin_keep] for dir_curr in dir_ims];

            for idx_im_pre_curr,im_pre_curr in enumerate(im_pre_rel):
                for idx_epoch in range(len(dir_ims)):
                    im_row=[util.getRelPath(os.path.join(dir_ims[idx_epoch],im_pre_curr+post_curr+'.jpg'),dir_server) for post_curr in posts];
                    caption_pre='right' if pred_rels[idx_epoch][idx_im_pre_curr]==gt_rel[idx_im_pre_curr] else 'wrong'
                    caption_row=[str(idx_im_pre_curr)+' '+caption_pre+' '+str(epoch_range[idx_epoch])+' '+str(int(gt_rel[idx_im_pre_curr]))+' '+str(int(pred_rels[idx_epoch][idx_im_pre_curr]))]*len(im_row);
                    ims_all.append(im_row[:]);
                    captions_all.append(caption_row[:])

        visualize.writeHTML(out_file_html,ims_all,captions_all,100,100);
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




def writeScriptSchemesFixThresh():

    # dir_exp_old='../experiments/khorrami_ck_rerun';
    # experiment_name='ck_meanBlur_fixThresh_100_inc';
    # out_dir_meta_meta='../experiments/'+experiment_name;
    # util.mkdir(out_dir_meta_meta);
    # num_scripts=1;
    # model_file='../models/base_khorrami_model_8.dat'
    # dir_files='../data/ck_96/train_test_files';    
    # folds_range=[6];
    
    dir_files='../data/tfd/train_test_files';
    dir_exp_old='../experiments/noBlur_meanFirst_pixel_augment/noBlur_meanFirst_7out';
    folds_range=[4];
    model_file='../models/base_khorrami_model_7.dat'
    experiment_name='tfd_meanBlur_fixThresh_100_inc';
    num_scripts=1;
    

    startingActivation=0.05;
    fixThresh=0.05;
    activationThreshMax=0.5;
    schemes=['ncl','mixcl','mix'];
    # path_to_th='train_khorrami_withBlur.th';withAnno=False;
    path_to_th='test_localization.th';withAnno=True;
    

    epoch_starts=range(100,600,100);
    out_dir_meta_meta='../experiments/'+experiment_name;
    util.mkdir(out_dir_meta_meta);
    out_script='../scripts/train_'+experiment_name+'_loc';
    
    # batchSizeTest=128;
    batchSize=128;
    ratioBlur=0.5;

    commands=[];
    for fold_num in folds_range:
        data_file=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
        num_lines=len(util.readLinesFromFile(data_file));
        epoch_size=num_lines/batchSize;
        for scheme in schemes:
            out_dir_meta=os.path.join(out_dir_meta_meta,scheme);
            util.mkdir(out_dir_meta);
            for epoch_start in epoch_starts: 
                out_dir_meta_curr=os.path.join(out_dir_meta,str(epoch_start));
                util.mkdir(out_dir_meta_curr);
                if scheme=='mix':
                    if 1000%epoch_start==0:
                        activationThreshMax=fixThresh*(1000/epoch_start-1);
                    else:
                        activationThreshMax=fixThresh*math.floor(1000/epoch_start);
                    print activationThreshMax
                    command=writeBlurScript(path_to_th,out_dir_meta_curr,dir_files,fold_num,model_file=model_file,
                                            batchSize=batchSize,
                                            scheme=scheme,
                                            ratioBlur=ratioBlur,
                                            incrementDifficultyAfter=0,
                                            startingActivation=0,
                                            fixThresh=fixThresh,
                                            activationThreshMax=activationThreshMax,
                                            iterations=1000,withAnno=withAnno)
                else:
                    resume_model=os.path.join(dir_exp_old,str(fold_num),'intermediate','model_all_'+str(epoch_start*epoch_size)+'.dat');
                    assert(os.path.exists(resume_model));
                    command=writeBlurScript(path_to_th,out_dir_meta_curr,dir_files,fold_num,
                                            batchSize=batchSize,
                                            model_file=resume_model,
                                            scheme=scheme,
                                            ratioBlur=ratioBlur,
                                            incrementDifficultyAfter=epoch_start,
                                            startingActivation=startingActivation,
                                            fixThresh=fixThresh,
                                            activationThreshMax=activationThreshMax,
                                            iterations=1000-epoch_start,withAnno=withAnno)
                
                commands.append(command);

    len(commands)
    print commands[0];
    print commands[-1];
    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_script+'_'+str(idx_commands)+'.sh';
        print idx_commands,out_file_script_curr,len(commands)
        # print commands;
        util.writeFile(out_file_script_curr,commands);



def writeScriptSchemesAutoThresh():

    dir_exp_old='../experiments/khorrami_ck_rerun';
    dir_files='../data/ck_96/train_test_files';
    experiment_name='ck_meanBlur_autoThresh_100_inc';
    folds_range=[6];

    # dir_exp_old='../experiments/noBlur_meanFirst_pixel_augment/noBlur_meanFirst_7out';
    # dir_files='../data/tfd/train_test_files';
    # experiment_name='tfd_meanBlur_autoThresh_100_inc';
    # folds_range=[4];

    out_dir_meta_meta='../experiments/'+experiment_name;
    util.mkdir(out_dir_meta_meta);
    out_script='../scripts/train_'+experiment_name+'_loc';
    num_scripts=1;
    

    util.mkdir(out_dir_meta_meta);
    
    epoch_starts=[200,500,100,400];
    # range(100,600,100);
    startingActivation=0.05;
    fixThresh=0.05;
    activationThreshMax=0.15;
    total_epochs=1000;
    schemes=['ncl','mixcl']
    # ['ncl','mixcl']
    # epoch_starts=[[100,200,300,400,500],[100,200,300,400,500]];
    # ,'mix'];
    # schemes=['mix'];
    path_to_th='train_khorrami_withBlur.th';withAnno=False;
    path_to_th='test_localization.th';withAnno=True;

    
    
    # batchSizeTest=128;
    batchSize=128;
    ratioBlur=0.5;

    commands=[];
    for fold_num in folds_range:
        data_file=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
        num_lines=len(util.readLinesFromFile(data_file));
        epoch_size=num_lines/batchSize;
        for idx_scheme,scheme in enumerate(schemes):
            out_dir_meta=os.path.join(out_dir_meta_meta,scheme);
            util.mkdir(out_dir_meta);
            for epoch_start in epoch_starts:
            # epoch_starts[idx_scheme]: 
                out_dir_meta_curr=os.path.join(out_dir_meta,str(epoch_start));
                util.mkdir(out_dir_meta_curr);
                # if scheme=='mix':
                #     if 1000%epoch_start==0:
                #         activationThreshMax=fixThresh*(1000/epoch_start-1);
                #     else:
                #         activationThreshMax=fixThresh*math.floor(1000/epoch_start);
                #     print activationThreshMax
                #     command=writeBlurScript(path_to_th,out_dir_meta_curr,dir_files,fold_num,
                #                             batchSize=batchSize,
                #                             scheme=scheme,
                #                             ratioBlur=ratioBlur,
                #                             incrementDifficultyAfter=0,
                #                             startingActivation=0,
                #                             fixThresh=fixThresh,
                #                             activationThreshMax=activationThreshMax,
                #                             iterations=1000)
                # else:
                resume_model=os.path.join(dir_exp_old,str(fold_num),'intermediate','model_all_'+str(epoch_start*epoch_size)+'.dat');
                assert(os.path.exists(resume_model));
                if total_epochs%epoch_start==0:
                    num_inc=total_epochs/epoch_start -1;
                else:
                    num_inc=math.floor(total_epochs/epoch_start);
                startingActivation=activationThreshMax/num_inc;
                print startingActivation,num_inc,startingActivation*num_inc

                command=writeBlurScript(path_to_th,out_dir_meta_curr,dir_files,fold_num,
                                        batchSize=batchSize,
                                        model_file=resume_model,
                                        scheme=scheme,
                                        ratioBlur=ratioBlur,
                                        incrementDifficultyAfter=epoch_start,
                                        startingActivation=startingActivation,
                                        fixThresh=0,
                                        activationThreshMax=activationThreshMax,
                                        iterations=total_epochs-epoch_start,withAnno=withAnno)
                
                commands.append(command);

    len(commands)
    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_script+'_'+str(idx_commands)+'.sh';
        print idx_commands,out_file_script_curr,len(commands)
        # print commands;
        util.writeFile(out_file_script_curr,commands);


def createComparativeHtml():
    # experiment_name='test_mean_blur';
    # out_dir_meta_meta=os.path.join(dir_server,'expression_project','experiments');
    # out_dir_meta=os.path.join(out_dir_meta_meta,'ck_meanBlur_fixThresh_100_inc')
    # out_dir_baseline=os.path.join(out_dir_meta_meta,'khorrami_ck_rerun');
    # dir_files='../data/ck_96/train_test_files';
    # dir_caption=['ncl','mixcl','mix','bl']
    # dir_comps=[os.path.join(out_dir_meta,dir_caption_curr,'200') for dir_caption_curr in dir_caption[:3]]+[out_dir_baseline];
    # range_folds=[6];

    out_dir_meta=os.path.join(dir_server,'expression_project','experiments');
    # out_dir_meta=os.path.join(out_dir_meta_meta,'ck_meanBlur_fixThresh_100_inc')
    # out_dir_baseline=os.path.join(out_dir_meta_meta,'khorrami_ck_rerun');
    # dir_files='../data/karina_vids/train_test_files_deviceInstalled_01_192';
    # dir_caption=['last','last_self','higher','higher_self','slr_self']
    # dir_comps=['horses_twoClass_01_ft_onlyLast_192_dI',
    #             'horses_twoClass_01_ft_onlyLast_192_dI_selfMean',
    #             'horses_twoClass_01_ft_higherLast_192_dI',
    #             'horses_twoClass_01_ft_higherLast_192_dI_selfMean',
    #             # 'horses_twoClass_01_ft_slr_192_dI',
    #             'horses_twoClass_01_ft_slr_192_dI_selfMean']
    
    # dir_caption=['slr_self']
    # dir_comps=[ 'horses_twoClass_01_ft_slr_192_dI_selfMean']

    # dir_files='../data/karina_vids/train_test_files_deviceInstalled_01_192_mask';
    # experiment_pre='horses_twoClass_01_ft_192_dI_mask_'
    # # experiment_name,mean_im_path,std_im_path,lower,onlyLast
    # dir_caption=['onlyLast_selfMean', 
    #             'onlyLast', 
    #             'higherLast_selfMean', 
    #             'higherLast', 
    #             'slr_selfMean', 
    #             'slr']
    
    # dir_comps=[experiment_pre+'onlyLast_selfMean', 
    #             experiment_pre+'onlyLast', 
    #             experiment_pre+'higherLast_selfMean', 
    #             experiment_pre+'higherLast', 
    #             experiment_pre+'slr_selfMean', 
    #             experiment_pre+'slr']
    
    

    # dir_comps=[os.path.join(out_dir_meta,dir_curr) for dir_curr in dir_comps]
    # range_folds=[0];


    dir_files='../data/office/domain_adaptation_images_256/train_test_files';
    experiment_pre='office_bl_amazon_less_256'
    # experiment_name,mean_im_path,std_im_path,lower,onlyLast
    dir_caption=['0']
    
    dir_comps=[experiment_pre]
    
    

    dir_comps=[os.path.join(out_dir_meta,dir_curr) for dir_curr in dir_comps]
    range_folds=['amazon_0'];


    out_file_html=os.path.join(dir_comps[0],'comparison.html');
    
    expressions=[None]
    right=True;

    npy_file_pred='1_pred_labels.npy';
    npy_file_gt='1_gt_labels.npy';
    posts=['_org','_gb_gcam','_gb_gcam_pred','_gb_gcam_org','_gb_gcam_org_pred','_gb_gcam_th','_gb_gcam_th_pred','_blur','_blur_pred']
    
    batchSizeTest=128;
    for expression_curr in expressions:
        if out_file_html is None:
            out_file_html=os.path.join(out_dir_meta,str(expression_curr)+'.html');
        ims_all=[];
        captions_all=[];
        for fold_num in range_folds:    
            data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
            gt_labels_file=os.path.join(dir_comps[0],str(fold_num),'test_images_blur',npy_file_gt);
            gt_labels=np.load(gt_labels_file);    
            num_batches=int(math.ceil(len(gt_labels)/float(batchSizeTest)))
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

            dir_ims=[os.path.join(dir_curr,str(fold_num),'test_images_blur') for dir_curr in dir_comps];
            pred_rels=[np.load(os.path.join(dir_curr,npy_file_pred))[bin_keep] for dir_curr in dir_ims];

            for idx_im_pre_curr,im_pre_curr in enumerate(im_pre_rel):
                for idx_epoch in range(len(dir_ims)):
                    im_row=[util.getRelPath(os.path.join(dir_ims[idx_epoch],im_pre_curr+post_curr+'.jpg'),dir_server) for post_curr in posts];
                    caption_pre='right' if pred_rels[idx_epoch][idx_im_pre_curr]==gt_rel[idx_im_pre_curr] else 'wrong'
                    caption_row=[str(idx_im_pre_curr)+' '+caption_pre+' '+dir_caption[idx_epoch]+' '+str(int(gt_rel[idx_im_pre_curr]))+' '+str(int(pred_rels[idx_epoch][idx_im_pre_curr]))]*len(im_row);
                    ims_all.append(im_row[:]);
                    captions_all.append(caption_row[:])

        visualize.writeHTML(out_file_html,ims_all,captions_all,192,192);
        print out_file_html.replace(dir_server,click_str);

def printTestAccuracy():
    dir_meta='../experiments/ck_meanBlur_autoThresh_100_inc';
    # schemes=['mixcl','ncl','mix'];
    schemes=['mixcl','ncl'];
    # ,'mix'];
    inc_range=range(100,600,100);
    num_fold=6;

    # dir_meta='../experiments/tfd_meanBlur_fixThresh_100_inc';
    # schemes=['mixcl','ncl','mix'];
    # inc_range=range(100,600,100);
    # dir_meta='../experiments/tfd_meanBlur_autoThresh_100_inc';
    # schemes=['mixcl','ncl'];
    # inc_range=[100,200,400,500];
    # schemes=['mix'];
    
    # num_fold=4;

    for scheme in schemes:
        for inc in inc_range:
            file_curr=os.path.join(dir_meta,scheme,str(inc),str(num_fold),'test_images','log_test.txt');
            lines=util.readLinesFromFile(file_curr);
            accu=lines[-1].split(':')[-1];

            file_log=os.path.join(dir_meta,scheme,str(inc),str(num_fold),'intermediate','log.txt');
            log_lines=util.readLinesFromFile(file_log)
            # print log_lines[-2].split('at: ')[1][:6];
            
            # print file_curr
            print '\t'.join([scheme,str(inc),'0.2',accu[1:6]]);
            # print '\t'.join([scheme,str(inc),log_lines[-2].split('at: ')[1][:6],accu[1:6]]);
            # print lines[-1];

    # return
    # # writeScriptSchemesFixThresh()
    # out_dir_meta='../experiments/khorrami_ck_rerun'
    # num_folds=10;
    # for fold_num in range(num_folds):
    #     test_file=os.path.join(out_dir_meta,str(fold_num),'test_images','log_test.txt');
    #     print test_file
    #     lines=util.readLinesFromFile(test_file);
    #     print fold_num
    #     print lines[-1];


def createComparativeHTMLs_diffTypes_1():
    experiment_name='scheme_200_inc';
    out_dir_meta_meta=os.path.join(dir_server,'expression_project','experiments',experiment_name);
    dir_files='../data/ck_96/train_test_files';
    num_folds=1;
    epoch_range=list(range(100,900,100));
    epoch_range=[100,800];


    expressions=[None]
    range_folds=[6];
    # range(1,9);
    right=True;

    npy_file_pred='1_pred_labels.npy';
    npy_file_gt='1_gt_labels.npy';
    schemes=['ncl','mixcl','mix','bl']
    posts=['_org','_gb_gcam','_gb_gcam_pred','_hm','_hm_pred','_gb_gcam_org','_gb_gcam_org_pred']
    # ,
    #         '_gb_gcam_th','_gb_gcam_th_pred',
    #         '_blur','_blur_pred']
    
    # num_folds=1;
    # im_posts=[pre+exp_curr+'.jpg' for pre in pres for exp_curr in expressions]; 
    

    
    batchSizeTest=64;
    for scheme_curr in schemes:
        out_dir_meta=os.path.join(out_dir_meta_meta,scheme_curr);
        for expression_curr in expressions:
        
            out_file_html=os.path.join(out_dir_meta,str(expression_curr)+'.html');
            ims_all=[];
            captions_all=[];
            for fold_num in range_folds:    
                data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
                gt_labels_file=os.path.join(out_dir_meta,str(epoch_range[0]),str(fold_num),'test_images',npy_file_gt);
                gt_labels=np.load(gt_labels_file);    
                num_batches=int(math.ceil(len(gt_labels)/float(batchSizeTest)))
                im_pre=np.array([str(batch_num)+'_'+str(im_num) for batch_num in range(1,num_batches+1) for im_num in range(1,batchSizeTest+1)]);
                if expression_curr is not None:
                    bin_keep=gt_labels==expression_curr;
                else:
                    bin_keep=gt_labels==gt_labels
                gt_rel=gt_labels[bin_keep]
                im_pre_rel=im_pre[bin_keep];

                dir_ims=[os.path.join(out_dir_meta,str(epoch_num),str(fold_num),'test_images') for epoch_num in epoch_range];
                pred_rels=[np.load(os.path.join(dir_curr,npy_file_pred))[bin_keep] for dir_curr in dir_ims];

                for idx_im_pre_curr,im_pre_curr in enumerate(im_pre_rel):
                    for idx_epoch in range(len(dir_ims)):
                        im_row=[util.getRelPath(os.path.join(dir_ims[idx_epoch],im_pre_curr+post_curr+'.jpg'),dir_server) for post_curr in posts];
                        caption_pre='right' if pred_rels[idx_epoch][idx_im_pre_curr]==gt_rel[idx_im_pre_curr] else 'wrong'
                        caption_row=[str(idx_im_pre_curr)+' '+caption_pre+' '+str(epoch_range[idx_epoch])+' '+str(int(gt_rel[idx_im_pre_curr]))+' '+str(int(pred_rels[idx_epoch][idx_im_pre_curr]))]*len(im_row);
                        ims_all.append(im_row[:]);
                        captions_all.append(caption_row[:])

            visualize.writeHTML(out_file_html,ims_all,captions_all,100,100);
            print out_file_html.replace(dir_server,click_str);

def createComparativeHTMLs_diffTypes_2():
    experiment_name='scheme_200_inc';
    # out_dir_meta='../experiments/'+experiment_name;
    out_dir_meta='../experiments/'+experiment_name;
    util.mkdir(out_dir_meta);
    out_script='../scripts/test_'+experiment_name;
    # +'.sh';
    # path_to_th='train_khorrami_basic.th';
    path_to_th='train_khorrami_withBlur.th';

    dir_files='../data/ck_96/train_test_files';
    # resume_dir_meta='../experiments/khorrami_basic_aug_fix_resume'

    out_dir_meta_scheme=os.path.join(dir_server,'expression_project','experiments','ck_meanBlur_fixThresh_100_inc')
    out_dir_baseline=os.path.join(dir_server,'expression_project','experiments','khorrami_ck_rerun');
    schemes=['ncl','mixcl','mix','bl']
    scheme_dirs=[os.path.join(out_dir_meta_scheme,dir_caption_curr,'200') for dir_caption_curr in schemes[:-1]]+[out_dir_baseline];
    


    range_folds=[6];

    resume_dir_meta=None
    noBias=False;
    learningRate = 0.01;
    batchSizeTest=128;
    batchSize=128;
    ratioBlur=0.0;

    out_dir_model='../experiments/khorrami_ck_rerun';
    
    # print '{',
    # out_script_curr=out_script+'_'+str(fold_num)+'.sh';

    commands=[];
    for idx_scheme,out_dir_scheme in enumerate(scheme_dirs):
        for fold_num in range_folds:
            for epoch_num in range(100,900,100):
                out_dir_meta_curr=os.path.join(out_dir_meta,schemes[idx_scheme],str(epoch_num));
                util.makedirs(out_dir_meta_curr);
                out_dir_model_curr=os.path.join(out_dir_scheme,str(fold_num));
                data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
                epoch_size=len(util.readLinesFromFile(data_path))/batchSize;
                if schemes[idx_scheme]=='mix' or schemes[idx_scheme]=='ncl':
                    inc_num=epoch_size*(epoch_num+200);
                else:
                    inc_num=epoch_size*epoch_num;
                modelTest=os.path.join(out_dir_model_curr,'intermediate','model_all_'+str(inc_num)+'.dat');
                print modelTest
                assert os.path.exists(modelTest);
                command=writeBlurScript(path_to_th,out_dir_meta_curr,dir_files,fold_num,modelTest=modelTest,batchSizeTest=64);
                # print command;
                commands.append(command);
                # raw_input();
    num_scripts=1;
    # print out_script+'.sh'
    # print len(commands);
    # util.writeFile(out_script+'.sh',commands);

    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        # print commands;
        util.writeFile(out_file_script_curr,commands);


def writeScriptTFDTestsDiffSchemes():
    out_dir_meta_meta='../experiments';
    # out_dir_metas=[os.path.join(out_dir_meta_meta,dir_curr) for dir_curr in ['ck_meanBlur_fixThresh_100_inc','ck_meanBlur_autoThresh_100_inc']];
    # schemes=['mixcl','ncl'];
    # incs=[str(num) for num in range(100,600,100)];
    # meta_dirs=['ck_meanBlur_autoThresh_100_inc']
    # out_results=os.path.join(out_dir_meta_meta,'ck_testing_tfd_4_auto');
    # schemes=['ncl','mixcl'];
    # num_scripts=1;

    meta_dirs=['ck_meanBlur_fixThresh_100_inc']
    out_results=os.path.join(out_dir_meta_meta,'ck_testing_tfd_4_fix');

    # incs=[str(num) for num in range(100,600,100)];
    # schemes=['ncl','mixcl','mix'];
    incs=['None'];
    schemes=['bl'];
    data_dir='../data/tfd/train_test_files';
    
    # test_file='../data/tfd/train_test_files/test_ckLabels_4.txt';
    test_file='../data/tfd/train_test_files/test_ckLabels_4_withAnno.txt';

    num_fold=6;    
    out_script='../scripts/test_tfd_ck_scheme';

    # incs=[str(num) for num in [100,200,400,500]];
    # meta_dirs=['tfd_meanBlur_autoThresh_100_inc']
    # out_results=os.path.join(out_dir_meta_meta,'tfd_testing_ck_all_auto');
    # schemes=['ncl','mixcl'];

    # incs=[str(num) for num in range(100,600,100)];
    # meta_dirs=['tfd_meanBlur_fixThresh_100_inc']
    # out_results=os.path.join(out_dir_meta_meta,'tfd_testing_ck_all_fix');
    # schemes=['ncl','mixcl','mix'];

    # incs=['None'];
    # meta_dirs=['tfd_meanBlur_fixThresh_100_inc']
    # out_results=os.path.join(out_dir_meta_meta,'tfd_testing_ck_all_fix');
    # schemes=['bl'];
    
    num_scripts=1;

    # data_dir='../data/ck_96/train_test_files';
    # num_fold=4;    
    # test_file=os.path.join(data_dir,'test_tfdLabels_all_withAnno.txt');
    # out_script='../scripts/test_ck_tfd_scheme';



    out_dir_metas=[os.path.join(out_dir_meta_meta,dir_curr) for dir_curr in meta_dirs];
    
    # path_to_th='train_khorrami_withBlur.th';
    path_to_th='test_localization.th';

    models=[os.path.join(out_dir_meta_curr,scheme_curr,str(num_fold),'final','model_all_final.dat') for out_dir_meta_curr in out_dir_metas for scheme_curr in schemes];
    
    
    util.mkdir(out_results);
    
    commands=[];
    for out_dir_meta in out_dir_metas:
        for scheme in schemes:
            for epoch_num in incs:
                out_dir_curr=os.path.join(out_results,scheme,epoch_num);
                util.makedirs(out_dir_curr)
                if epoch_num is not 'None':
                    model_curr=os.path.join(out_dir_meta,scheme,epoch_num,str(num_fold),'final','model_all_final.dat');
                else:
                    model_curr=os.path.join(out_dir_meta,scheme,str(num_fold),'final','model_all_final.dat');
                print model_curr
                assert (os.path.exists(model_curr))
                command_curr=writeBlurScript(path_to_th,out_dir_curr,data_dir,0,val_data_path=test_file,modelTest=model_curr);
                commands.append(command_curr);

    commands=np.array(commands);
    print commands.shape
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_script+'_'+str(idx_commands)+'.sh';
        print idx_commands
        print out_file_script_curr
        # print commands;
        util.writeFile(out_file_script_curr,commands);



def printAccuResultsGeneralize():
    dir_meta_meta='../experiments'
    dir_metas=['ck_testing_tfd_4_auto','ck_testing_tfd_4','ck_testing_tfd_4'];
    schemes=[['mixcl','ncl'],['mixcl','ncl','mix'],['bl']]
    inc_ranges=[range(100,600,100),range(100,600,100),['None']]
    
    num_fold=0;
    for dir_meta_curr,schemes,inc_range in zip(dir_metas,schemes,inc_ranges):
        dir_meta=os.path.join(dir_meta_meta,dir_meta_curr);
        print dir_meta;
        for scheme in schemes:
            for inc in inc_range:
                file_curr=os.path.join(dir_meta,scheme,str(inc),str(num_fold),'test_images','log_test.txt');
                lines=util.readLinesFromFile(file_curr);
                accu=lines[-1].split(':')[-1];

                print '\t'.join([scheme,str(inc),'x',accu[1:6]]);

def main():
    
    

    createComparativeHtml();
    # writeScriptTFDTestsDiffSchemes();
    # writeScriptSchemesAutoThresh();
    # writeScriptTFDTestsDiffSchemes();
    # writeScriptSchemesFixThresh();
    return
    num_folds=10;
    dir_files='../data/ck_192/train_test_files';
    input_size=192;
    experiment_name='khorrami_large_ck'
    path_to_th='train_khorrami_withBlur.th';
    model_file='../models/khorrami_bigger_8.dat'
    num_scripts=2;

    out_dir_meta='../experiments/'+experiment_name;
    out_script=os.path.join('../scripts','train_'+experiment_name);
    util.mkdir(out_dir_meta);
    
    commands=[];
    for fold_num in range(num_folds):
        command=writeBlurScript(path_to_th,out_dir_meta,dir_files,fold_num,input_size=input_size,model_file=model_file);
        commands.append(command)

    len(commands)
    commands=np.array(commands);
    commands_split=np.array_split(commands,num_scripts);
    for idx_commands,commands in enumerate(commands_split):
        out_file_script_curr=out_script+'_'+str(idx_commands)+'.sh';
        print idx_commands,out_file_script_curr,len(commands)
        # print commands;
        util.writeFile(out_file_script_curr,commands);
    
    # writeScriptSchemesAutoThresh();
    # printTestAccuracy();

    # writeScriptSchemesFixThresh()
    # writeScriptTFDTestsDiffSchemes()
    # writeScriptSchemesAutoThresh()
    # writeTFDSchemeScripts();
    # writeHTMLs_viz_inc();
    # writeCKScripts_viz_inc();
    # script_vizTestGradCam()
    # writeCKScripts()
    # script_vizCompareActivations()
    # writeTFDSchemeScripts();
    # writeCKScripts();
    # writeTestGradCamScript()
    # script_vizTestGradCam();

if __name__=='__main__':
    main();