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
import itertools;
from sklearn.manifold import TSNE
import time;

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
    # domain_train='amazon';
    # domain_test='webcam';
    num_folds=5;
    path_to_th='train_alexnet_withBlur.th';
    num_scripts=1;
    dir_files='../data/office/domain_adaptation_images_256/train_test_files'
    model_file='../models/alexnet_31.dat';
    learningRate=0.001;
    testAfter=1;
    saveAfter=5;
    iterations=50;
    lower=True;
    
    for domain_train in ['webcam','dslr']:
        experiment_name='office_bl_'+domain_train+'_less_256';
        out_dir_meta=os.path.join('../experiments',experiment_name);
        out_script=os.path.join('../scripts','train_'+experiment_name);
        util.mkdir(out_dir_meta);
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


def writeSchemeScripts():
    # domain_train='dslr';
    domain_train='amazon';
    domain_test='amazon';

    num_folds=1;
    path_to_th='train_alexnet_withBlur.th';

    # autoThresh=True; 
    # for activationThreshMax in [0.15,0.5]:
    #     schemes=['mixcl','ncl']
    dir_files='../data/office/domain_adaptation_images_256/train_test_files'
    model_file='../models/alexnet_31.dat';
    num_scripts=1;
    learningRate=0.001;
    testAfter=1;
    saveAfter=5;
    epoch_total=50;
    lower=True;
    weights=True;
    mean_im_path='';
    std_im_path='';
    modelTest=None;

    out_file_script_meta=os.path.join('../scripts',domain_train+'_'+domain_test+'_fc6.sh')
    commands_all=[];
    # autoThresh_all=[False,False,True,True];
    # activationThreshMax_all=[0,0,0.15,0.5];
    # epoch_starts_all=[range(5,30,5),[None],range(5,30,5),range(5,30,5)];
    # schemes_all=[['mixcl','ncl','mix'],['bl'],['mixcl','ncl'],['mixcl','ncl']]
    # autoThresh_all=[True];
    # activationThreshMax_all=[0.15];
    # epoch_starts_all=[range(5,30,5)];
    # schemes_all=[['mixcl','ncl']]

    autoThresh_all=[False,True];
    activationThreshMax_all=[0,0.5];
    epoch_starts_all=[[None],[5]];
    schemes_all=[['bl'],['mixcl']]

    for autoThresh,activationThreshMax,epoch_starts,schemes in zip(autoThresh_all,activationThreshMax_all,epoch_starts_all,schemes_all):
        # activationThreshMax=0.5
        # schemes=['mixcl','ncl','mix']
        # autoThresh=False
        dir_exp_old='../experiments/office_bl_'+domain_train+'_less_256';
        folds_range=[domain_train+'_'+str(num_fold) for num_fold in range(num_folds)];

        val_data_path=None;
        
        outDirTest=None;

        if autoThresh:
            experiment_name='office_autoThresh_'+domain_train+'_less_256_'+str(int(activationThreshMax*100));
        else:
            experiment_name='office_fixThresh_'+domain_train+'_less_256';
        if domain_train!=domain_test:
            # experiment_name=experiment_name+'_test_'+domain_test;
            # modelTest=os.path.join(dir_exp_old,domain_train+'_0','final','model_all_final.dat');
            val_data_path=os.path.join(dir_files,'test_'+domain_test+'_all.txt')
            outDirTest='test_images_'+domain_test
        

        # modelTest=os.path.join(dir_exp_old,'amazon_0','final','model_all_final.dat');
        # val_data_path=os.path.join(dir_files,'test_'+domain_test+'_all.txt')
        # outDirTest='test_images_'+domain_test;
        # epoch_starts=[None]
        # schemes=['bl']
        # val_data_path=val_data_path,
        # modelTest=modelTest,
        # )

        commands_curr=scripts_and_viz.writeSchemeScripts_fixed(path_to_th,
                        dir_files,
                        dir_exp_old,
                        folds_range,
                        model_file,
                        experiment_name,
                        epoch_starts=epoch_starts,
                        epoch_total=epoch_total,
                        num_scripts=num_scripts,
                        learningRate=learningRate,
                        saveAfter=saveAfter,
                        testAfter=testAfter,
                        mean_im_path=mean_im_path,
                        std_im_path=std_im_path,
                        lower=lower,
                        weights=weights,
                        schemes=schemes,
                        autoThresh=autoThresh,
                        val_data_path=val_data_path,
                        modelTest=modelTest,
                        activationThreshMax=activationThreshMax,
                        outDirTest=outDirTest
                        )
        # print commands_curr
        commands_all=commands_all+commands_curr;

    if out_file_script_meta is not None:
        util.writeFile(out_file_script_meta,commands_all);
        print len(commands_all),out_file_script_meta


def printTestResults():
    domain_train='amazon'
    domain_test='dslr'
    dir_meta_meta='../experiments'

    autoThresh=[0,0,0.5,0.15];
    schemes=[['mixcl','ncl','mix'],['bl'],['mixcl','ncl'],['mixcl','ncl']]
    inc_ranges=[range(5,30,5),['None'],range(5,30,5),range(5,30,5)];

    # autoThresh=[0.5,0.15];
    # schemes=[['mixcl','ncl'],['mixcl','ncl']]
    # inc_ranges=[range(5,30,5),range(5,30,5)];
    # test_dir='test_images';
    range_folds=[0];
    print 'train',domain_train,'test',domain_test
    # print '\t'.join(['scheme','increment','max_blur','accuracy'])
    for autoThresh_curr,schemes_curr,inc_ranges_curr in zip(autoThresh,schemes,inc_ranges):
        post = 'autoThresh' if autoThresh_curr>0 else 'fixThresh';
        dir_meta_curr=['office'];
        dir_meta_curr.append('autoThresh' if autoThresh_curr else 'fixThresh');
        dir_meta_curr.append(domain_train);
        dir_meta_curr.extend(['less','256']);
        if autoThresh_curr>0:
            dir_meta_curr.append(str(int(autoThresh_curr*100)));
        if domain_train!=domain_test:
            test_dir='test_images_'+domain_test;
            # dir_meta_curr.extend(['test',domain_test]);
        print dir_meta_curr
        dir_meta_curr='_'.join(dir_meta_curr)
        dir_meta=os.path.join(dir_meta_meta,dir_meta_curr);
        posts=itertools.product(schemes_curr,inc_ranges_curr,range_folds);
        for post in posts:
            # print post
            dir_curr=os.path.join(dir_meta,str(post[0]),str(post[1]),domain_train+'_'+str(post[2]))
            file_curr=os.path.join(dir_curr,test_dir,'log_test.txt');
            thresh_curr='x';
            
            # thresh_file=os.path.join(dir_curr,'intermediate','log.txt');
            # if os.path.exists(thresh_file):
            #     thresh_curr=util.readLinesFromFile(thresh_file)[-2];
            #     print thresh_curr;
            #     thresh_curr=thresh_curr.split(':')[2];
            #     thresh_curr=thresh_curr[:thresh_curr.index(',')].strip()[:4]
                # print thresh_curr;

            assert os.path.exists(file_curr);

        # for scheme_curr in schemes_curr:
        #     for inc_range in inc_ranges_curr:
        #         for num_fold in range_folds:
        # for dir_meta_curr,scheme,inc_range in zip(dir_metas,schemes,inc_ranges):
                    
    #         num_fold=dir_meta_curr.split('_')[2]+'_'+str(num_fold);    
    #         file_curr=os.path.join(dir_meta,str(num_fold),'test_images'+domain_test_post,'log_test.txt');
            lines=util.readLinesFromFile(file_curr);
            accu=lines[-3].split(':')[-1];
            print '\t'.join([post[0],str(post[1]),thresh_curr,accu[1:6]]);


def createCondensedFiles(fc_file,test_file,num_to_keep):
    fc=np.load(fc_file);
    lines=util.readLinesFromFile(test_file);

    cats=[int(line_curr.split(' ')[1]) for line_curr in lines];
    cats=np.array(cats);

    print fc.shape,len(lines);
    idx_keep=[];

    for cat_num in np.unique(cats):
        idx=np.where(cats==cat_num)[0];
        np.random.shuffle(idx);
        idx=idx[:num_to_keep];
        # print idx.shape;
        idx_keep.extend(list(idx));
        print cat_num,np.sum(cats==cat_num),len(idx),len(idx_keep);


    idx_keep=np.array(idx_keep);
    # print idx_keep.shape;
    fc_keep=fc[idx_keep];
    lines_keep=np.array(lines)[idx_keep];
    np.save(fc_file[:fc_file.rindex('.')]+'_condense.npy',fc_keep)
    util.writeFile(test_file[:test_file.rindex('.')]+'_condense.txt',lines_keep);

def saveTSNE(out_file_pre,fc_all,title,colors,legend_entries,markers,split_viz=None,split_title=None):
    model=TSNE(n_components=2,random_state=0);num_tries=1;
        # out_file_curr=os.path.join(out_dir_im,'0'*(3-len(str(num_try)))+str(num_try)+'_'+title+'.jpg');

    fc=np.concatenate(tuple(fc_all),axis=0);
    print fc.shape;

    t=time.time();
    fc_low=model.fit_transform(fc);
    print (time.time()-t);
    
    xAndYs=[];
    start_idx=0;
    for fc_curr in fc_all:
        end_idx=start_idx+fc_curr.shape[0];
        x=fc_low[start_idx:end_idx,0];
        y=fc_low[start_idx:end_idx,1];
        start_idx=end_idx;
        xAndYs.append((x,y));

    if split_viz is None:
        out_file_curr=out_file_pre+'.jpg';
        print out_file_curr
        visualize.plotSimple(xAndYs,out_file_curr,title,'x','y',colors=colors,scatter=True,legend_entries=legend_entries,markers=markers,outside=True);
    else:
        for idx in range(max(split_viz)+1):
            xAndYs_keep=[xAndY for idx_idx,xAndY in enumerate(xAndYs) if split_viz[idx_idx]==idx];
            out_file_curr=out_file_pre+'_'+split_title[idx]+'.jpg';
            colors_keep=[color_curr for idx_idx,color_curr in enumerate(colors) if split_viz[idx_idx]==idx];
            markers_keep=[color_curr for idx_idx,color_curr in enumerate(markers) if split_viz[idx_idx]==idx];
            legend_entries_keep=[color_curr for idx_idx,color_curr in enumerate(legend_entries) if split_viz[idx_idx]==idx];
            visualize.plotSimple(xAndYs_keep,out_file_curr,title,'x','y',colors=colors_keep,scatter=True,legend_entries=legend_entries_keep,markers=markers_keep,outside=True);   


def saveAllVizTsne():
    dir_files='../data/office/domain_adaptation_images_256/train_test_files';
    out_dir_bl='../experiments/office_fixThresh_amazon_less_256/bl/None/amazon_0';
    out_dirs=[out_dir_bl,'../experiments/office_autoThresh_amazon_less_256_50/mixcl/5/amazon_0'];

    np_pre='1_fc7';
    train_domain='amazon'
    test_domains=['amazon','webcam'];

    limit=None;
    colors=['c','m','b','r']
    markers=['o','o','.','.']
    # split_viz=[0,1,0,1];
    split_viz=None;
    schemes=['bl','mixcl']
    classes_to_viz=range(31)
    # +[None];

    out_dir_im='../experiments/tsne_viz'+'_'.join(schemes)+'_'.join(test_domains);

    util.mkdir(out_dir_im);
    
    files_curr=[];
    legend_entries=[];
    test_files=[];
    for test_domain in test_domains:
        for out_dir,scheme in zip(out_dirs,schemes):
            test_folder='test_images_'+test_domain if test_domain!=train_domain else 'test_images';
            test_file_name='test_'+test_domain+'_all.txt' if test_domain!=train_domain else 'test_'+test_domain+'_0_condense.txt';
            test_files.append(os.path.join(dir_files,test_file_name));
            legend_entries.append(scheme+' '+test_domain);
            files_curr.append(os.path.join(out_dir,test_folder,np_pre+'.npy'));
    test_domains=test_domains*len(out_dirs);
    
    # split_title=schemes;
    # for class_to_keep in classes_to_viz:
    #     if class_to_keep is None:
    #         title='all classes';class_to_keep=None;
    #     else:        
    #         title='0'*(2-len(str(class_to_keep)))+str(class_to_keep);        

    fc_all=[];
    split_viz=[];
    title='all classes'
    
    legend_entries_org=legend_entries[:];
    colors_org=colors[:];
    markers_org=markers[:];
    legend_entries=[];colors=[];markers=[];
    for idx_file_curr,(file_curr,test_domain) in enumerate(zip(files_curr,test_domains)):
        fc_org=np.load(file_curr);
            
        for class_to_keep in classes_to_viz:
            # print test_domain,file_curr
            fc=fc_org;
            if limit is not None:
                fc=fc[:limit];
            if class_to_keep is not None:
                classes=[int(line_curr.split(' ')[1]) for line_curr in util.readLinesFromFile(test_files[idx_file_curr])];
                classes=np.array(classes);
                assert len(classes)==len(fc);
                fc=fc[classes==class_to_keep];
                split_viz.append(class_to_keep);
                markers.append(markers_org[idx_file_curr]);
                colors.append(colors_org[idx_file_curr]);
                legend_entries.append(legend_entries_org[idx_file_curr]);


            fc_all.append(fc);
    print len(fc_all),len(split_viz),len(colors),len(legend_entries),len(markers);
            

    out_file_pre=os.path.join(out_dir_im,title.replace(' ','_'));
    print out_file_pre,len(fc_all[0]);
    saveTSNE(out_file_pre,fc_all,title,colors,legend_entries,markers,split_viz=split_viz,split_title=[str(class_to_keep) for class_to_keep in classes_to_viz])
    visualize.writeHTMLForFolder(out_dir_im);


def main():

    saveAllVizTsne()
    return
    dir_files='../data/office/domain_adaptation_images_256/train_test_files';
    
    test_file=os.path.join(dir_files,'test_amazon_0.txt');
    fc_file=os.path.join(out_dir,'test_images','1_fc7.npy');
    num_to_keep=20;
    
    # test_file=os.path.join(dir_files,'test_webcam_all.txt');
    # fc_file=os.path.join(out_dir,'test_images_webcam','1_fc7.npy');
    # test_file=os.path.join(dir_files,'test_dslr_all.txt');
    # fc_file=os.path.join(out_dir,'test_images_dslr','1_fc7.npy');
    
    createCondensedFiles(fc_file,test_file,num_to_keep);

    fc_file=fc_file[:fc_file.rindex('.')]+'_condense.npy';
    print fc_file;
    fc=np.load(fc_file);
    print fc.shape;


    # printTestResults()
    # writeSchemeScripts();
    # writeScriptForBaseline()
    # saveFullTestFiles()
    # saveTrainTestFiles()

    

if __name__=='__main__':
    main();