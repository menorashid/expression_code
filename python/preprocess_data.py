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

def saveCroppedFace(in_file,out_file,im_size=None,classifier_path=None,savegray=True):
    if classifier_path==None:
        classifier_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml';

    img = cv2.imread(in_file);
    gray  =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade  =  cv2.CascadeClassifier(classifier_path)
    faces  =  face_cascade.detectMultiScale(gray)
    if len(faces)==0:
        print 'PROBLEM';
        return in_file;
    else:
        print len(faces),
        sizes=np.array([face_curr[2]*face_curr[3] for face_curr in faces]);
        faces=faces[np.argmax(sizes)];
        print np.max(sizes);

    [x,y,w,h] = faces;
    roi = gray[y:y+h, x:x+w]
    if not savegray:
        roi = img[y:y+h, x:x+w]
    
    if im_size is not None:
        roi=cv2.resize(roi,tuple(im_size));
    cv2.imwrite(out_file,roi)
    

def saveCKresizeImages():
    anno_file='../data/ck_original/anno_all.txt';
    
    dir_meta=os.path.join(dir_server,'expression_project/data/ck_96');
    out_file_html=os.path.join(dir_meta,'check_face.html');
    replace=False
    im_size=[96,96];
    out_dir_meta_meta='../data/ck_'+str(im_size[0])

    out_dir_meta=os.path.join(out_dir_meta_meta,'im');
    old_out_dir_meta='../data/ck_original/cohn-kanade-images';
    out_file_anno=os.path.join(out_dir_meta_meta,'anno_all.txt');

    util.makedirs(out_dir_meta);
    old_anno_data=util.readLinesFromFile(anno_file)
    ims=[line_curr.split(' ')[0] for line_curr in old_anno_data];
    problem_cases=[];
    new_anno_data=[];

    # ims=ims[:10];
    for idx_im_curr,im_curr in enumerate(ims):
        print idx_im_curr,
        out_file_curr=im_curr.replace(old_out_dir_meta,out_dir_meta);
        problem=None;
        if not os.path.exists(out_file_curr) or replace:
            out_dir_curr=os.path.split(out_file_curr)[0];
            util.makedirs(out_dir_curr);
            problem=saveCroppedFace(im_curr,out_file_curr,im_size);

        if problem is not None:
            problem_cases.append(problem);
        else:
            new_anno_data.append(old_anno_data[idx_im_curr].replace(old_out_dir_meta,out_dir_meta));

    print len(problem_cases);
    # new_anno_data=[line_curr.replace(old_out_dir_meta,out_dir_meta) for line_curr in old_anno_data];
    util.writeFile(out_file_anno,new_anno_data);

    ims=np.array([line_curr.split(' ')[0].replace(out_dir_meta_meta,dir_meta) for line_curr in new_anno_data]);
    print ims[0];
    im_dirs=np.array([os.path.split(im_curr)[0] for im_curr in ims]);
    im_files=[];
    captions=[];
    for im_dir in np.unique(im_dirs):
        im_files_curr=[util.getRelPath(im_curr, dir_server) for im_curr in ims[im_dirs==im_dir]];
        captions_curr=[os.path.split(im_curr)[1] for im_curr in im_files_curr];
        im_files.append(im_files_curr);
        captions.append(captions_curr);

    visualize.writeHTML(out_file_html,im_files,captions);
    print out_file_html.replace(dir_server,click_str);



def splitCKNeutralEmotion():
    im_list = '../data/ck_original/cohn-kanade-images/im_list.txt';
    emotion_list = '../data/ck_original/Emotion/emotion_list.txt';
    out_file_anno= '../data/ck_original/anno_all.txt';

    emotion_dir = os.path.split(emotion_list)[0];
    im_dir = os.path.split(im_list)[0];
    
    emotion_files = util.readLinesFromFile(emotion_list);
    emotion_dirs = [os.path.split(line_curr)[0].replace(emotion_dir,im_dir) for line_curr in emotion_files];
    ims = np.array(util.readLinesFromFile(im_list));
    ims_dirs = np.array([os.path.split(line_curr)[0] for line_curr in ims])

    list_ims=[];

    for idx_emotion_dir,emotion_dir_curr in enumerate(emotion_dirs):
        emotion_file_curr = emotion_files[idx_emotion_dir];
        ims_curr = ims[ims_dirs == emotion_dir_curr];
        ims_curr = np.sort(ims_curr);
        
        if len(ims_curr)<4:
            print 'CONTINUING',emotion_dir_curr;
            
        emotion_curr = int(float(util.readLinesFromFile(emotion_file_curr)[0]));
        list_ims.append(ims_curr[0]+' '+'0');
        list_ims.extend([im_curr+' '+str(emotion_curr) for im_curr in ims_curr[-3:]]);
    
    print len(list_ims);
    util.writeFile(out_file_anno,list_ims);

def makeCKTrainTestFolds():
    anno_file='../data/ck_96/anno_all.txt'
    out_dir='../data/ck_96/train_test_files';
    util.mkdir(out_dir);
    step_size=10;
    num_folds=10;
    train_pre=os.path.join(out_dir,'train_');
    test_pre=os.path.join(out_dir,'test_');

    anno_data=util.readLinesFromFile(anno_file);
    im_files=[file_curr.split(' ')[0] for file_curr in anno_data];
    subjects_all=[os.path.split(file_curr)[1].split('_')[0] for file_curr in im_files]

    subjects=list(set(subjects_all));
    subjects.sort();
    # print len(subjects);
    
    subject_folds={}
    for fold in range(num_folds):
        subjects_curr=subjects[fold::step_size];
        subject_folds[fold]=subjects_curr;
    
    anno_data=np.array(anno_data);
    subjects_all=np.array(subjects_all);
    for fold_num in subject_folds:
        subjects_curr=subject_folds[fold_num];
        bin_data=np.zeros((len(subjects_all),));
        for subject_curr in subjects_curr:
            # print subject_curr;
            bin_data[subjects_all==subject_curr]=1
        test_data=anno_data[bin_data>0];
        train_data=anno_data[bin_data==0];
        test_file=test_pre+str(fold_num)+'.txt';
        train_file=train_pre+str(fold_num)+'.txt';
        util.writeFile(test_file,test_data);
        util.writeFile(train_file,train_data);


def saveMeanSTDFiles(train_file,out_file_pre,resize_size,disp_idx=100):
    im_files=util.readLinesFromFile(train_file);
    im_files=[line_curr.split(' ')[0] for line_curr in im_files];
    mean_im = getMeanImage(im_files,resize_size,disp_idx=disp_idx)
    std_im = getSTDImage(im_files,mean_im,resize_size,disp_idx=disp_idx);
    out_file_mean=out_file_pre+'_mean.png';
    out_file_std=out_file_pre+'_std.png';
    print mean_im.shape
    cv2.imwrite(out_file_mean,mean_im);
    cv2.imwrite(out_file_std,std_im);
    return out_file_mean,out_file_std;

def getMeanImage(im_files,resize_size,disp_idx=1000):
    running_total=np.zeros((resize_size[0],resize_size[1]));
    for idx_file_curr,file_curr in enumerate(im_files):
        if idx_file_curr%disp_idx==0:
            print idx_file_curr;
        im=cv2.imread(file_curr,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        if im.shape[0]!=resize_size[0] or im.shape[1]!=resize_size[1]:
            print 'RESIZING'
            im=scipy.misc.imresize(im,resize_size);
        running_total=running_total+im;

    running_total=running_total.astype(np.float);
    mean=running_total/len(im_files);
    # print np.min(mean),np.max(mean);
    return mean;

def getSTDImage(im_files,mean,resize_size,disp_idx=1000):
    running_total=np.zeros(mean.shape);
    resize_size=(mean.shape[0],mean.shape[1])
    for idx_file_curr,file_curr in enumerate(im_files):
        if idx_file_curr%disp_idx==0:
            print idx_file_curr;
        im=cv2.imread(file_curr,cv2.CV_LOAD_IMAGE_GRAYSCALE);
        if im.shape[0]!=resize_size[0] or im.shape[1]!=resize_size[1]:
            im=scipy.misc.imresize(im,resize_size);
#         im=scipy.misc.imresize(im,resize_size);
        std_curr=np.power(mean-im,2)
        running_total=running_total+im;

    running_total=running_total.astype(np.float);
    std=running_total/len(im_files);
    std=np.sqrt(std)
    # print np.min(std),np.max(std);
    return std;

def saveCKMeanSTDImages(data_dir=None,train_pre=None,resize_size=None,num_folds=None):
    if data_dir is None:
        data_dir='../data/ck_96/train_test_files';
    if train_pre is None:
        train_pre='train_';
    if resize_size is None:
        resize_size=[96,96];
    if num_folds is None:
        num_folds=10;

    for num_fold in range(num_folds):
        train_file=os.path.join(data_dir,train_pre+str(num_fold)+'.txt');
        out_file_pre=os.path.join(data_dir,train_pre+str(num_fold));
        mean_file,std_file=saveMeanSTDFiles(train_file,out_file_pre,resize_size,disp_idx=2000);

        mean_im=cv2.imread(mean_file,cv2.CV_LOAD_IMAGE_GRAYSCALE);
        std_im=cv2.imread(std_file,cv2.CV_LOAD_IMAGE_GRAYSCALE);
        print mean_file,np.min(mean_im),np.max(mean_im),mean_im.shape;
        print std_file,np.min(std_im),np.max(std_im),std_im.shape;


def scratch():
    data_dir='../data/ck_96/train_test_files';
    train_pre='train_';
    
    fold_num=0;
    paths_file=os.path.join(data_dir,train_pre+str(fold_num)+'.txt');
    weights_file=os.path.join(data_dir,train_pre+str(fold_num)+'_weights.npy');

    lines=util.readLinesFromFile(paths_file);
    classes=[int(line_curr.split(' ')[1]) for line_curr in lines];
    classes_set=set(classes);
    n_samples=len(classes);
    n_classes=len(classes_set);
    class_counts=[classes.count(c) for c in range(n_classes)]
    print class_counts;
    
    balanced_weights=float(n_samples)/(n_classes*np.array(class_counts))
    balanced_weights=balanced_weights/np.sum(balanced_weights);
    print balanced_weights
    np.save(weights_file,balanced_weights);


def saveTFDImages():
    tfd_file_name = '../data/tfd/TFD_96x96.mat';
    out_dir_im='../data/tfd/im';
    out_file_anno='../data/tfd/anno_all.txt';
    util.mkdir(out_dir_im);

    fold_info=[('train_',1),('val_',2),('test_',3)];

    tfd_data=scipy.io.loadmat(tfd_file_name);
    ims = tfd_data['images'];
    labels = tfd_data['labs_ex'];
    folds = tfd_data['folds'];
    ids = tfd_data['labs_id'];

    labels= labels[:,0];
    ims = ims[labels>0,:,:];
    folds = folds[labels>0,:];
    num_folds=5;
    ids = ids[labels>0,:];
    labels = labels[labels>0];
    labels = labels-1;

    annos=[];
    for im_num in range(ims.shape[0]):
        im_curr = ims[im_num];
        out_file_curr=os.path.join(out_dir_im,'im_'+str(im_num)+'.jpg');
        scipy.misc.imsave(out_file_curr,im_curr);
        anno_curr=out_file_curr+' '+str(labels[im_num]);
        annos.append(anno_curr) 

    util.writeFile(out_file_anno,annos);

def saveTFDTrainTestFolds():
    tfd_file_name = '../data/tfd/TFD_96x96.mat';
    out_dir_im='../data/tfd/im';
    out_dir_files = '../data/tfd/train_test_files';
    util.mkdir(out_dir_files);
    fold_info=[('train_',1),('val_',2),('test_',3)];

    tfd_data=scipy.io.loadmat(tfd_file_name);

    ims = tfd_data['images'];
    labels = tfd_data['labs_ex'];
    folds = tfd_data['folds'];
    ids = tfd_data['labs_id'];

    labels= labels[:,0];
    ims = ims[labels>0,:,:];
    folds = folds[labels>0,:];
    num_folds=folds.shape[1];
    ids = ids[labels>0,:];
    labels = labels[labels>0];
    labels = labels-1;

    for fold_num in range(num_folds):
        for file_pre,fold_id in fold_info:
            out_file_curr = os.path.join(out_dir_files,file_pre+str(fold_num)+'.txt');
            ids_ims = np.where(folds[:,fold_num]==fold_id)[0];
            im_files=[os.path.join(out_dir_im,'im_'+str(im_num)+'.jpg') for im_num in ids_ims];
            annos=labels[ids_ims];
            lines=[im_file_curr+' '+str(annos[idx]) for idx,im_file_curr in enumerate(im_files)];
            print out_file_curr,lines[0],len(lines);
            util.writeFile(out_file_curr,lines);

def saveTFDMeanSTDImages():
    data_dir='../data/tfd/train_test_files';
    train_pre='train_';
    num_folds=5;
    resize_size=[96,96];
    saveCKMeanSTDImages(data_dir,train_pre,resize_size,num_folds)

def sanityCheckTFD():
    # sanity check.
    # every line exists in anno
    out_dir_meta='../data/tfd';
    num_folds=5;
    anno_file=os.path.join(out_dir_meta,'anno_all.txt');
    anno_data=np.array(util.readLinesFromFile(anno_file));
    file_pres=[os.path.join(out_dir_meta,'train_test_files',type_file+'_') for type_file in ['train','val','test']];

    for fold_num in range(num_folds):
        for file_pre in file_pres:
            file_curr = file_pre+str(fold_num)+'.txt';
            lines=util.readLinesFromFile(file_curr);
            lines=np.array(lines);
            idx_overlap=np.in1d(lines,anno_data);
            assert lines.shape[0]==np.sum(idx_overlap);


    # train test and val are separate
    for fold_num in range(num_folds):
        files=[file_pre+str(fold_num)+'.txt' for file_pre in file_pres];
        lines=[np.array(util.readLinesFromFile(file_curr)) for file_curr in files];
        assert np.sum(np.in1d(lines[1],lines[0]))==0;
        assert np.sum(np.in1d(lines[2],lines[0]))==0;
        assert np.sum(np.in1d(lines[1],lines[2]))==0;

    # other check
    # train of one does not overlap with test of any other
    # not true;
    for fold_num in range(num_folds):
        file_train = file_pres[0]+str(fold_num)+'.txt';
        lines_train = util.readLinesFromFile(file_train);
        for fold_num_test in range(num_folds):
            # if fold_num_test==fold_num:
            #     continue;
            file_test=file_pres[2]+str(fold_num_test)+'.txt'
            lines_test = util.readLinesFromFile(file_test);
            print fold_num,fold_num_test,len(lines_train),len(lines_test);
            print np.sum(np.in1d(lines_test,lines_train));


def downloadImage((url,im_out,im_num)):
    if im_num%100==0:
        print im_num;
    try:
        line_curr='wget --tries=2 -q --timeout=30 '+util.escapeString(url)+' -O '+im_out 
        os.system(line_curr);
    except:
        pass;

def downloadEmotionetDataset():
    # file_post):
    dir_files='../data/emotionet/emotioNet_URLs'
    file_pre='emotioNet_'
    out_dir_meta='../data/emotionet';

    file_posts=range(1,10);
    commands=[];
    for file_post in file_posts:
        out_dir_curr=os.path.join(out_dir_meta,file_pre+str(file_post));
        util.mkdir(out_dir_curr);
        lock=os.path.join(out_dir_curr,'lock');
        if os.path.exists(lock):
            print 'exists ',file_post
            time.sleep(5*random.random());
            continue;

        util.mkdir(lock);

        # t=time.time();
        # time.sleep(120);
        # print file_post,time.time()-t;

        file_curr=os.path.join(dir_files,file_pre+str(file_post)+'.txt');
        
        print file_curr;
        urls=util.readLinesFromFile(file_curr);
        
        args_all=[];
        for idx_url,url in enumerate(urls):
            out_file_curr=os.path.join(out_dir_curr,str(idx_url)+'.jpg');
            if os.path.exists(out_file_curr):
                continue;
            args_all.append((url,out_file_curr,idx_url));
        
        print len(urls),len(args_all);
        print len(args_all);

        idx_range=list(range(0,len(args_all),min(10000,len(args_all)-1)));
        idx_range.append(len(args_all));
        total=0;
        for idx_idx_range in range(len(idx_range)-1):
            start_idx=idx_range[idx_idx_range];
            end_idx=idx_range[idx_idx_range+1];
            args=args_all[start_idx:end_idx]
            print start_idx,end_idx,len(args);
            total=len(args)+total;

            print multiprocessing.cpu_count()
            p=multiprocessing.Pool(multiprocessing.cpu_count())
            #     # 8);
            t=time.time();
            p.map(downloadImage,args);
            print (time.time()-t);

        print len(args_all),total;
    
def viewDatasetEmotionet():

    out_dir_meta=os.path.join('/home/SSD3/maheen-data/expression_project','data/emotionet');
    file_pre='emotioNet_'
    dir_files=file_pre+'URLs';
    file_post='1';
    file_text=os.path.join(out_dir_meta,dir_files,file_pre+file_post+'.txt');
    htmls=util.readLinesFromFile(file_text)
    htmls=htmls[:1000];
    num_files=len(htmls);
    files_all=[os.path.join(out_dir_meta,file_pre+file_post,str(file_num)+'.jpg') for file_num in range(num_files)];
    
    num_gifs=0;
    for html in htmls:
        if '.gif' in html:
            num_gifs+=1

    print num_files,num_gifs,num_gifs/float(num_files)
    # files_all=files_all[:1000];
    files_to_keep=[];
    sizes=[];
    for file_curr,html_link in zip(files_all,htmls):
        if os.path.exists(file_curr):
            size_curr=os.stat(file_curr).st_size
            
            if size_curr>5000 and '.gif' not in html_link:
                #  and 
             # and size_curr>2000:
            #     print file_curr,html_link,size_curr
                files_to_keep.append(file_curr);
                sizes.append(size_curr);
    out_file=os.path.join('../scratch/emotionet_size.jpg');
    # print len(files_to_keep),num_files
    # writeHTML(file_name,im_paths,captions,height=200,width=200)

    print sizes[:10];
    sizes=np.sort(sizes);
    print sizes[:10];
    
    visualize.plotSimple([(range(len(sizes)),sizes)],out_file);
    im_paths=[[util.getRelPath(file_curr,dir_server)] for file_curr in files_to_keep];
    captions=[[''] for file_curr in files_to_keep];
    out_file_html=os.path.join(out_dir_meta,'viz.html');
    visualize.writeHTML(out_file_html,im_paths,captions);
    print len(im_paths);
    print out_file_html.replace(dir_server,click_str);


def isValidImage((html,in_file)):
    
    if not os.path.exists(in_file):
        return False;

    # if '.gif' in html:
    #     return False;
    
    if os.stat(in_file).st_size<3000:
        return False;

    return True;
    # time.sleep(2*random.random());
    # print num_file
    # return num_file;

def getListOfValidImagesEmotionet():
    # out_dir_meta=os.path.join('/home/SSD3/maheen-data/expression_project','data/emotionet');
    out_dir_meta = '../data/emotionet';
    file_pre='emotioNet_'
    dir_files=file_pre+'URLs';
    file_post='1';
    for file_post in range(9,10):
        file_post=str(file_post);
        file_text = os.path.join(out_dir_meta,dir_files,file_pre+file_post+'.txt');
        out_text = os.path.join(out_dir_meta,dir_files,file_pre+file_post+'_kept.txt');
        print file_text;
        htmls=util.readLinesFromFile(file_text)
        # htmls=htmls[:1000];
        num_files=len(htmls);
        files_all=[os.path.join(out_dir_meta,file_pre+file_post,str(file_num)+'.jpg') for file_num in range(num_files)];
        args=zip(htmls,files_all);
        t=time.time();
        p=multiprocessing.Pool(multiprocessing.cpu_count());
        bool_keep = p.map(isValidImage,args);
        print time.time()-t;
        
        # bool_keep = np.array(bool_keep);
        # files_to_keep=np.array(files_all)[bool_keep];
        # htmls_to_keep=np.array(htmls)[bool_keep];

        out_lines=[htmls[idx]+' '+files_all[idx] for idx,bool_val in enumerate(bool_keep) if bool_val];
        print len(out_lines),len(htmls),len(out_lines)/float(len(htmls))
        print out_text;    
        util.writeFile(out_text,out_lines);

    # im_paths=[[util.getRelPath(file_curr,dir_server)] for file_curr in files_to_keep];
    # captions=[[''] for file_curr in files_to_keep];
    # out_file_html=os.path.join(out_dir_meta,'viz.html');
    # print len(im_paths);
    # visualize.writeHTML(out_file_html,im_paths,captions);
    # print out_file_html.replace(dir_server,click_str);

def writeIndexFileEmotionet():
    out_dir_meta = '../data/emotionet';
    dir_anno= os.path.join(out_dir_meta,'emotioNet_challenge_files_updated');
    file_pre  = 'dataFile_00';
    out_file = os.path.join(dir_anno,'index_file.txt');
    file_posts= range(1,94);
    lines_to_write=[];
    for file_post in file_posts:

        if file_post<10:
            file_post='0'+str(file_post)+'.txt'
        else:
            file_post=str(file_post)+'.txt'
        
        file_curr=os.path.join(dir_anno,file_pre+file_post);
        print file_curr;
        lines=util.readLinesFromFile(file_curr);
        # print lines[0];
        # print lines[0].rsplit(' ',20);
        # raw_input();
        lines_to_write.extend([line_curr.split('\t')[0]+' '+file_curr for line_curr in lines]);
        print len(lines_to_write);

    util.writeFile(out_file,lines_to_write);

def writeMissingImageFile():
    out_dir_meta = '../data/emotionet';
    anno_dir = os.path.join(out_dir_meta,'emotioNet_challenge_files_updated');
    kept_dir =  os.path.join(out_dir_meta,'emotioNet_URLs');
    index_file = os.path.join(anno_dir,'index_file.txt');
    out_file = os.path.join(kept_dir,'emotioNet_leftover.txt');

    index_data=util.readLinesFromFile(index_file);
    index_data_pre=[line_curr.split(' ')[0] for line_curr in index_data];
    # print index_data_pre[:10];

    in_files=[os.path.join(kept_dir,'emotioNet_'+str(num)+'_kept.txt') for num in range(1,10)]
    hash_list=getHashList(in_files);
    
    im_paths=[];
    match_file=[];
    missing_htmls=[];
    not_found=0;
    for idx_index_data_curr,index_data_curr in enumerate(index_data_pre):
        if idx_index_data_curr%10000==0:
            print idx_index_data_curr;

        found=False;
        for idx_hash_table_curr,hash_table_curr in enumerate(hash_list):
            if hash_table_curr.has_key(index_data_curr):
                im_paths.append(hash_table_curr[index_data_curr]);
                match_file.append(idx_hash_table_curr);
                found=True;
                break;
        
        if not found:
            missing_htmls.append(index_data_curr);
            not_found+=1;
            im_paths.append('-1');
            match_file.append('-1');

    assert len(im_paths)==len(index_data);
    print not_found,len(index_data),not_found/float(len(index_data));
    index_data=[line_curr+' '+im_paths[idx] for idx,line_curr in enumerate(index_data)];
    print out_file;
    print len(missing_htmls);   
    util.writeFile(out_file,missing_htmls);

def getHashList(in_files):
    hash_list=[];
    for in_file in in_files:
        htmls=util.readLinesFromFile(in_file);
        hash_curr={};
        for line_curr in htmls:
            line_split=line_curr.split(' ');
            hash_curr[line_split[0]]=line_split[1];
        hash_list.append(hash_curr);
    return hash_list;

def getAnnos(anno_file,hash_list,exp_flag=False):
    anno_data=util.readLinesFromFile(anno_file);
    annos_skipped=[];
    annos_all=[];
    for idx_line_curr,line_curr in enumerate(anno_data):
        if exp_flag:
            num_toks=16;
        else:
            num_toks=60;
        line_curr=line_curr.rsplit(None,num_toks);
        assert len(line_curr)==num_toks+1
        html_curr=line_curr[0].strip("'");
        im_curr=None;
        for hash_curr in hash_list:
            if hash_curr.has_key(html_curr):
                im_curr=hash_curr[html_curr];
                break;

        if im_curr is None:
            annos_skipped.append(html_curr);
            continue;

        annos_curr=[str(idx+1) for idx,val in enumerate(line_curr[1:]) if int(val)==1];
        if len(annos_curr)==0:
            annos_curr=[str(-1)];
        annos_curr=[html_curr,im_curr]+annos_curr
        annos_curr=' '.join(annos_curr);
        annos_all.append(annos_curr);

    return annos_all,annos_skipped

def parseAnnotationsEmotionet():
    out_dir_meta = '../data/emotionet';
    anno_dir = os.path.join(out_dir_meta,'emotioNet_challenge_files_updated');
    kept_dir =  os.path.join(out_dir_meta,'emotioNet_URLs');
    anno_file_pre  = 'dataFile_00';

    kept_files=[os.path.join(kept_dir,'emotioNet_'+str(num)+'_kept.txt') for num in range(1,10)]
    hash_list = getHashList(kept_files);
    
    # annos_all_file=os.path.join(anno_dir,'annos_all.txt');
    annos_all_file=os.path.join(anno_dir,'annos_val.txt');
    annos_all=[];
    annos_skipped_all=[];
    # for anno_num in range(1,94):
    # print anno_num
    #     if anno_num<10:
    #         anno_file_post='0'+str(anno_num)+'.txt';
    #     else:
    #         anno_file_post=str(anno_num)+'.txt';
    anno_file=os.path.join(anno_dir,'val_data.txt')
    # anno_file=os.path.join(anno_dir,anno_file_pre+anno_file_post);    
    annos,annos_skipped=getAnnos(anno_file,hash_list);
    print len(annos),len(annos_skipped),len(annos_skipped)/float(len(annos));
    annos_all.extend(annos);
    annos_skipped_all.extend(annos_skipped);
    
    print len(annos_skipped_all);
    print len(annos_all);

    util.writeFile(annos_all_file,annos_all);

def parseAnnotationsExpressionEmotionet():
    out_dir_meta = '../data/emotionet';
    anno_dir = os.path.join(out_dir_meta,'anno_exp');
    kept_dir =  os.path.join(out_dir_meta,'emotioNet_URLs');
    anno_file=os.path.join(anno_dir,'val_exp.txt')
    kept_files=[os.path.join(kept_dir,'emotioNet_'+str(num)+'_kept.txt') for num in range(1,10)]
    hash_list = getHashList(kept_files);
    
    # annos_all_file=os.path.join(anno_dir,'annos_all.txt');
    annos_all_file=os.path.join(anno_dir,'annos_val.txt');
    annos_all=[];
    annos_skipped_all=[];
    # for anno_num in range(1,94):
    # print anno_num
    #     if anno_num<10:
    #         anno_file_post='0'+str(anno_num)+'.txt';
    #     else:
    #         anno_file_post=str(anno_num)+'.txt';
    
    # anno_file=os.path.join(anno_dir,anno_file_pre+anno_file_post);    
    annos,annos_skipped=getAnnos(anno_file,hash_list,True);
    print len(annos),len(annos_skipped),len(annos_skipped)/float(len(annos));
    annos_all.extend(annos);
    annos_skipped_all.extend(annos_skipped);
    
    print len(annos_skipped_all);
    print len(annos_all);

    util.writeFile(annos_all_file,annos_all);

def mappingCheck():
    out_dir_meta='../data/emotionet';
    au_dir=os.path.join(out_dir_meta,'emotioNet_challenge_files_updated');
    exp_dir=os.path.join(out_dir_meta,'anno_exp');
    val_exp=os.path.join(exp_dir,'annos_val.txt')
    all_au=os.path.join(au_dir,'annos_all.txt');
    val_au=os.path.join(au_dir,'annos_val.txt');
    exp_data=util.readLinesFromFile(val_exp);
    exp_data=[line_curr.split(' ') for line_curr in exp_data];
    exp_data=np.array(exp_data);
    au_data=util.readLinesFromFile(val_au)+util.readLinesFromFile(all_au);
    au_data=[line_curr.split(None,2) for line_curr in au_data];
    au_data=np.array(au_data);

    exp=np.unique(exp_data[:,-1]);
    exp_dict={};
    for exp_curr in exp:
        exp_dict[exp_curr]=[];

    for exp_curr in exp:
        htmls=exp_data[exp_data[:,-1]==exp_curr,1];
        html_bin=np.in1d(au_data[:,1],htmls);
        aus=np.unique(au_data[html_bin,-1]);
        exp_dict[exp_curr].extend(list(aus));
    
    for exp in exp_dict.keys():
        print 'EXP',exp,len(exp_dict[exp]);
        for val in exp_dict[exp]:
            print val;
        print '___';

def makeTrainTestAu():
    out_dir_meta = '../data/emotionet';
    au_dir = os.path.join(out_dir_meta,'emotioNet_challenge_files_updated');
    all_au = os.path.join(au_dir,'annos_all.txt');
    val_au = os.path.join(au_dir,'annos_val.txt');
    out_dir_anno=os.path.join(out_dir_meta,'train_test_files');
    util.mkdir(out_dir_anno);
    train_out=os.path.join(out_dir_anno,'train_au_0.txt');
    val_out=os.path.join(out_dir_anno,'val_au_0.txt');
    all_au_data_list=util.readLinesFromFile(all_au);
    
    all_au_data=np.array([line_curr.split(None,2) for line_curr in all_au_data_list]);
    
    val_au_data_list=util.readLinesFromFile(val_au);
    val_au_data=np.array([line_curr.split(None,2) for line_curr in val_au_data_list]);
    
    train_au_bool=np.in1d(all_au_data[:,1], val_au_data[:,1],invert=True);
    train_au_data=np.array(all_au_data_list)[train_au_bool];
    util.writeFile(train_out,train_au_data)
    util.writeFile(val_out,val_au_data_list)

def makeTrainTestExp():
    out_dir_meta = '../data/emotionet';
    out_dir_exp=os.path.join(out_dir_meta,'anno_exp');
    out_dir_anno=os.path.join(out_dir_meta,'train_test_files');
    ref_file=os.path.join(out_dir_anno,'exp_ref.txt');
    train_au=os.path.join(out_dir_anno,'train_au_0.txt');
    val_exp=os.path.join(out_dir_exp,'annos_val.txt');

    train_out=os.path.join(out_dir_anno,'train_0.txt');
    val_out=os.path.join(out_dir_anno,'val_0.txt');
    

    au_data_list=util.readLinesFromFile(train_au);
    au_data=np.array([line_curr.split(None,2) for line_curr in au_data_list]);
    
    val_data_list=util.readLinesFromFile(val_exp);
    val_data=np.array([line_curr.split(None,2) for line_curr in val_data_list]);
    bool_keep=np.in1d(au_data[:,1],val_data[:,1],invert=True);
    print bool_keep.shape,au_data.shape,val_data.shape,sum(bool_keep);
    au_data_list=np.array(au_data_list)[bool_keep];
    au_data=au_data[bool_keep,:];
    print bool_keep.shape,au_data.shape,val_data.shape,sum(bool_keep);

    ref_data=util.readLinesFromFile(ref_file);
    ref_data=[line_curr.split(None,2) for line_curr in ref_data];

    train_all=np.zeros((0,au_data.shape[1]));
    for [exp_str,exp_num,exp_au] in ref_data:
        print exp_str,exp_num,exp_au
        bool_keep=au_data[:,2]==exp_au;
        print sum(bool_keep);
        train_exp_curr=au_data[bool_keep,:];
        train_all=np.concatenate((train_all,train_exp_curr));

    train_all=[' '.join(list(line_curr)) for line_curr in train_all];
    print len(train_all),train_all[100],train_out;
    util.writeFile(train_out,train_all);
    print len(val_data_list),val_data_list[100],val_out;
    util.writeFile(val_out,val_data_list); 

        

    # print ref_data;


def visualizeTestData():
    data_dir='../data/tfd/train_test_files';
    data_set='tfd_ckLabels';
    num_folds=5;
    num_classes=8;
    
    # data_dir='../data/ck_96/train_test_files';
    # data_set='ck';
    # num_folds=10;
    # num_classes=8;

    test_pre='test_ckLabels_';
    replace_path=['..',os.path.join(dir_server,'expression_project')];
    out_dir_html=os.path.join(replace_path[1],'scratch','view_tfd_classes');
    util.mkdir(out_dir_html);
    
    out_file_html=os.path.join(out_dir_html,data_set+'.html');
    

    img_paths=[];
    captions=[];
    for class_num in range(num_classes):
        img_paths.append([]);
        captions.append([]);
    
    for num_fold in range(num_folds):
        test_file_curr=os.path.join(data_dir,test_pre+str(num_fold)+'.txt');
        data=util.readLinesFromFile(test_file_curr);
        for line_curr in data:
            im_curr=line_curr.split(' ');
            label_curr=im_curr[1];
            im_curr=util.getRelPath(im_curr[0].replace(replace_path[0],replace_path[1]),dir_server);
            img_paths[int(label_curr)].append(im_curr);
            captions[int(label_curr)].append(label_curr);

    visualize.writeHTML(out_file_html,img_paths,captions);
    print out_file_html.replace(dir_server,click_str);

def modifyTFDClassesForCK():

    data_dir='../data/tfd/train_test_files';
    in_pre='test_';
    out_pre='test_ckLabels_';
    num_folds=5;

    for num_fold in range(num_folds):
        in_file=os.path.join(data_dir,in_pre+str(num_fold)+'.txt');
        out_file=os.path.join(data_dir,out_pre+str(num_fold)+'.txt');
        lines=util.readLinesFromFile(in_file);
        lines_new=[];
        for line_curr in lines:
            line_curr=line_curr.split(' ');
            label=int(line_curr[1]);
            if label==0:
                label=1;
            elif label==6:
                label=0;
            else:
                label=label+2;
            line_curr=line_curr[0]+' '+str(label);
            lines_new.append(line_curr);
        print out_file;
        util.writeFile(out_file,lines_new);

def combineTFDCkFiles():
    num_folds=5;
    dir_data='../data/tfd/train_test_files';
    in_pre='test_ckLabels_';
    out_file=os.path.join(dir_data,in_pre+'all.txt');
    lines_all=[];
    for num_fold in range(num_folds):
        file_curr=os.path.join(dir_data,in_pre+str(num_fold)+'.txt')
        lines_all.extend(util.readLinesFromFile(file_curr));
    
    print len(lines_all)
    lines_all=list(set(lines_all))
    print len(lines_all)
    util.writeFile(out_file,lines_all);
    
def main(args):
    data_dir='../data/karina_vids/data_unprocessed/test_train_images';
    train_pre='train_';
    num_folds=6;

    saveCKMeanSTDImages(data_dir,train_pre,resize_size=None,num_folds=num_folds)
    # combineTFDCkFiles();

    # modifyTFDClassesForCK();
    # visualizeTestData();




    # mappingCheck();
    # makeTrainTestExp();
    # makeTrainTestAu();

    # util.writeFile(out_file,lines_all);
    # print out_file
    
    # parseAnnotationsExpressionEmotionet();
    # im='../data/emotionet/emotioNet_9/100021.jpg';
    # print os.stat(im).st_size
    # writeIndexFileEmotionet()
    # writeMissingImageFile();
    # getListOfValidImagesEmotionet
    # getListOfValidImagesEmotionet()
    # viewDatasetEmotionet()    
    # downloadEmotionetDataset();
    # url='http://image.pixmac.com/4/horror-dark-emotion-young-girl-face-abstract-pixmac-image-86178608.jpg';
    # url = 'https://sassafrasjunction.files.wordpress.com/2011/02/girl-scout.gif';
    # out_file='../scratch/test.jpg';
    # # downloadImage((url,out_file,1));
    # # print args[1];
    # time.sleep(5*random.random());
    # downloadEmotionetDataset()
    # int(args[1]));
    # t=time.time();
    # time.sleep(120);
    # print time.time()-t;


if __name__ == '__main__':
    main(sys.argv);