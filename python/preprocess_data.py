import numpy as np;
import cv2;
import util;
import os;
import visualize;
import scipy;
import scipy.io;
import multiprocessing;
import subprocess;
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
        # t=time.time();
        downloader=urllib.URLopener()
        # print time.time()-t;
        # t=time.time();
        downloader.retrieve(url, im_out)
        # print time.time()-t;
    except:
        pass;

def downloadEmotionetDataset():
    dir_files='../data/emotionet/emotioNet_URLs'
    file_pre='emotioNet_'
    out_dir_meta='../data/emotionet';

    file_posts=range(1,9);

    out_file_commands=os.path.join('../scripts/downloading_commands.sh');
    commands=[];
    for file_post in file_posts:
        file_curr=os.path.join(dir_files,file_pre+str(file_post)+'.txt');
        out_dir_curr=os.path.join(out_dir_meta,file_pre+str(file_post));
        util.mkdir(out_dir_curr);

        print file_curr;
        urls=util.readLinesFromFile(file_curr);
        # urls=urls[:1000];
        
        args=[];
        for idx_url,url in enumerate(urls):
            out_file_curr=os.path.join(out_dir_curr,str(idx_url)+'.jpg');
            if os.path.exists(out_file_curr):
                continue;
            args.append((url,out_file_curr,idx_url));
            # out_files=[os.path.join(out_dir_curr,str(im_num)+'.jpg') for im_num in range(len(urls))];

        # args=zip(urls,out_files,range(len(urls)));
        # for arg_curr in args[:10]:
        #     downloadImage(arg_curr);        
        print len(urls),len(args);
        args=args[:100];
        file_for_wget=os.path.join(out_dir_curr,'wget_file.txt');
        lines=['wget --tries=1 -q --timeout=15 '+util.escapeString(url)+' -O '+out_file for url,out_file,num in args];
        util.writeFile(file_for_wget,lines);
        # # commands.append('cat '+file_for_wget+'| parallel -j20');
        t=time.time();
        # # print 'cat '+file_for_wget+'| head -5 | parallel -j20'
        os.system('cat '+file_for_wget+'| parallel -j20');
        print time.time()-t;
        # print file_for_wget;

        # t=time.time();
        # p.map(downloadImage,args);
        # print (time.time()-t);
        # print args[:10];


        # total_len=len(ims);
        # ims_name=[os.path.split(line_curr)[1] for line_curr in ims];
        # ims_ends=[im_curr[im_curr.rindex('.'):] for im_curr in ims_name if '.' in im_curr];
        # print total_len-len(ims_ends);
        # ims_ends=list(set(ims_ends));
        # print len(ims_ends);
        # for im_end in ims_ends:
        #     print im_end;
        # break;
        # print total_len,len(set(ims_name)),len(set(ims_name))==total_len
        # assert len(set(ims_name))==total_len;
    # print out_file_commands
    # util.writeFile(out_file_commands,commands);

    #     
    # p.map(saveBBoxIm,args);
    # p.map(saveBBoxNpy,args_bbox_npy);




def main():
    url='http://image.pixmac.com/4/horror-dark-emotion-young-girl-face-abstract-pixmac-image-86178608.jpg';
    url = 'https://sassafrasjunction.files.wordpress.com/2011/02/girl-scout.gif';
    out_file='../scratch/test.jpg';
    # downloadImage((url,out_file,1));
    downloadEmotionetDataset();

    
    








    

    
    
    







if __name__ == '__main__':
    main();