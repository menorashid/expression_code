import numpy as np;
import cv2;
import util;
import os;
import visualize;
import scipy;
dir_server='/home/SSD3/maheen-data/';
click_str='http://vision1.idav.ucdavis.edu:1000/';

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

def saveCKMeanSTDImages():
    data_dir='../data/ck_96/train_test_files';
    train_pre='train_';
    resize_size=[96,96];
    for num_fold in range(10):
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

def main():
    experiment_name='khorrami_basic_aug_fix_resume_again';
    out_dir_meta='../experiments/'+experiment_name;
    util.mkdir(out_dir_meta);
    out_script='../scripts/train_'+experiment_name+'.sh';
    path_to_th='train_khorrami_basic.th';

    dir_files='../data/ck_96/train_test_files';
    resume_dir_meta='../experiments/khorrami_basic_aug_fix_resume'
    num_folds=10;

    commands=[];
    # print '{',
    for fold_num in range(num_folds):
        
        
        data_path=os.path.join(dir_files,'train_'+str(fold_num)+'.txt');
        val_data_path=os.path.join(dir_files,'test_'+str(fold_num)+'.txt');
        mean_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_mean.png');
        std_im_path=os.path.join(dir_files,'train_'+str(fold_num)+'_std.png');

        batchSize=64;
        epoch_size=len(util.readLinesFromFile(data_path))/batchSize;
        batchSizeTest=len(util.readLinesFromFile(val_data_path));
        # print str(batchSizeTest)+',',

        outDir=os.path.join(out_dir_meta,str(fold_num));

        command=['th',path_to_th];
        if resume_dir_meta is not None:
            model_path_resume=os.path.join(resume_dir_meta,str(fold_num),'final','model_all_final.dat');
            command = command+['-model',model_path_resume];            

        command = command+['-mean_im_path',mean_im_path];
        command = command+['-std_im_path',std_im_path];
        command = command+['-batchSize',batchSize];

        command = command+['-iterations',250*epoch_size];
        command = command+['-saveAfter',10*epoch_size];
        command = command+['-testAfter',5*epoch_size];
        command = command+['-dispAfter',1];
        command = command+['-dispPlotAfter',5*epoch_size];

        command = command+['-val_data_path',val_data_path];
        command = command+['-data_path',data_path];
        
        command = command+['-iterationsTest',1];
        command = command+['-batchSizeTest',batchSizeTest];
        
        command = command+['-outDir',outDir];
        command = command+['-modelTest',os.path.join(outDir,'final','model_all_final.dat')];
        command = [str(c_curr) for c_curr in command];
        command=' '.join(command);
        print command;
        commands.append(command);
    # print '}';
    print out_script
    util.writeFile(out_script,commands);



    
    
    







if __name__ == '__main__':
    main();