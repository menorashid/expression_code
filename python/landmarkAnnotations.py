import os;
import dlib
import numpy as np;
import util;
import cv2;
import visualize;
from skimage import io;
import math;
import random;
import preprocess_data;
import scripts_and_viz

dir_server='/home/SSD3/maheen-data/';
click_str='http://vision1.idav.ucdavis.edu:1000/';


def getAnnoNumpy(im_file,detector,predictor,num_kp=68):
	im=io.imread(im_file);
	dets=detector(im);
	if len(dets)>1:
		print 'MOREDETS';
	shape=predictor(im,dets[0]);
	parts=[[shape.part(num).x,shape.part(num).y] for num in range(num_kp)]
	return parts

def saveHappyNeutralTrainTest():
	data_dir='../data/ck_96/train_test_files';

	out_file_train=os.path.join(data_dir,'train_happy_neutral.txt');
	out_file_test=os.path.join(data_dir,'test_happy_neutral.txt');
	out_file_mean_pre=os.path.join(data_dir,'train_happy_neutral');

	file_names=[os.path.join(data_dir,file_pre+'_'+str(file_post)+'.txt') for file_pre in ['train','test'] for file_post in range(10)];

	lines_all=[];
	for file_name in file_names:
		lines_all.extend(util.readLinesFromFile(file_name));
	print len(lines_all);
	lines_all=list(set(lines_all));
	print len(lines_all);

	classes_to_keep=[0,5];
	class_labels=['neutral','happy'];
	new_labels=['-1','1']

	lines_classes=[int(line_curr.split(' ')[1]) for line_curr in lines_all];
	lines_classes=np.array(lines_classes);
	lines_all=np.array(lines_all)
	class_lines=[];
	for new_label,class_to_keep in zip(new_labels,classes_to_keep):
		lines_curr=lines_all[lines_classes==class_to_keep]
		lines_curr=[line_curr.split(' ')[0]+' '+str(new_label) for line_curr in lines_curr];
		class_lines.append(np.sort(lines_curr));
	
	ratio_train=0.9;
	train_lines_all=[];
	test_lines_all=[];
	
	for class_lines_curr in class_lines:
		total=len(class_lines_curr);
		num_train=math.floor(total*ratio_train);
		train_lines=class_lines_curr[:num_train];
		test_lines=class_lines_curr[num_train:];
		train_subs=np.array([os.path.split(file_curr)[1].split('_')[0] for file_curr in train_lines]);
		test_subs=np.array([os.path.split(file_curr)[1].split('_')[0] for file_curr in test_lines]);
		assert np.sum(np.in1d(test_subs,train_subs))==0;
		print train_subs[0],test_subs[0]
		train_lines_all.extend(train_lines);
		test_lines_all.extend(test_lines);
		print len(train_lines_all),len(test_lines_all);

	random.shuffle(train_lines_all);
	random.shuffle(test_lines_all);
	util.writeFile(out_file_train,train_lines_all);
	util.writeFile(out_file_test,test_lines_all);
	preprocess_data.saveMeanSTDFiles(out_file_train,out_file_mean_pre,[96,96],disp_idx=100);


def getAnnoString(annos,annos_to_keep,avg_nums):
	annos_all=[];
	for annos_idx in annos_to_keep:
		annos_curr_tot=np.array([0,0]);
		for anno_idx_curr in annos_idx:
			anno_curr=annos[anno_idx_curr];
			annos_curr_tot=annos_curr_tot+anno_curr;
		annos_curr_tot=annos_curr_tot/len(annos_idx)
		annos_all.append(annos_curr_tot);

	annos_final=[];
	for supp_num in avg_nums:
		anno_curr=(annos_all[supp_num[0]]+annos_all[supp_num[1]])/2;

		annos_final.extend(list(anno_curr));
	annos_final=[str(num) for num in annos_final];
	annos_final=' '.join(annos_final);
	return annos_final

def writeTrainTestFilesWithAnno():
	predictor_file='../../dlib-19.4.0/python_examples/shape_predictor_68_face_landmarks.dat';
	in_file_pre='happy_neutral';
	out_dir=os.path.join(dir_server,'expression_project','scratch',in_file_pre+'_anno');
	data_dir='../data/ck_96/train_test_files';
	util.mkdir(out_dir);

	annos_to_keep=[[48],[54],[62,66]];
	avg_nums=[[0,0],[1,1],[2,2],[0,2],[1,2]];
	
	detector=dlib.get_frontal_face_detector()
	predictor=dlib.shape_predictor(predictor_file);

	file_list=[os.path.join(data_dir,file_pre+'_'+in_file_pre+'.txt') for file_pre in ['train','test']];
	new_files=[os.path.join(data_dir,file_pre+'_'+in_file_pre+'_withAnno.txt') for file_pre in ['train','test']];
	
	for data_file,out_file in zip(file_list,new_files):
		
		lines_old=util.readLinesFromFile(data_file);
		lines_new=[];
		for im_num,line_curr in enumerate(lines_old):
			im_file=line_curr.split(' ')[0]
			if im_num%100==0:
				print im_num,len(lines_old);
			annos=getAnnoNumpy(im_file,detector,predictor);
			annos_string=getAnnoString(annos,annos_to_keep,avg_nums);
			lines_new.append(line_curr+' '+annos_string)			
			
		util.writeFile(out_file,lines_new);
		print out_file;
	

def main():
	
	# saveHappyNeutralTrainTest()
	# writeTrainTestFilesWithAnno()


	path_to_th='train_withAnno.th';
	out_dir_meta='../experiments/happy_neutral/bl';
	util.makedirs(out_dir_meta);
	dir_files='../data/ck_96/train_test_files';
	fold_num='happy_neutral_withAnno';
	model_file='../models/base_khorrami_model_1.dat'
	iterations=150;
	saveAfter=5;
	testAfter=1;
	learningRate=0.01;
	twoClass=True;

	command = scripts_and_viz.writeBlurScript(path_to_th,out_dir_meta,dir_files,fold_num,model_file=model_file,twoClass=twoClass,iterations=iterations,saveAfter=saveAfter,learningRate=learningRate,testAfter=testAfter);
	print command;
 #    fold_num,
 #    model_file=None,
 #    batchSize=128,
 #    learningRate=0.01,
 #    scheme='ncl',
 #    ratioBlur=0,
 #    incrementDifficultyAfter=-1,
 #    startingActivation=0,
 #    fixThresh=-1,
 #    activationThreshMax=0.5,
 #    iterations=1000,
 #    saveAfter=100,
 #    testAfter=10,
 #    dispAfter=1,
 #    dispPlotAfter=10,
 #    batchSizeTest=128,
 #    modelTest=None,
 #    resume_dir_meta=None
 #    )




if __name__=='__main__':
	main();
