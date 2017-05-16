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
	if detector is None:
		dets=[dlib.rectangle(0,0,96,96)];
	else:	
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
	
def writeTrainScript():
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

def writeAnnoFile(file_list,new_files,detector_file=None,predictor_file=None,range_keep=range(17,68)):
	if detector_file is None:
		detector=None;
	if predictor_file is None:
		predictor_file='../../dlib-19.4.0/python_examples/shape_predictor_68_face_landmarks.dat';

	predictor=dlib.shape_predictor(predictor_file);
	
	annos_to_keep=[[num] for num in range_keep];
	avg_nums=[[num,num] for num in range(len(annos_to_keep))];

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
			# break;
		util.writeFile(out_file,lines_new);
		print out_file;


def writeTrainTestFilesWithAnno_TFD_51():
	out_dir=os.path.join(dir_server,'expression_project','scratch','tfd_4_anno');
	util.mkdir(out_dir);

	data_dir='../data/tfd/train_test_files';
	# /test_4.txt';
	in_file_pre='4';

	data_dir='../data/ck_96/train_test_files';
	# /test_4.txt';
	in_file_pre='6';


	file_list=[os.path.join(data_dir,file_pre+'_'+in_file_pre+'.txt') for file_pre in ['train','test']];
	new_files=[os.path.join(data_dir,file_pre+'_'+in_file_pre+'_withAnno.txt') for file_pre in ['train','test']];
	writeAnnoFile(file_list,new_files)
	# ,detector_file=None,predictor_file=None,range_keep=range(17,68))
	
	
def makeRegionGraphs(in_file,out_file_pre,expression):
	hist_curr=np.load(in_file);
	# print hist_curr,in_file;
	vals_all=hist_curr[:,0]/hist_curr[:,1];
	out_mouth=[31,32,42,33,41,34,40,35,39,36,38,37];
	in_mouth=[43,44,50,45,49,46,48,47];
	nose=range(10,19);
	brows=range(10);
	eyes=[19,20,24,21,23,22,25,26,30,27,29,28];
	nums=[out_mouth,in_mouth,nose,brows,eyes];
	names=['out_mouth','in_mouth','nose','brows','eyes'];
	for area_name,area_num in zip(names,nums):
		vals=list(vals_all[np.array(area_num)]);
		out_file=out_file_pre+'_'+area_name+'.jpg';
		x=range(len(vals));
		xAndYs=[(x,vals)];
		title=area_name+' '+expression;
		xlabel='Keypoint';
		ylabel='Average Importance'
		xticks=[str(num) for num in area_num];
		# print xAndYs,out_file,title,xlabel,ylabel,xticks
		visualize.plotSimple(xAndYs,out_file,title=title,xlabel=xlabel,ylabel=ylabel,xticks=xticks);
			# ,xticks=xticks);
    
		# visualize.makeBarGraph(out_file,

def makeComparativeHtmlAreaGraphs():
	expression_nums=range(8);
	# expression_nums=[6];expression_names=['sadness'];
	# range(8);
	expression_names=['neutral', 'anger','contempt','disgust', 'fear', 'happy', 'sadness', 'surprise'];
	
	in_dir_meta=os.path.join(dir_server,'expression_project','experiments/ck_meanBlur_fixThresh_100_inc');
	out_file_html=os.path.join(in_dir_meta,'comparison_loc.html');
	schemes=['bl','ncl','mixcl','mix'];
	incs=['None','300','200','200'];
	fold='6';
	area_names=['out_mouth','in_mouth','nose','brows','eyes'];
	ims_html=[];captions=[];
	for exp_num,exp_name in zip(expression_nums,expression_names):
		for scheme,inc in zip(schemes,incs):
			im_row_curr=[];
			caption_row_curr=[];
			if inc=='None':
				dir_curr=os.path.join(in_dir_meta,scheme,fold,'test_images_localization')
			else:
				dir_curr=os.path.join(in_dir_meta,scheme,inc,fold,'test_images_localization')
			for area_name in area_names:
				caption_curr=' '.join([scheme,exp_name,area_name]);
				im_file=os.path.join(dir_curr,exp_name+'_'+area_name+'.jpg');
				im_row_curr.append(util.getRelPath(im_file,dir_server));
				caption_row_curr.append(caption_curr);
			ims_html.append(im_row_curr);
			captions.append(caption_row_curr);

	visualize.writeHTML(out_file_html,ims_html,captions,200,300);
	print out_file_html.replace(dir_server,click_str);

def makeHTMLForMaps():
	# in_dir_meta=os.path.join(dir_server,'expression_project','experiments/ck_meanBlur_fixThresh_100_inc');
	# fold='6';
	in_dir_meta=os.path.join(dir_server,'expression_project','experiments/ck_testing_tfd_4_fix');
	fold='0';
	expression_names=['neutral', 'anger','contempt','disgust', 'fear', 'happy', 'sadness', 'surprise'];
	expression_nums=range(8);
	incs=['None','200','200','200'];

	# in_dir_meta=os.path.join(dir_server,'expression_project','experiments/tfd_meanBlur_fixThresh_100_inc');
	# fold='4';
	# in_dir_meta=os.path.join(dir_server,'expression_project','experiments/tfd_testing_ck_all_fix');
	# fold='0';
	# expression_names=['anger','disgust', 'fear', 'happy', 'sadness', 'surprise','neutral'];
	# expression_nums=range(7);
	# incs=['None','300','300','300'];

	out_file_html=os.path.join(in_dir_meta,'comparison_loc.html');
	schemes=['bl','ncl','mixcl','mix'];
	
	
	# area_names=['out_mouth','in_mouth','nose','brows','eyes'];
	ims_html=[];captions=[];
	for exp_num,exp_name in zip(expression_nums,expression_names):
		im_row_curr=[];
		caption_row_curr=[];

		for scheme,inc in zip(schemes,incs):
			# if inc=='None':
			# 	dir_curr=os.path.join(in_dir_meta,scheme,fold,'test_images_localization')
			# else:
			dir_curr=os.path.join(in_dir_meta,scheme,inc,fold,'test_images_localization')

			caption_curr=' '.join([scheme,exp_name]);
			im_file=os.path.join(dir_curr,'exp_num_map_'+str(exp_num)+'.jpg');
			im_row_curr.append(util.getRelPath(im_file,dir_server));
			caption_row_curr.append(caption_curr);
		ims_html.append(im_row_curr);
		captions.append(caption_row_curr);

	visualize.writeHTML(out_file_html,ims_html,captions,200,200);
	print out_file_html.replace(dir_server,click_str);

def script_makeRegionGraphs():

	in_dir_meta=os.path.join(dir_server,'expression_project','experiments/ck_meanBlur_fixThresh_100_inc');
	
	schemes=['mix','mixcl','ncl'];
	inc_range=[str(num) for num in range(100,600,100)]
	

	# return
	# schemes=['bl']
	# ,'mixcl','ncl'];
	# inc_range=['None']
	
	# '200';
	
	fold='6';
	for scheme in schemes:
		for inc in inc_range:
			# dir_curr=os.path.join(in_dir_meta,scheme,fold,'test_images_localization')
			if inc=='None':
				dir_curr=os.path.join(in_dir_meta,scheme,fold,'test_images_localization')
			else:
				dir_curr=os.path.join(in_dir_meta,scheme,inc,fold,'test_images_localization')
			for exp_num,exp_name in zip(expression_nums,expression_names):
				file_curr=os.path.join(dir_curr,'exp_num_'+str(exp_num)+'.npy');
				out_file_pre=os.path.join(dir_curr,exp_name);
				makeRegionGraphs(file_curr,out_file_pre,exp_name);

			visualize.writeHTMLForFolder(dir_curr);

def main():
	makeHTMLForMaps()
	# data_dir='../data/tfd/train_test_files';
	# file_list=[os.path.join(data_dir,'test_ckLabels_4.txt')];
	# new_files=[file_curr[:file_curr.rindex('.')]+'_withAnno.txt' for file_curr in file_list];	
	# writeAnnoFile(file_list,new_files)

	# saveHappyNeutralTrainTest()
	# writeTrainTestFilesWithAnno()


	
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
