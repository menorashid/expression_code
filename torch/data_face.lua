
do  
    local data = torch.class('data_face')

    function data:__init(args)
        self.file_path=args.file_path;
        self.batch_size=args.batch_size;
        self.mean_file=args.mean_file;
        self.std_file=args.std_file;
        self.limit=args.limit;
        self.augmentation=args.augmentation;
        
        print ('self.augmentation',self.augmentation);

        self.start_idx_face=1;
        self.input_size={96,96};

        
        self.angles={-5,5};
        self.pixel_augment={0.5,1.5};
        self.scale={0.7,1.4};
        self.a_range = {0.25,4}
        self.b_range = {0.7,1.4};
        self.c_range = {-0.1,0.1};

        self.out_dir_diff=args.out_dir_diff;
        self.start_idx_face_blur=1;
        self.ratio_blur=args.ratio_blur;

        self.lines_face_diff={};

        -- if args.num_labels then
        --     self.num_labels=args.num_labels;
        -- else
        --     self.num_labels=8;
        -- end

        if args.input_size then
            self.input_size=args.input_size
        end

        self.mean_im=image.load(self.mean_file)*255;
        self.std_im=image.load(self.std_file)*255;
        self.training_set={};
        self.lines_face=self:readDataFile(self.file_path);
        
        if self.augmentation then
            self.lines_face =self:shuffleLines(self.lines_face);
        end

        if self.limit~=nil then
            local lines_face=self.lines_face;
            self.lines_face={};

            for i=1,self.limit do
                self.lines_face[#self.lines_face+1]=lines_face[i];
            end
        end
        
        self.lines_face_diff=self.lines_face;

        print (#self.lines_face);
    end

    function data:unMean(mean,std)
        if not mean then
            mean=self.mean_im:view(1,self.mean_im:size(1),self.mean_im:size(2),self.mean_im:size(3));
            mean=torch.repeatTensor(mean,self.training_set.data:size(1),1,1,1):type(self.training_set.data:type());
        end
        if not std then
            std=self.std_im:view(1,self.std_im:size(1),self.std_im:size(2),self.std_im:size(3));
            std=torch.repeatTensor(std,self.training_set.data:size(1),1,1,1):type(self.training_set.data:type());
        end
        local im =torch.cmul(self.training_set.data,std)+mean;
        return im,mean,std
    end 

    function data:getGCamEtc(net,net_gb,layer_to_viz,batch_inputs,batch_targets)
        net:zeroGradParameters();
        net_gb:zeroGradParameters();

        local outputs=net:forward(batch_inputs);
        local outputs_gb= net_gb:forward(batch_inputs);

        local scores, pred_labels = torch.max(outputs, 2);
        pred_labels = pred_labels:type(batch_targets:type());
        pred_labels = pred_labels:view(batch_targets:size());

        local doutput_pred = utils.create_grad_input_batch(net.modules[#net.modules], pred_labels)
        local gcam_pred = utils.grad_cam_batch(net, layer_to_viz, doutput_pred);
        net:zeroGradParameters();
        local doutput_gt =  utils.create_grad_input_batch(net.modules[#net.modules], batch_targets)
        local gcam_gt = utils.grad_cam_batch(net, layer_to_viz, doutput_gt);

        local gb_viz_pred = net_gb:backward(batch_inputs, doutput_pred)
        net_gb:zeroGradParameters();
        local gb_viz_gt = net_gb:backward(batch_inputs, doutput_gt)
        return {gcam_pred,gcam_gt},{gb_viz_pred,gb_viz_gt},pred_labels;
        
    end

    function data:saveDifficultImages(net,net_gb,layer_to_viz,activation_thresh,conv_size,logger)
        -- set augmentation off; set start idx;
        local aug_org=self.augmentation;
        local start_idx_face_org = self.start_idx_face;

        self.start_idx_face = 1;
        self.augmentation=false;
        -- empty difficult face lines;
        self.lines_face_diff={};
        -- set nets
        local train_state=net.train;
        net:evaluate();
        net_gb:evaluate();

        -- loop and save images;
        local epoch=math.ceil(#self.lines_face/self.batch_size);
        local rem= epoch*self.batch_size - #self.lines_face;

        local mean,std;
        local gauss_big = image.gaussian(2*conv_size+1,2*conv_size+1):float();
        local gauss =  image.gaussian(conv_size,conv_size):float();
        for epoch_num=1,epoch do
            if logger then
                local disp_str=string.format("saving blur images iter "..epoch_num .." of "..epoch .." "..activation_thresh)
                print (disp_str);
                logger:writeString(dump.tostring(disp_str)..'\n');
            end

            self:getTrainingData(0);
            local batch_inputs=self.training_set.data:cuda();
            local batch_targets=self.training_set.label:cuda();
            local gcam_all,gb_viz_all,pred_labels = self:getGCamEtc(net,net_gb,layer_to_viz,batch_inputs,batch_targets);

            local im;
            if not mean then
                im,mean,std = self:unMean();
            else
                assert (std)
                im=self:unMean(mean,std);
            end
            
            for im_num=1,batch_targets:size(1) do
                
                if (epoch_num-1)*self.batch_size +im_num > #self.lines_face then
                    break;
                end

                -- if pred_labels[im_num]==batch_targets[im_num] then
                    local gb_viz=gb_viz_all[2][im_num][1]:float();
                    local gcam=gcam_all[2][im_num][1]:float();
                    local im_org=im[im_num][1]:div(255):float();
                    local path_org=self.training_set.input[im_num];
                    gcam=image.scale(gcam, self.input_size[1], self.input_size[2]);
                    local gb_gcam=torch.cmul(gcam,gb_viz);                  
                    -- gb_gcam=torch.sum(gb_gcam,1);
                    gb_gcam=torch.abs(gb_gcam);
                    gb_gcam=gb_gcam-torch.min(gb_gcam);
                    gb_gcam:div(torch.max(gb_gcam));
                    local im_g = image.convolve(im_org,gauss_big,'same');
                        -- torch.ones(2*conv_size,2*conv_size):float(),'same');
                    im_g:div(torch.max(im_g));

                    local gb_gcam_vals=torch.sort(gb_gcam:view(-1),1,true);
                    local idx=math.floor(gb_gcam_vals:size(1)*activation_thresh);
                    local gb_gcam_th =torch.zeros(gb_gcam:size()):type(gb_gcam:type());
                    gb_gcam_th[gb_gcam:ge(gb_gcam_vals[idx])]=1;
                    gb_gcam_th[gb_gcam:le(gb_gcam_vals[idx])]=0;
                    gb_gcam_th = image.convolve(gb_gcam_th,gauss,'same');
                        -- torch.ones(conv_size,conv_size):float(),'same');
                    gb_gcam_th:div(torch.max(gb_gcam_th));
                    -- gb_gcam_th[gb_gcam_th:gt(0.5)]=1;


                    local im_blur=torch.cmul(gb_gcam_th,im_g)+torch.cmul((1-gb_gcam_th),im_org);
                    local out_file_blur = paths.concat(self.out_dir_diff,paths.basename(path_org));
                    image.save(out_file_blur,image.toDisplayTensor(im_blur));
                    self.lines_face_diff[#self.lines_face_diff+1]={out_file_blur,''..batch_targets[im_num]-1};
                -- end
            end
        end

        -- get rid of repeated images 
        -- if #self.lines_face_diff~=#self.lines_face then
        --     local lines_face=self.lines_face_diff;
        --     self.lines_face_diff={};

        --     for i=1,#self.lines_face do
        --         self.lines_face_diff[#self.lines_face_diff+1]=lines_face[i];
        --     end
        -- end

        -- set augmentation back and start idx back;
        self.augmentation=aug_org;
        self.start_idx_face = start_idx_face_org;
        self.start_idx_face_blur=1;
        -- shuffle lines_diff if needed
        if self.augmentation then
            self.lines_face_diff=self:shuffleLines(self.lines_face_diff);
        end
        print (#self.lines_face_diff)
        -- set nets back
        if train_state then
            net:training();
            net_gb:training();
        end

    end

    function data:shuffleLines(lines)
        local x=lines;
        local len=#lines;

        local shuffle = torch.randperm(len)
        
        local lines2={};
        for idx=1,len do
            lines2[idx]=x[shuffle[idx]];
        end
        return lines2;
    end

    function data:getTrainingData(ratio_blur)
        
        if not ratio_blur then
            if self.ratio_blur then
                ratio_blur=self.ratio_blur;
            else
                ratio_blur=0;
            end
        end

        local num_blur=math.min(math.ceil(self.batch_size*ratio_blur),#self.lines_face_diff);
        local start_blur=self.batch_size-num_blur;
        -- print ('ratio_blur',ratio_blur,'num_blur',num_blur,'start_blur',start_blur);
        -- print ('sizes of lists',#self.lines_face,#self.lines_face_diff);

        local start_idx_face_before = self.start_idx_face
        local start_idx_face_blur_before = self.start_idx_face_blur;
        -- print ('start_idx_face_before,start_idx_face_blur_before',start_idx_face_before,start_idx_face_blur_before);

        self.training_set.data=torch.zeros(self.batch_size,1,self.input_size[1]
            ,self.input_size[2]);
        self.training_set.label=torch.zeros(self.batch_size);
        self.training_set.input={};
        
        self.start_idx_face=self:addTrainingData(self.training_set,start_blur,
            self.lines_face,self.start_idx_face,1)    

        -- print ('self.batch_size,self.lines_face_diff,self.start_idx_face_blur,start_blur+1',
            -- self.batch_size,#self.lines_face_diff,self.start_idx_face_blur,start_blur+1)
        self.start_idx_face_blur=self:addTrainingData(self.training_set,self.batch_size,
            self.lines_face_diff,self.start_idx_face_blur,start_blur+1)
        
        -- print ('start_idx_face,start_idx_face_blur',self.start_idx_face,self.start_idx_face_blur);

        if self.start_idx_face<start_idx_face_before and self.augmentation then
            -- print ('shuffling data'..self.start_idx_face..' '..start_idx_face_before )
            self.lines_face=self:shuffleLines(self.lines_face);
        end

        if self.start_idx_face_blur<start_idx_face_blur_before and self.augmentation then
            -- print ('shuffling data blur'..self.start_idx_face_blur..' '..start_idx_face_blur_before )
            self.lines_face_diff=self:shuffleLines(self.lines_face_diff);
        end

    end

    
    function data:readDataFile(file_path)
        local file_lines = {};
        for line in io.lines(file_path) do 
            local start_idx, end_idx = string.find(line, ' ');
            local img_path=string.sub(line,1,start_idx-1);
            local img_label=string.sub(line,end_idx+1,#line);
            file_lines[#file_lines+1]={img_path,img_label};
        end 
        return file_lines

    end

    -- function data:rotateIm(img_face,angles)
    --     local angle = math.random(angles[1],angles[2])
    --     angle=math.rad(angle)
    --     img_face=image.rotate(img_face,angle,"bilinear");
    --     return img_face
    -- end

    

    function data:processIm(img_face)
        

        if img_face:size(2)~=self.input_size[1] then 
            img_face = image.scale(img_face,self.input_size[1],self.input_size[2]);
        end

        
        -- img_face=img_face-self.mean_im;
        if self.augmentation then
            local rand=math.random(2);
            if rand==1 then
                image.hflip(img_face,img_face);
            end
            
            local angle_deg = (math.random()*(self.angles[2]-self.angles[1]))+self.angles[1]
            local angle=math.rad(angle_deg)
            img_face=image.rotate(img_face,angle,"bilinear");

            local alpha = (math.random()*(self.scale[2]-self.scale[1]))+self.scale[1]
            local img_face_sc=image.scale(img_face,'*'..alpha);
            if alpha<1 then
                local pos=math.floor((img_face:size(2)-img_face_sc:size(2))/2)+1
                img_face=torch.zeros(img_face:size());
                img_face[{{},{pos,pos+img_face_sc:size(2)-1},{pos,pos+img_face_sc:size(2)-1}}]=img_face_sc;
            else
                local pos=math.floor((img_face_sc:size(2)-img_face:size(2))/2)+1
                img_face=torch.zeros(img_face:size());
                img_face=img_face_sc[{1,{pos,pos+img_face:size(2)-1},{pos,pos+img_face:size(2)-1}}];
            end
            -- print (alpha,img_face_sc:size(2));

            local delta = math.floor(torch.abs(alpha-1)*self.input_size[1]);
            local x_translate=math.random(-delta,delta)
            local y_translate=math.random(-delta,delta)
            img_face = image.translate(img_face, x_translate, y_translate);
            -- print (alpha,delta,x_translate,y_translate);

            local a=(math.random()*(self.a_range[2]-self.a_range[1]))+self.a_range[1];
            local b=(math.random()*(self.b_range[2]-self.b_range[1]))+self.b_range[1];
            local c=(math.random()*(self.c_range[2]-self.c_range[1]))+self.c_range[1];
            img_face = (torch.pow(img_face,a)*b) +c
            -- print (a,b,c);
        end

        img_face:mul(255);
        img_face=torch.cdiv((img_face-self.mean_im),self.std_im);

        return img_face
    end

    function data:addTrainingData(training_set,batch_size,lines_face,start_idx_face,curr_idx)
        local list_idx=start_idx_face;
        local list_size=#lines_face;
        -- local 
        if not curr_idx then
            curr_idx=1;
        end

        while curr_idx<= batch_size do
            local img_path_face=lines_face[list_idx][1];
            local label_face=lines_face[list_idx][2];
            
            local status_img_face,img_face=pcall(image.load,img_path_face);
            
            if status_img_face then
                img_face=self:processIm(img_face)
                training_set.data[curr_idx]=img_face;
                training_set.label[curr_idx]=tonumber(label_face)+1;
                training_set.input[curr_idx]=img_path_face;
            else
                print ('PROBLEM READING INPUT');
                curr_idx=curr_idx-1;
            end
            list_idx=(list_idx%list_size)+1;
            curr_idx=curr_idx+1;
        end
        return list_idx;
    end

    
end

return data