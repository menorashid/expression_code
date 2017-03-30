
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
        self.activation_upper=args.activation_upper;
        -- self.lines_face_diff={};

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

        self.mean_batch=self.mean_im:view(1,self.mean_im:size(1),self.mean_im:size(2),self.mean_im:size(3));
        self.mean_batch=torch.repeatTensor(self.mean_im,self.batch_size,1,1,1);

        self.std_batch=self.std_im:view(1,self.std_im:size(1),self.std_im:size(2),self.std_im:size(3));
        self.std_batch=torch.repeatTensor(self.std_im,self.batch_size,1,1,1);

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
        
        -- self.lines_face_diff=self.lines_face;

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
        -- im[im:ne(im)]=0;
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

    function data:getHelperNets(conv_size_blur,conv_size_mask)
        -- build helper nets for blurring and min maxing with cuda
        local gauss_big = image.gaussian({height=conv_size_blur,width=conv_size_blur,normalize=true}):cuda();
        local gauss_layer= nn.SpatialConvolution(1,1,conv_size_blur,conv_size_blur,
                                                1,1,(conv_size_blur-1)/2,(conv_size_blur-1)/2):cuda();
        gauss_layer.weight=gauss_big:view(1,1,gauss_big:size(1),gauss_big:size(2)):clone()
        gauss_layer.bias:fill(0);
        
        local gauss =  image.gaussian({height=conv_size_mask,width=conv_size_mask,normalize=true}):cuda();
        local gauss_layer_small = nn.SpatialConvolution(1,1,conv_size_mask,conv_size_mask,
                                                1,1,(conv_size_mask-1)/2,(conv_size_mask-1)/2):cuda();
        gauss_layer_small.weight = gauss:view(1,1,gauss:size(1),gauss:size(2)):clone()
        gauss_layer_small.bias:fill(0);
        
        local up_layer = nn.SpatialUpSamplingBilinear({oheight=self.input_size[1],owidth=self.input_size[2]}):cuda();

        local min_layer = nn.Sequential();
        min_layer:add(nn.View(-1):setNumInputDims(3));
        min_layer:add(nn.Min(1,1));
        min_layer:add(nn.View(1));
        min_layer:add(nn.Replicate(self.input_size[1],2,3));
        min_layer:add(nn.Replicate(self.input_size[1],3,3));
        min_layer = min_layer:cuda();

        local max_layer = min_layer:clone();
        max_layer:remove(2);
        max_layer:insert(nn.Max(1,1),2);
        max_layer:cuda();
        
        return gauss_layer,gauss_layer_small,up_layer,min_layer,max_layer;
    end


    function data:buildBlurryBatch(net,net_gb,layer_to_viz,activation_thresh,conv_size,strategy,out_file_pre)
        
        -- set nets mode
        local train_state=net.train;
        net:evaluate();
        net_gb:evaluate();

        -- build a batch of images 0 255 mean subtracted
        self:getTrainingData();

        -- set cuda
        self.training_set.data = self.training_set.data:cuda();
        self.training_set.label = self.training_set.label:cuda();
        local batch_inputs = self.training_set.data;
        local batch_targets = self.training_set.label;
        self.mean_batch = self.mean_batch:cuda();
        self.std_batch = self.std_batch:cuda();

        -- get helper nets
        local gauss_layer,gauss_layer_small,up_layer,min_layer,max_layer = self:getHelperNets(2*conv_size+1,conv_size);
        
        -- get gcam stuff
        local gcam_both,gb_viz_both,pred_labels = self:getGCamEtc(net,net_gb,layer_to_viz,batch_inputs,batch_targets)

        -- set input range between 0-1 again
        local inputs_org=self:unMean(self.mean_batch,self.std_batch):div(255)
        -- create input with blur
        local inputs_blur=gauss_layer:forward(inputs_org);
        inputs_blur:cdiv(max_layer:forward(inputs_blur:csub(min_layer:forward(inputs_blur))));
        
        -- get activations
        local gcam_curr = gcam_both[2];
        local gb_viz_curr = gb_viz_both[2]
        gcam_curr = up_layer:forward(gcam_curr);
        local gb_gcam_org_all = torch.cmul(gb_viz_curr,gcam_curr);
        local gb_gcam_all = torch.abs(gb_gcam_org_all);
        gb_gcam_all:cdiv(max_layer:forward(gb_gcam_all:csub(min_layer:forward(gb_gcam_all))));
        
        local gb_gcam_th_all =torch.zeros(gb_gcam_all:size()):type(gb_gcam_all:type());
        local vals_all = gb_gcam_th_all:clone();
    
        local im_num_end_blur=inputs_org:size(1);
        local max_val = torch.max(gb_gcam_all);
        if self.ratio_blur then
            im_num_end_blur=inputs_org:size(1)*self.ratio_blur;
        end
        

        for im_num =1, inputs_org:size(1) do
            if im_num<=im_num_end_blur then
                local activation_thresh_curr;
                
                if strategy=='ncl' then
                    activation_thresh_curr=activation_thresh;
                elseif strategy=='mix' then
                    activation_thresh_curr=torch.uniform(0,self.activation_upper);
                elseif strategy=='mixcl' then
                    if im_num<(im_num_end_blur/2) then
                        activation_thresh_curr=torch.uniform(0,self.activation_upper);
                        
                    else
                        activation_thresh_curr=activation_thresh;
                    end
                end
                
                local gb_gcam = gb_gcam_all[im_num][1];
                local gb_gcam_vals = torch.sort(gb_gcam:view(-1),1,true);
                local val = gb_gcam_vals[math.floor(gb_gcam_vals:size(1)*activation_thresh_curr)];
                vals_all[im_num][1]:fill(val);
            else
                vals_all[im_num][1]:fill(max_val+1);
            end
        end
        
        -- create masks and blur
        gb_gcam_th_all[gb_gcam_all:ge(vals_all)]=1;
        gb_gcam_th_all[gb_gcam_all:lt(vals_all)]=0;
        gb_gcam_th_all=gauss_layer_small:forward(gb_gcam_th_all);
        gb_gcam_th_all:cdiv(max_layer:forward(gb_gcam_th_all:csub(min_layer:forward(gb_gcam_th_all))));
        gb_gcam_th_all[gb_gcam_th_all:ne(gb_gcam_th_all)]=0;
        -- blur specific parts in input image
        local im_blur_all=torch.cmul(gb_gcam_th_all,inputs_blur)+torch.cmul((1-gb_gcam_th_all),inputs_org);
        
        -- mean subtract
        im_blur_all:mul(255);
        im_blur_all=torch.cdiv((im_blur_all-self.mean_batch),self.std_batch);
        
        self.training_set.data = im_blur_all;
        self.training_set.label = batch_targets;

        if out_file_pre then
            im_blur_all=self:unMean(self.mean_batch,self.std_batch);
            print ('gb_gcam_org_all',torch.min(gb_gcam_org_all),torch.max(gb_gcam_org_all));
            print ('inputs_org',torch.min(inputs_org),torch.max(inputs_org));
            print ('inputs_blur',torch.min(inputs_blur),torch.max(inputs_blur));
            print ('gb_gcam_th_all',torch.min(gb_gcam_th_all),torch.max(gb_gcam_th_all));
            print ('im_blur_all',torch.min(im_blur_all),torch.max(im_blur_all));
            
            for im_num=1,inputs_org:size(1) do
                local out_file_org=out_file_pre..im_num..'_org.jpg';
                local out_file_gb_gcam=out_file_pre..im_num..'_gb_gcam.jpg';
                local out_file_hm=out_file_pre..im_num..'_hm.jpg';
                local out_file_gb_gcam_org=out_file_pre..im_num..'_gb_gcam_org.jpg';
                local out_file_gb_gcam_th=out_file_pre..im_num..'_gb_gcam_th.jpg';
                local out_file_g = out_file_pre..im_num..'_gaussian.jpg';
                local out_file_blur = out_file_pre..im_num..'_blur.jpg';
                
                local gcam=gcam_curr[im_num]
                local gb_gcam = gb_gcam_all[im_num];
                local gb_gcam_org = gb_gcam_org_all[im_num]
                local hm = utils.to_heatmap(gcam:float())
                local im_org = inputs_org[im_num][1];
                local im_g = inputs_blur[im_num][1];
                local gb_gcam_th = gb_gcam_th_all[im_num][1];
                local im_blur = im_blur_all[im_num][1]

                
                image.save(out_file_blur,image.toDisplayTensor(im_blur));
                image.save(out_file_gb_gcam_th, image.toDisplayTensor(gb_gcam_th))
                image.save(out_file_gb_gcam, image.toDisplayTensor(gb_gcam))
                image.save(out_file_gb_gcam_org, image.toDisplayTensor(gb_gcam_org))
                image.save(out_file_hm, image.toDisplayTensor(hm))
                image.save(out_file_org, image.toDisplayTensor(im_org))
                image.save(out_file_g,image.toDisplayTensor(im_g));
            end
        end

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

    function data:getTrainingData()
        
        local start_idx_face_before = self.start_idx_face
        
        self.training_set.data=torch.zeros(self.batch_size,1,self.input_size[1]
            ,self.input_size[2]);
        self.training_set.label=torch.zeros(self.batch_size);
        self.training_set.input={};
        
        self.start_idx_face=self:addTrainingData(self.training_set,self.batch_size,
            self.lines_face,self.start_idx_face)    

        if self.start_idx_face<start_idx_face_before and self.augmentation then
            print ('shuffling data'..self.start_idx_face..' '..start_idx_face_before )
            self.lines_face=self:shuffleLines(self.lines_face);
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

    function data:augmentImage(img_face) 

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
        -- if (torch.sum(img_face:eq(img_face))~=img_face:nElement()) then
        --     print (torch.sum(img_face:ne(img_face)),img_face:nElement(),rand,angle_deg,alpha,delta,a,b,c)
        -- end
        img_face[img_face:ne(img_face)]=0;
        return img_face;
    end

    function data:processImBatch(im_batch)
        
        if self.augmentation then
            for img_face_num=1,im_batch:size(1) do
                im_batch[img_face_num]=self:augmentImage(im_batch[img_face_num]);
            end
        end
        
        im_batch:mul(255);
        im_batch=torch.cdiv((im_batch-self.mean_batch),self.std_batch);
        
        return im_batch;
    end

    function data:processIm(img_face)
        

        if img_face:size(2)~=self.input_size[1] then 
            img_face = image.scale(img_face,self.input_size[1],self.input_size[2]);
        end
        
        -- img_face=img_face-self.mean_im;
        if self.augmentation then
            img_face = self:augmentImage(img_face);
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