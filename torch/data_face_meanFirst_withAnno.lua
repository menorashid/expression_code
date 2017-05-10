
do  
    local data = torch.class('data_face')

    function data:__init(args)
        print ('MEAN FIRST');
        self.file_path=args.file_path;
        self.batch_size=args.batch_size;
        self.mean_file=args.mean_file;
        self.std_file=args.std_file;
        self.limit=args.limit;
        self.augmentation=args.augmentation;
        self.twoClass=args.twoClass;        
        self.numAnnos=args.numAnnos;        
        print ('self.augmentation',self.augmentation);

        self.start_idx_face=1;
        self.input_size={96,96};

        
        self.angles={-5,5};
        self.pixel_augment={0.5,1.5};
        self.scale={0.7,1.4};
        self.a_range = {0.25,4}
        self.b_range = {0.1,1};
        -- self.c_range = {-0.1,0.1};

        self.ratio_blur=args.ratio_blur;
        self.activation_upper=args.activation_upper;

        -- set blurring parameters and nets
        self.net=args.net;
        self.net_gb=args.net_gb;
        self.conv_size=args.conv_size
        if self.conv_size then
            self.gauss_layer,self.gauss_layer_small,self.up_layer,self.min_layer,self.max_layer=self:getHelperNets(2*self.conv_size+1,self.conv_size)
        end

        -- optimize flag
        self.optimize=args.optimize;

        if args.input_size then
            self.input_size=args.input_size
        end

        -- MODIFIED
        self.mean_im=image.load(self.mean_file);
        self.std_im=image.load(self.std_file);

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
        
        if self.optimize then
            print ('OPTIMIZING');
            self.training_images,self.training_labels,self.lines_face,self.training_annos = self:loadTrainingImages();
            print (self.training_images:size());
            print (self.training_labels:size());
        end

        print (#self.lines_face);
        
    end


    function data:loadTrainingImages()
        local im_all=torch.zeros(#self.lines_face,1,self.input_size[1]
            ,self.input_size[2]);
        local labels = torch.zeros(#self.lines_face);
        local lines_face={};
        local annos = torch.zeros(#self.lines_face,self.numAnnos,2);

        local curr_idx=0
        for line_idx=1,#self.lines_face do
            local img_path_face=self.lines_face[line_idx][1];
            local label_face=self.lines_face[line_idx][2];
            local status_img_face,img_face=pcall(image.load,img_path_face);
            if status_img_face then
                curr_idx=curr_idx+1;
                if img_face:size(2)~=self.input_size[1] then 
                    img_face = image.scale(img_face,self.input_size[1],self.input_size[2]);
                end
                im_all[curr_idx]=img_face;
                labels[curr_idx]=label_face;
                lines_face[curr_idx]=self.lines_face[line_idx];
                annos[curr_idx]=self:getTensorFromTable(self.lines_face[line_idx],3);
            end
        end
        return im_all,labels,lines_face,annos     
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

    function data:getGCamEtc(net,net_gb,layer_to_viz,batch_inputs,batch_targets,also_pred)

        net:zeroGradParameters();
        net_gb:zeroGradParameters();
        local outputs=net:forward(batch_inputs);
        local scores, pred_labels = torch.max(outputs, 2);
        pred_labels = pred_labels:type(batch_targets:type());
        pred_labels = pred_labels:view(batch_targets:size());
        local outputs_gb= net_gb:forward(batch_inputs);
        local doutput_gt =  utils.create_grad_input_batch(net.modules[#net.modules], batch_targets)
        local gcam_gt = utils.grad_cam_batch(net, layer_to_viz, doutput_gt);
        local gb_viz_gt = net_gb:backward(batch_inputs, doutput_gt)

        local gcam_pred,gb_viz_pred
        if also_pred then
            net:zeroGradParameters();
            net_gb:zeroGradParameters();
            local doutput_pred = utils.create_grad_input_batch(net.modules[#net.modules], pred_labels)
            gcam_pred = utils.grad_cam_batch(net, layer_to_viz, doutput_pred);
            gb_viz_pred = net_gb:backward(batch_inputs, doutput_pred)
        else
            gcam_pred=gcam_gt
            gb_viz_pred=gb_viz_gt
        end
        
        return {gcam_pred,gcam_gt},{gb_viz_pred,gb_viz_gt},pred_labels
        -- return {gcam_gt,gcam_gt},{gb_viz_gt,gb_viz_gt},pred_labels;
        
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



    function data:plotPoint(im,point,pointSize,color_curr)
        local x=point[1];
        local y=point[2];
        if x>im:size(3) or y>im:size(2) then
            return im;
        end
        -- local point=torch.Tensor(point);
        local starts=torch.round(point-pointSize/2);
        local ends=torch.round(point+pointSize/2);

        for x_curr=math.max(1,starts[1]),math.min(im:size(3),ends[1]) do
            for y_curr=math.max(1,starts[2]),math.min(im:size(2),ends[2]) do
                for idx_rgb=1,im:size(1) do
                    im[idx_rgb][y_curr][x_curr]=color_curr[idx_rgb];            
                end
            end
        end
        return im;

    end

    function data:buildBlurryBatch(pointSize,bin_blur)

        if self.net:type()~='torch.CudaTensor' then
            self.net=self.net:cuda();
        end
        
        local train_state=self.net.train;
        if train_state then
            self.net:evaluate();
        end

        -- build a batch of images mean preprocessed before augmentation
        self:getTrainingData();
        
        -- set cuda
        self.training_set.data = self.training_set.data:cuda();
        self.training_set.label = self.training_set.label:cuda();
        
        local batch_inputs = self.training_set.data;
        local batch_targets = self.training_set.label;
        local annos = self.training_set.anno;
        
        -- local preds=self.net:forward(batch_inputs:clone());
        -- local indices=torch.zeros(preds:size()):type(preds:type());
        -- indices[preds:gt(0)]=1;
        -- indices[preds:le(0)]=-1;
        -- indices=indices:view(batch_targets:size())
        local bin_keep=torch.ones(batch_targets:size())
        -- batch_targets:eq(indices)
        -- print (torch.sum(bin_keep)/batch_targets:nElement());
                
        local color=torch.zeros(1);
        for im_num =1,batch_inputs:size(1) do
            if bin_keep[im_num]==1 then
                local annos_curr=annos[im_num];
                local im_curr=batch_inputs[im_num];
                for anno_num=1,annos_curr:size(1) do
                    if bin_blur[anno_num]>0 then
                        im_curr=self:plotPoint(im_curr,annos_curr[anno_num],pointSize,color);
                    end
                end
            -- else
            --     print (im_num);
            end
        end

    end

    function data:getPointImportance(layer_to_viz,also_pred)
        -- local also_pred=false;

        local gauss_layer_small = self.gauss_layer_small;
        local up_layer = self.up_layer;
        local min_layer = self.min_layer;
        local max_layer = self.max_layer;
        
        local train_state=self.net.train;
        
        if self.net:type()~='torch.CudaTensor' then
            self.net=self.net:cuda();
        end
        
        if self.net_gb:type()~='torch.CudaTensor' then
            self.net_gb=self.net_gb:cuda();
        end
        
        if train_state then
            self.net:evaluate();
            self.net_gb:evaluate();
        end

        -- build a batch of images mean preprocessed before augmentation
        self:getTrainingData();
        
        -- set cuda
        self.training_set.data = self.training_set.data:cuda();
        self.training_set.label = self.training_set.label:cuda();
        local batch_inputs = self.training_set.data;
        local batch_targets = self.training_set.label;
        
        -- get gcam stuff
        local gcam_both,gb_viz_both,pred_labels = self:getGCamEtc(self.net,self.net_gb,layer_to_viz,batch_inputs,batch_targets,also_pred)
        
        local bin_keep=pred_labels:eq(batch_targets);
        -- print (torch.sum(bin_keep)/bin_keep:nElement());

        -- set input range 0-1
        local inputs_org=batch_inputs;
        local min_vals=self.min_layer:forward(inputs_org):clone();
        inputs_org=inputs_org-min_vals;
        local max_vals=self.max_layer:forward(inputs_org):clone();
        inputs_org:cdiv(max_vals);

        -- local im_blur_all,gcam_curr,gb_gcam_all,gb_gcam_org_all,gb_gcam_th_all=self:getMeanBlurredImages(inputs_org,min_vals,max_vals,gcam_both[2],gb_viz_both[2],activation_thresh,strategy,bin_keep)
        -- local im_blur_all_pred,gcam_curr_pred,gb_gcam_all_pred,gb_gcam_org_all_pred,gb_gcam_th_all_pred;
        -- if also_pred then
        --     im_blur_all_pred,gcam_curr_pred,gb_gcam_all_pred,gb_gcam_org_all_pred,gb_gcam_th_all_pred=self:getMeanBlurredImages(inputs_org,min_vals,max_vals,gcam_both[1],gb_viz_both[1],activation_thresh,strategy,bin_keep)
        -- end

        -- getMeanBlurredImages(inputs_org,min_vals,max_vals,gcam_curr,gb_viz_curr,activation_thresh,strategy,bin_keep);
        local idx_gcam=1;
        local gcam_curr=gcam_both[idx_gcam];
        local gb_viz_curr=gb_viz_both[idx_gcam];
        gcam_curr = up_layer:forward(gcam_curr):clone();
        local gb_gcam_org_all = torch.cmul(gb_viz_curr,gcam_curr);
        local gb_gcam_all = torch.abs(gb_gcam_org_all);
        gb_gcam_all:cdiv(max_layer:forward(gb_gcam_all:csub(min_layer:forward(gb_gcam_all))));
        gb_gcam_all=gauss_layer_small:forward(gb_gcam_all):clone();
        gb_gcam_all:cdiv(max_layer:forward(gb_gcam_all:csub(min_layer:forward(gb_gcam_all))));
        gb_gcam_all[gb_gcam_all:ne(gb_gcam_all)]=0;
        return gb_gcam_all

    end


    function data:shuffleLines(lines)
        local x=lines;
        local len=#lines;
        -- print (len);

        local shuffle = torch.randperm(len)
        
        local lines2={};
        for idx=1,len do
            lines2[idx]=x[shuffle[idx]];
        end
        return lines2;
    end


    function data:shuffleLinesOptimizing(lines,im,labels,annos)
        local x=lines;
        local len=#lines;

        local shuffle = torch.randperm(len)
        
        local im_2=im:clone();
        local labels_2=labels:clone();
        local annos_2=annos:clone();

        local lines2={};
        for idx=1,len do
            lines2[idx]=x[shuffle[idx]];
            im_2[idx]=im[shuffle[idx]];
            labels_2[idx]=labels[shuffle[idx]];
            annos_2[idx]=annos[shuffle[idx]];
        end
        im=0;
        labels=0;
        lines=0;
        return lines2,im_2,labels_2,annos_2;
    end

    function data:getTrainingData()
        local start_idx_face_before = self.start_idx_face
        self.training_set.data=torch.zeros(self.batch_size,1,self.input_size[1]
            ,self.input_size[2]);
        self.training_set.label=torch.zeros(self.batch_size);
        self.training_set.anno=torch.zeros(self.batch_size,self.numAnnos,2);
        self.training_set.input={};
        
        self.start_idx_face=self:addTrainingData(self.training_set,self.batch_size,self.start_idx_face)    

        if self.start_idx_face<=start_idx_face_before and self.augmentation then
            print ('shuffling data'..self.start_idx_face..' '..start_idx_face_before )
            if not self.optimize then
                self.lines_face=self:shuffleLines(self.lines_face);
            else
                self.lines_face,self.training_images,self.training_labels=self:shuffleLinesOptimizing(self.lines_face,self.training_images,self.training_labels,self.training_annos);
            end
        end
    end
    
    function data:readDataFile(file_path)
        local file_lines = {};
        for line in io.lines(file_path) do 
            local line_curr_split={}
            for i=1,2*self.numAnnos+1 do
                local start_idx, end_idx = string.find(line, ' ');
                local start_string=string.sub(line,1,start_idx-1);
                line=string.sub(line,end_idx+1,#line);
                line_curr_split[#line_curr_split+1]=start_string;
            end
            line_curr_split[#line_curr_split+1]=line;
            file_lines[#file_lines+1]=line_curr_split;
        end 
        return file_lines
    end


    function data:rotateImAndLabel(img_horse,label_horse)

        local angle_deg = (math.random()*(self.angles[2]-self.angles[1]))+self.angles[1]
        local angle=math.rad(angle_deg)
        img_horse=image.rotate(img_horse,angle,"bilinear");

        label_horse=label_horse/img_horse:size(2);
        label_horse=(label_horse*2)-1;
        -- print (torch.min(label_horse),torch.max(label_horse));

        local rotation_matrix=torch.zeros(2,2);
        rotation_matrix[1][1]=math.cos(angle);
        rotation_matrix[1][2]=math.sin(angle);
        rotation_matrix[2][1]=-1*math.sin(angle);
        rotation_matrix[2][2]=math.cos(angle);

        for i=1,label_horse:size(1) do
            local ans = rotation_matrix*torch.Tensor({label_horse[i][1],label_horse[i][2]}):view(2,1);
            label_horse[i][1]=ans[1][1];
            label_horse[i][2]=ans[2][1];
        end

        label_horse=(label_horse+1)/2*img_horse:size(2);
        -- print (torch.min(label_horse),torch.max(label_horse));
        return img_horse,label_horse
    end


    function data:augmentImage(img_face,annos) 

        -- MODIFIED
        
        local rand=math.random(2);
        -- local rand=1;
        if rand==1 then
            image.hflip(img_face,img_face);
            local anno_clone=annos:clone();
            -- print (annos)
            local len_annos=annos:size(1)
            for anno_idx=1,len_annos do
                local new_anno_idx=len_annos-anno_idx+1;
                -- print (anno_idx,new_anno_idx)
                
                annos[anno_idx][2]=anno_clone[new_anno_idx][2];
                annos[anno_idx][1]=img_face:size(3)-anno_clone[new_anno_idx][1];
            end
            -- print (annos)
        end
            
        img_face,annos=self:rotateImAndLabel(img_face,annos);



        local alpha = (math.random()*(self.scale[2]-self.scale[1]))+self.scale[1]
        local img_face_sc=image.scale(img_face,'*'..alpha);
        annos=annos*alpha;

        if alpha<1 then
            local pos=math.floor((img_face:size(2)-img_face_sc:size(2))/2)+1
            img_face=torch.zeros(img_face:size());
            img_face[{{},{pos,pos+img_face_sc:size(2)-1},{pos,pos+img_face_sc:size(2)-1}}]=img_face_sc;
            annos=annos+pos;
        else
            local pos=math.floor((img_face_sc:size(2)-img_face:size(2))/2)+1
            img_face=torch.zeros(img_face:size());
            img_face=img_face_sc[{1,{pos,pos+img_face:size(2)-1},{pos,pos+img_face:size(2)-1}}];
            annos=annos-pos
        end
        
        

        local delta = math.floor(torch.abs(alpha-1)*self.input_size[1]);
        local x_translate=math.random(-delta,delta)
        local y_translate=math.random(-delta,delta)
        img_face = image.translate(img_face, x_translate, y_translate);
        annos[{{},1}]=annos[{{},1}]+x_translate;
        annos[{{},2}]=annos[{{},2}]+y_translate;

        -- -- print (torch.min(img_face),torch.max(img_face));
        -- local min_im =torch.min(img_face);
        -- img_face=img_face-min_im;
        -- local max_im = torch.max(img_face);
        -- img_face=img_face/max_im;
        -- -- print (torch.min(img_face),torch.max(img_face));

        -- -- print (torch.min(img_face),torch.max(img_face))
        -- local a=(math.random()*(self.a_range[2]-self.a_range[1]))+self.a_range[1];
        -- local b=(math.random()*(self.b_range[2]-self.b_range[1]))+self.b_range[1];
        -- img_face = (torch.pow(img_face,a)*b)
        -- -- print (torch.min(img_face),torch.max(img_face));
        -- img_face[img_face:le(0)]=0;
        -- img_face[img_face:ge(1)]=1;

        -- img_face=img_face*max_im;
        -- img_face=img_face+min_im;
        -- print (torch.min(img_face),torch.max(img_face));
        -- assert (torch.min(img_face)==min_im);
        -- assert (torch.max(img_face)==(max_im-min_im))
        

        img_face[img_face:ne(img_face)]=0;
        
        return img_face,annos
    end

    function data:processImBatch(im_batch)
        -- MODIFIED
        im_batch=torch.cdiv((im_batch-self.mean_batch),self.std_batch);
        
        if self.augmentation then
            for img_face_num=1,im_batch:size(1) do
                im_batch[img_face_num]=self:augmentImage(im_batch[img_face_num]);
            end
        end
        
        return im_batch;
    end

    function data:processIm(img_face,annos)
        -- MODIFIED

        img_face=torch.cdiv((img_face-self.mean_im),self.std_im);        
        
        if img_face:size(2)~=self.input_size[1] then 
            img_face = image.scale(img_face,self.input_size[1],self.input_size[2]);
        end
        
        if self.augmentation then
            img_face,annos = self:augmentImage(img_face,annos);
        end

        return img_face,annos
    end

    function data:getTensorFromTable(table,start_idx)
        local start_idx=start_idx-1;
        local len_table=#table-start_idx;
        local end_idx=#table;
        assert (len_table%2==0);
        local row_map;
        if self.numAnnos==5 then
            row_map={1,5,3,2,4};
        else
            row_map={};
            for row_num=1,self.numAnnos do
                row_map[#row_map+1]=row_num;
            end
        end
        
        local tensor=torch.zeros(len_table/2,2);
        for i=1,len_table do
            local row=math.ceil(i/2);
            local column=(i+1)%2+1;
            tensor[row_map[row]][column]=tonumber(table[i+start_idx]);
        end

        return tensor;
    end

    function data:addTrainingData(training_set,batch_size,start_idx_face,curr_idx)
        local list_idx=start_idx_face;
        local list_size=#self.lines_face;
        -- local 
        if not curr_idx then
            curr_idx=1;
        end

        while curr_idx<= batch_size do
            local img_path_face,label_face,status_img_face,img_face;
            local anno_face;
            if self.optimize then
                status_img_face=true;
                img_path_face=self.lines_face[list_idx];
                label_face=self.training_labels[list_idx];
                img_face=self.training_images[list_idx]:clone();
                anno_face=self.training_annos[list_idx]:clone();
            else
                img_path_face=self.lines_face[list_idx][1];
                label_face=self.lines_face[list_idx][2];
                anno_face=self:getTensorFromTable(self.lines_face[list_idx],3);
                status_img_face,img_face=pcall(image.load,img_path_face);
            end
            
            if status_img_face then
                img_face,anno_face=self:processIm(img_face,anno_face)
                training_set.data[curr_idx]=img_face;
                if self.twoClass then
                    training_set.label[curr_idx]=tonumber(label_face);
                else
                    training_set.label[curr_idx]=tonumber(label_face)+1;
                    -- if tonumber(label_face)==-1 then
                    --     training_set.label[curr_idx]=1;
                    -- else
                    --     training_set.label[curr_idx]=2;
                    -- end
                end

                training_set.anno[curr_idx]=anno_face;

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