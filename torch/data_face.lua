
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
        self.angles={-20,20};
        self.pixel_augment={0.5,1.5};
        self.scale={0.75,1.25};
        self.translate={-10,10};
        
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
        print (#self.lines_face);
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

    -- function data:rotateIm(img_face,angles)
    --     local angle = math.random(angles[1],angles[2])
    --     angle=math.rad(angle)
    --     img_face=image.rotate(img_face,angle,"bilinear");
    --     return img_face
    -- end

    

    function data:processIm(img_face)
        img_face:mul(255);

        if img_face:size(2)~=self.input_size[1] then 
            img_face = image.scale(img_face,self.input_size[1],self.input_size[2]);
        end

        
        -- img_face=img_face-self.mean_im;
        if self.augmentation then
            local rand=math.random(2);
            -- local rand=1;
            if rand==1 then
                image.hflip(img_face,img_face);
            end

            rand=math.random(2);
            if rand==1 then
                local pixel_augment_curr = (math.random()*(self.pixel_augment[2]-self.pixel_augment[1]))+self.pixel_augment[1]
                img_face=img_face*pixel_augment_curr;
            end

            rand=math.random(2);
            if rand==1 then
                local angle = math.random(self.angles[1],self.angles[2])
                angle=math.rad(angle)
                img_face=image.rotate(img_face,angle,"bilinear");
            end

            rand=math.random(2);
            if rand==1 then
                -- translate
                local x_translate=math.random(self.translate[1],self.translate[2])
                local y_translate=math.random(self.translate[1],self.translate[2])
                img_face = image.translate(img_face, x_translate, y_translate);
            end
        
            rand=math.random(2);
            if rand==1 then
                -- scale
                local scale_curr = (math.random()*(self.scale[2]-self.scale[1]))+self.scale[1]
                local img_face_sc=image.scale(img_face,'*'..scale_curr);
                if scale_curr<1 then
                    local pos=math.floor((img_face:size(2)-img_face_sc:size(2))/2)+1
                    img_face=torch.zeros(img_face:size());
                    img_face[{{},{pos,pos+img_face_sc:size(2)-1},{pos,pos+img_face_sc:size(2)-1}}]=img_face_sc;
                else
                    local pos=math.floor((img_face_sc:size(2)-img_face:size(2))/2)+1
                    img_face=torch.zeros(img_face:size());
                    img_face=img_face_sc[{1,{pos,pos+img_face:size(2)-1},{pos,pos+img_face:size(2)-1}}];

                end
            end
        end

        img_face=torch.cdiv((img_face-self.mean_im),self.std_im);

        return img_face
    end

    function data:addTrainingData(training_set,batch_size,lines_face,start_idx_face)
        local list_idx=start_idx_face;
        local list_size=#lines_face;
        local curr_idx=1;
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