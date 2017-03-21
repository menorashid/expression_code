require 'torch'
require 'nn'
require 'lfs'
require 'image'
require 'loadcaffe'
utils = require 'misc.utils'

cmd = torch.CmdLine()
cmd:text('Options')

-- Model parameters
cmd:option('-proto_file', 'models/VGG_ILSVRC_16_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_16_layers.caffemodel')
cmd:option('-input_sz', 224, 'Input image dimensions (use 227 for AlexNet)')
cmd:option('-backend', 'nn')

-- Grad-CAM parameters
cmd:option('-layer_name', 'relu5_3', 'Layer to use for Grad-CAM (use relu5_4 for VGG-19 and relu5 for AlexNet)')
cmd:option('-input_image_path', '../expression_code/data/ck_96/im/S035/006/S035_006_00000016.png', 'Input image path')
cmd:option('-mean_im_path', '../expression_code/data/ck_96/train_test_files/train_0_mean.png', 'Input image path')
cmd:option('-std_im_path', '../expression_code/data/ck_96/train_test_files/train_0_mean.png', 'Input image path')

cmd:option('-output_image_name', '', 'Output image name')
cmd:option('-label',-1, 'Class label to generate grad-CAM for (-1 = use predicted class, 283 = Tiger cat, 243 = Boxer)')
cmd:option('-save_as_heatmap', 1, 'Whether to save heatmap or raw Grad-CAM. 1 = save heatmap, 0 = save raw Grad-CAM.')

-- Miscellaneous
cmd:option('-seed', 123, 'Torch manual random number generator seed')
cmd:option('-gpuid', 0, '0-indexed id of GPU to use. -1 = CPU')
cmd:option('-out_path', 'output/', 'Output path')

-- Parse command-line parameters
opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')
lfs.mkdir(opt.out_path)

if opt.gpuid >= 0 then
  require 'cunn'
  require 'cutorch'
  cutorch.setDevice(opt.gpuid + 1)
  cutorch.manualSeed(opt.seed)
end

-- Load CNN
-- local cnn = loadcaffe.load(opt.proto_file, opt.model_file, opt.backend)
print (opt.model_file)
local cnn=torch.load(opt.model_file);
print (cnn)
-- for i=1,#cnn do
--   print (cnn:get(i).name)
-- end
-- Set to evaluate and remove softmax layer
cnn:evaluate()
-- to remove
-- cnn:remove()

-- Clone & replace ReLUs for Guided Backprop
local cnn_gb = cnn:clone()
cnn_gb:replace(utils.guidedbackprop)
-- print (cnn_gb);

-- Load image
-- to modify
local img = utils.preprocess_us(opt.input_image_path, opt.mean_im_path, opt.std_im_path)

-- Transfer to GPU
if opt.gpuid >= 0 then
  cnn:cuda()
  cnn_gb:cuda()
  img = img:cuda()
else
  img = img:float()
end

-- Forward pass
local output = cnn:forward(img)
local output_gb = cnn_gb:forward(img)

-- Take argmax
local score, pred_label = torch.max(output,1)
print ('pred_label',pred_label,opt.label);
if opt.label == -1 then
  print("No label provided, using predicted label ", pred_label)
  opt.label = pred_label[1]
end

-- Set gradInput
local doutput = utils.create_grad_input(cnn.modules[#cnn.modules], opt.label)

-- Grad-CAM
local gcam = utils.grad_cam(cnn, opt.layer_name, doutput)
gcam = image.scale(gcam:float(), opt.input_sz, opt.input_sz)

local hm = utils.to_heatmap(gcam)

if opt.output_image_name == "" then
  opt.output_image_name = opt.label
end

if opt.save_as_heatmap == 1 then
  image.save(opt.out_path .. 'classify_gcam_hm_' .. opt.output_image_name .. '.png', image.toDisplayTensor(hm))
else
  image.save(opt.out_path .. 'classify_gcam_' .. opt.output_image_name .. '.png', image.toDisplayTensor(gcam))
end

-- Guided Backprop
local gb_viz = cnn_gb:backward(img, doutput)
-- BGR to RGB
gb_viz = gb_viz:index(1, torch.LongTensor{3, 2, 1})
image.save(opt.out_path .. 'classify_gb_' .. opt.output_image_name .. '.png', image.toDisplayTensor(gb_viz))

-- Guided Grad-CAM
local gb_gcam = gb_viz:float():cmul(gcam:expandAs(gb_viz))
gb_gcam=torch.sum(gb_gcam,1);
-- gb_gcam=gb_gcam-torch.min(gb_gcam);
-- gb_gcam:div(torch.max(gb_gcam));
-- gb_gcam=torch.pow(gb_gcam,5);

gb_gcam=torch.abs(torch.pow(gb_gcam,1));
gb_gcam=gb_gcam-torch.min(gb_gcam);
gb_gcam:div(torch.max(gb_gcam));
        

print (gb_gcam:size());
print (torch.mean(gb_gcam),torch.min(gb_gcam),torch.max(gb_gcam));
-- gb_gcam = utils.to_heatmap(gb_gcam)
image.save(opt.out_path .. 'classify_gb_gcam_' .. opt.output_image_name .. '.png', image.toDisplayTensor(gb_gcam))
