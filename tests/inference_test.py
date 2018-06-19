from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
import numpy as np
import urllib
import cv2
import os, time, getopt, sys, re
from caffe2.python import core, workspace, models
import urllib2
import operator
import caffe2.python._import_c_extension as C
import pdb

def url_to_image(url):
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    # Get original aspect ratio
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = cv2.resize(img, (res, input_width))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = cv2.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = cv2.resize(img, (input_width, input_height))
    return imgScaled



opts, args = getopt.getopt(sys.argv[1:], 'm:s:e:')

model = ''
img_size = 0
use_gpu = 0

for opt, arg in opts:
    if opt == '-m': 
        model = arg
    elif opt == '-s':
        img_size = int(arg)
    elif opt == '-e':
        use_gpu_engine = int(arg)

if model == '' or img_size == 0:
    print('Invalid command line argument')
    print('Usage:')
    print('-m <model directory> -s <input image size> -g [Use GPU option -- 0: None, 1: MIOPEN]')
    print('Model directory must contain both the init_net.pb and predict_net.pb, and optionally ilsvrc_2012_mean.npy files')
    exit()

# set paths and variables from model choice and prep image
model = os.path.expanduser(model)

# mean can be 128 or custom based on the model
# gives better results to remove the colors found in all of the training images
mean_file = os.path.join(model, 'ilsvrc_2012_mean.npy')

if not os.path.exists(mean_file):
    print("No mean file found!")
    mean = 128
else:
    print ("Mean file found!")
    mean = np.load(mean_file).mean(1).mean(1)
    mean = mean[:, np.newaxis, np.newaxis]

# make sure all of the files are around...
INIT_NET_PB = os.path.join(model, 'init_net.pb')
PREDICT_NET_PB = os.path.join(model, 'predict_net.pb')

# Check to see if the files exist
if not os.path.exists(INIT_NET_PB):
    print("WARNING: " + INIT_NET_PB + " not found!")
else:
    if not os.path.exists(PREDICT_NET_PB):
        print("WARNING: " + PREDICT_NET_PB + " not found!")

# Load the image as a 32-bit float
#    Note: skimage.io.imread returns a HWC ordered RGB image of some size
IMAGE_LOCATION =  "https://cdn.pixabay.com/photo/2015/02/10/21/28/flower-631765_1280.jpg"
img = url_to_image(IMAGE_LOCATION)
print("Original Image Shape: " , img.shape)
# Rescale the image to comply with our desired input size. This will not make the image 227x227
#    but it will make either the height or width 227 so we can get the ideal center crop.
img = rescale(img, img_size, img_size)
print("Image Shape after rescaling: " , img.shape)

img = crop_center(img, img_size, img_size)
print("Image Shape after cropping: " , img.shape)

# switch to CHW (HWC --> CHW)
img = img.swapaxes(1, 2).swapaxes(0, 1)
print("CHW Image Shape: " , img.shape)

img = img/255.0

# remove mean for better results
#img = img * 255 - mean

# add batch size axis which completes the formation of the NCHW shaped input that we want
img = img[np.newaxis, :, :, :].astype(np.float32)

print("NCHW image (ready to be used as input): ", img.shape)


print('Running on CPU......')

device_opts = caffe2_pb2.DeviceOption()
device_opts.device_type = caffe2_pb2.CPU
with open(INIT_NET_PB) as f:
    init_net = f.read()
with open(PREDICT_NET_PB) as f:
    predict_net = f.read()
p = workspace.Predictor(init_net, predict_net)

# run the net and return prediction
results_cpu = p.run({'gpu_0/data': img})
results_cpu = np.asarray(results_cpu)

print("======================================")   


print('Running on HIP..........')
device_opts = caffe2_pb2.DeviceOption()
device_opts.device_type = caffe2_pb2.HIP
device_opts.hip_gpu_id = 0

init_def = caffe2_pb2.NetDef()
with open(INIT_NET_PB, 'rb') as f:
    init_def.ParseFromString(f.read())
    init_def.device_option.CopyFrom(device_opts)

net_def = caffe2_pb2.NetDef()
with open(PREDICT_NET_PB, 'rb') as f:
    net_def.ParseFromString(f.read())
    net_def.device_option.CopyFrom(device_opts)

init_net = core.Net(init_def)
predict_net = core.Net(net_def)
if use_gpu_engine == 1:
    print('Using MIOPEN')
    init_net.RunAllOnGPU(use_gpu_engine=True)
    predict_net.RunAllOnGPU(use_gpu_engine=True)

for op in init_net.Proto().op:
    op.device_option.CopyFrom(device_opts)
for op in predict_net.Proto().op:
    op.device_option.CopyFrom(device_opts)
        
workspace.FeedBlob('gpu_0/data',img, device_option=device_opts)
workspace.RunNetOnce(init_net)
workspace.CreateNet(predict_net)
workspace.RunNet(predict_net)
results_gpu = workspace.FetchBlob('gpu_0/softmax')
results_gpu = np.asarray(results_gpu)

preds_cpu = np.squeeze(results_cpu)
preds_gpu = np.squeeze(results_gpu)

preds_cpu_norm = np.linalg.norm(preds_cpu, ord=None)
preds_gpu_norm = np.linalg.norm(preds_gpu, ord=None)


# Get the prediction and the confidence by finding the maximum value and index of maximum value in preds array
print("=================== cpu output =================")
preds = preds_cpu
results = results_cpu
curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
print("Prediction: ", curr_pred)
print("Confidence: ", curr_conf)

# the rest of this is digging through the results 
results = np.delete(results, 1)
index = 0
highest = 0
arr = np.empty((0,2), dtype=object)
arr[:,0] = int(10)
arr[:,1:] = float(10)
for i, r in enumerate(results):
    # imagenet index begins with 1!
    i=i+1
    arr = np.append(arr, np.array([[i,r]]), axis=0)
    if (r > highest):
        highest = r
        index = i 

# top N results
N = 5
topN = sorted(arr, key=lambda x: x[1], reverse=True)[:N]
print("Raw top {} results: {}".format(N,topN))

# Isolate the indexes of the top-N most likely classes
topN_inds = [int(x[0]) for x in topN]
print("Top {} classes in order: {}".format(N,topN_inds))

# Now we can grab the code list and create a class Look Up Table
codes =  "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"
response = urllib2.urlopen(codes)
class_LUT = []
for line in response:
    code, result = line.partition(":")[::2]
    code = code.strip()
    result = result.replace("'", "")
    if code.isdigit():
        class_LUT.append(result.split(",")[0][1:])
        
# For each of the top-N results, associate the integer result with an actual class
for n in topN:
    print("Model predicts '{}' with {}% confidence".format(class_LUT[int(n[0])],float("{0:.2f}".format(n[1]*100))))


print("=================== gpu output =================")
preds = preds_gpu
results = results_gpu
curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
print("Prediction: ", curr_pred)
print("Confidence: ", curr_conf)

# the rest of this is digging through the results 
results = np.delete(results, 1)
index = 0
highest = 0
arr = np.empty((0,2), dtype=object)
arr[:,0] = int(10)
arr[:,1:] = float(10)
for i, r in enumerate(results):
    # imagenet index begins with 1!
    i=i+1
    arr = np.append(arr, np.array([[i,r]]), axis=0)
    if (r > highest):
        highest = r
        index = i 

# top N results
N = 5
topN = sorted(arr, key=lambda x: x[1], reverse=True)[:N]
print("Raw top {} results: {}".format(N,topN))

# Isolate the indexes of the top-N most likely classes
topN_inds = [int(x[0]) for x in topN]
print("Top {} classes in order: {}".format(N,topN_inds))

# Now we can grab the code list and create a class Look Up Table
codes =  "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"
response = urllib2.urlopen(codes)
class_LUT = []
for line in response:
    code, result = line.partition(":")[::2]
    code = code.strip()
    result = result.replace("'", "")
    if code.isdigit():
        class_LUT.append(result.split(",")[0][1:])
        
# For each of the top-N results, associate the integer result with an actual class
for n in topN:
    print("Model predicts '{}' with {}% confidence".format(class_LUT[int(n[0])],float("{0:.2f}".format(n[1]*100))))


print("============== diff of preds:",abs(preds_cpu_norm-preds_gpu_norm))
if abs(preds_cpu_norm - preds_gpu_norm) > 1e-4:
    print("Mismatch between CPU and GPU")
    exit()
