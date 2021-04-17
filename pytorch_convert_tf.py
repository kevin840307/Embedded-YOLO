import torch
import pickle
from models.experimental import *
from utils.datasets import *
import csv
import os 
import tensorflow as tf
import numpy as np
import cv2 as cv
import argparse

model_path = './embedded_yolo.pt'
param_path = './embedded_yolo.dict'
save_dtype = tf.float32
input_shape=[288, 480]
strides = [8, 16, 32]
class_num=4
feat_size = [[input_shape[0] // s, input_shape[1] // s] for s in strides]

def generate_dict(model_path, param_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model['model'].eval().fuse()
    data_dict={}

    #for param in model['model'].parameters():
    #    print(param.shape)

    for k, v in model['model'].state_dict().items():
        vr = v.cpu().numpy()
        data_dict[k]=vr
        #print(k, ' ', vr.shape)

    fid=open(param_path,'wb')   
    pickle.dump(data_dict,fid)
    fid.close()
    
def bn(input, name='bn'):
    with tf.variable_scope(name):
        gamma=tf.Variable(tf.random_normal(shape=[input.shape[-1].value]), name='weight',trainable=False)
        beta = tf.Variable(tf.random_normal(shape=[input.shape[-1].value]), name='bias',trainable=False)
        mean = tf.Variable(tf.random_normal(shape=[input.shape[-1].value]), name='running_mean',trainable=False)
        var = tf.Variable(tf.random_normal(shape=[input.shape[-1].value]), name='running_var',trainable=False)

        out=tf.nn.batch_normalization(input,mean,var,beta,gamma,variance_epsilon=0.001)
        
        return out
    
def conv(input,out_channels,ksize,stride,name='conv',add_bias=False):
    filter = tf.Variable(tf.random_normal(shape=[ksize, ksize, input.shape[-1].value, out_channels], dtype=save_dtype), dtype=save_dtype,name=name+'/weight',trainable=False)
    if ksize>1:
        pad_h,pad_w=ksize//2,ksize//2
        paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
        input = tf.pad(input, paddings, 'CONSTANT')
    net = tf.nn.conv2d(input, filter, [1,stride, stride, 1], padding="VALID")
    
    if add_bias:
        bias = tf.Variable(tf.random_normal(shape=[out_channels], dtype=save_dtype),
                             name=name + '/bias',trainable=False, dtype=save_dtype)
        net=tf.nn.bias_add(net,bias)
    return net


def convBnLeakly(input,out_channels,ksize,stride,name):
    with tf.variable_scope(name):
        net=conv(input,out_channels,ksize,stride, add_bias=True)
        #net=bn(net)
        net=tf.nn.leaky_relu(net,alpha=0.1)
        return net
    
    
def DepthWiseConv(input,ksize,stride,name='conv',add_bias=False):
    out_channels = input.shape[-1].value
    filter = tf.Variable(tf.random_normal(shape=[ksize, ksize, out_channels, 1], dtype=save_dtype), dtype=save_dtype,name=name+'/weight',trainable=False)

    if ksize>1:
        pad_h,pad_w=ksize//2,ksize//2
        paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
        input = tf.pad(input, paddings, 'CONSTANT')
    net = tf.nn.depthwise_conv2d(input, filter, [1,stride, stride, 1], padding="VALID")
    
    if add_bias:
        bias = tf.Variable(tf.random_normal(shape=[out_channels], dtype=save_dtype),
                             name=name + '/bias',trainable=False, dtype=save_dtype)
        net=tf.nn.bias_add(net,bias)
    return net


    
def DepthWiseConvBnLeakly(input, out_channels,ksize,stride,name, e=1.5):
    hidden_dim = int(input.shape[-1].value * e)
    with tf.variable_scope(name):
        net=conv(input,hidden_dim,1,1,name='0')
        net=bn(net,name='1')
        net=tf.nn.leaky_relu(net,alpha=0.1)
        
        net=DepthWiseConv(net,ksize, stride,name='3')
        net=bn(net,name='4')
        net=tf.nn.leaky_relu(net,alpha=0.1)
        
        net=conv(net,out_channels,1,1,name='6')
        net=bn(net,name='7')
        net=tf.nn.leaky_relu(net,alpha=0.1)
        
        return net
    
def InvertedResidual(input,c1,c2,shortcut,e,name):
    identity = shortcut and c1 == c2
    with tf.variable_scope(name + '/conv'):
        conv = DepthWiseConvBnLeakly(input, c2, 3, 1, 'conv')
        
        if identity:
            return input + conv
        else:
            return conv
    
def DepthBottleneckCSP(input,c1,c2,n,shortcut,e,name):
    c_=int(c2*e)
    with tf.variable_scope(name):
        net1=convBnLeakly(input,c_,1,1,'cv1')
        for i in range(n):
            net1=InvertedResidual(net1,c_,c_,shortcut,1.0,name='m/%d'%i)
        net1=conv(net1,c_,1,1,name='cv3')

        net2 = conv(input, c_, 1, 1, 'cv2')

        net=tf.concat((net1,net2),-1)
        net=bn(net)
        net=tf.nn.leaky_relu(net,alpha=0.1)

        net=convBnLeakly(net,c2,1,1,'cv4')
        return net
    

def focus(input,out_channels,ksize,name):
    s1=input[:,::2,::2,:]
    s2=input[:,1::2,::2,:]
    s3 = input[:, ::2, 1::2, :]
    s4 = input[:, 1::2, 1::2, :]
        
    net=tf.concat([s1,s2,s3,s4],axis=-1)            
    net=convBnLeakly(net,out_channels,ksize,1,name+'/conv')
    return net

def bottleneck(input,c1,c2,shortcut,e,name):
    with tf.variable_scope(name):
        net=convBnLeakly(input,int(c2*e),1,1,'cv1')
        net=convBnLeakly(net,c2,3,1,'cv2')

        if (shortcut and c1==c2):
            net+=input
        return net

def bottleneckCSP(input,c1,c2,n,shortcut,e,name):
    c_=int(c2*e)
    with tf.variable_scope(name):
        net1=convBnLeakly(input,c_,1,1,'cv1')
        for i in range(n):
            net1=bottleneck(net1,c_,c_,shortcut,1.0,name='m/%d'%i)
        net1=conv(net1,c_,1,1,name='cv3')

        net2 = conv(input, c_, 1, 1, 'cv2')

        net=tf.concat((net1,net2),-1)
        net=bn(net)
        net=tf.nn.leaky_relu(net,alpha=0.1)

        net=convBnLeakly(net,c2,1,1,'cv4')
        return net
    
def spp(input,c1,c2,k1,k2,k3,name):
    c_=c1//2
    with tf.variable_scope(name):
        net=convBnLeakly(input,c_,1,1,'cv1')

        net1=tf.nn.max_pool(net,ksize=[1,k1,k1,1],strides=[1,1,1,1],padding="SAME")
        net2=tf.nn.max_pool(net,ksize=[1,k2,k2,1],strides=[1,1,1,1],padding="SAME")
        net3 = tf.nn.max_pool(net, ksize=[1, k3, k3, 1], strides=[1, 1, 1, 1], padding="SAME")

        net=tf.concat((net,net1,net2,net3),-1)

        net=convBnLeakly(net,c2,1,1,'cv2')
        return net
    
def yolov5(input,class_num):
    depth_multiple = 0.33
    width_multiple = 0.5
    
    
    w1 = int(round(64 * width_multiple))
    w2 = int(round(128 * width_multiple))
    w3 = int(round(256 * width_multiple))
    w4 = int(round(512 * width_multiple))
    w5 = int(round(1024 * width_multiple))

    d1 = int(max(round(3 * depth_multiple), 1))
    d2 = int(max(round(9 * depth_multiple), 1))

    focus0=focus(input,w1,3,'model/0')
    
    conv1=convBnLeakly(focus0,w2,3,2,'model/1')
    bottleneck_csp2=bottleneckCSP(conv1,w2,w2,d1,True,0.5,'model/2')
    conv3 = convBnLeakly(bottleneck_csp2, w3, 3, 2, 'model/3')
    bottleneck_csp4 = bottleneckCSP(conv3, w3, w3, d2, True, 0.5, 'model/4')
    conv5 = convBnLeakly(bottleneck_csp4, w4, 3, 2, 'model/5')
    bottleneck_csp6 = bottleneckCSP(conv5, w4, w4, d2, True, 0.5, 'model/6')
    conv7 = DepthWiseConvBnLeakly(bottleneck_csp6, w4, 3, 2, 'model/7/conv')
    spp8=spp(conv7,w4,w4,5,9,13,'model/8')

    bottleneck_csp9 = DepthBottleneckCSP(spp8, w4, w4, d1, False, 0.5, 'model/9')
    conv10 = convBnLeakly(bottleneck_csp9, w4, 1, 1, 'model/10')

    shape=[conv10.shape[1].value*2,conv10.shape[2].value*2]
    
    deconv11=tf.image.resize_images(conv10,shape,method=1)

    cat12=tf.concat((deconv11,bottleneck_csp6),-1)
    bottleneck_csp13=bottleneckCSP(cat12, w5, w4, d1, False, 0.5, 'model/13')
    conv14 = convBnLeakly(bottleneck_csp13, w3, 1, 1, 'model/14')

    shape = [conv14.shape[1].value * 2, conv14.shape[2].value * 2]
    deconv15 = tf.image.resize_images(conv14, shape,method=1)
    
    cat16 = tf.concat((deconv15, bottleneck_csp4), -1)
    bottleneck_csp17 = bottleneckCSP(cat16, w4, w3, d1, False, 0.5, 'model/17')
    conv18 = convBnLeakly(bottleneck_csp17, w3, 3, 2, 'model/18')

    cat19 = tf.concat((conv18, conv14), -1)
    bottleneck_csp20 = bottleneckCSP(cat19, w4, w4, d1, False, 0.5, 'model/20')
    conv21 = convBnLeakly(bottleneck_csp20, w4, 3, 2, 'model/21')

    cat22= tf.concat((conv21, conv10), -1)
    bottleneck_csp23 = bottleneckCSP(cat22, w5, w5, d1, False, 0.5, 'model/23')

    conv24m0=conv(bottleneck_csp17,3*(class_num+5),1,1,'model/24/m/0',add_bias=True)
    conv24m1 = conv(bottleneck_csp20, 3 * (class_num + 5), 1, 1, 'model/24/m/1',add_bias=True)
    conv24m2 = conv(bottleneck_csp23, 3 * (class_num + 5), 1, 1, 'model/24/m/2',add_bias=True)
    return conv24m0,conv24m1,conv24m2

    
def post_process(inputs,grids,strides,anchor_grid,class_num, iou_th=0.5, conf_th=0.03, is_multiple=False):

    total=[]
    for i,logits in enumerate(inputs):
        logits = tf.cast(logits, tf.float32)
        nb=logits.shape[0]#.value
        ny = logits.shape[1]#.value
        nx = logits.shape[2]#.value
        nc = logits.shape[3]#.value

        logits=tf.reshape(logits,[nb,ny,nx,3,nc//3])
        logits=tf.sigmoid(logits)

        logits_xy=(logits[...,:2]*2.-0.5+grids[i])*strides[i]
        logits_wh = ((logits[...,2:4] * 2)**2)*anchor_grid[i]

        logits_new=tf.concat((logits_xy,logits_wh,logits[...,4:]),axis=-1)

        total.append(tf.reshape(logits_new,[-1,nc//3]))
    total=tf.concat(total,axis=0)
    
    
    
    mask = total[:, 4] > conf_th
    total = tf.boolean_mask(total, mask)

    
    x,y,w,h,conf,prob=tf.split(total,[1,1,1,1,1,class_num],axis=-1)
    x1=x-w/2.
    y1=y-h/2.
    x2=x+w/2.
    y2=y+h/2.
    conf_prob=conf*prob
    
    if is_multiple:
        scores=tf.reduce_max(conf_prob,axis=-1)
        scores = tf.cast(scores,tf.float32)
        labels=tf.cast(tf.argmax(conf_prob,axis=-1),tf.float32)

        boxes=tf.concat([x1,y1,x2,y2],axis=1)
        boxes=tf.cast(boxes,tf.float32)
        
        indices=tf.image.non_max_suppression(boxes,scores,max_output_size=1000,iou_threshold=iou_th,score_threshold=conf_th)
    else:
        scores = tf.cast(conf_prob, tf.float32)
        scores = tf.reshape(scores, [-1])
        
        labels = tf.constant([0, 1, 2, 3], dtype=tf.float32)
        labels = tf.tile(labels, [tf.shape(scores)[0] // 4])

        boxes =tf.concat([x1,y1,x2,y2],axis=1)
        boxes = tf.cast(boxes, tf.float32)
        boxes = tf.tile(boxes, [1, 4])
        boxes = tf.reshape(boxes, [-1, 4])
        
        scores_mask = scores > conf_th
        labels = tf.boolean_mask(labels, scores_mask)
        boxes = tf.boolean_mask(boxes, scores_mask)
        scores = tf.boolean_mask(scores, scores_mask)
        indices=tf.image.non_max_suppression(boxes + tf.reshape(labels, [-1, 1]) * 4096,scores,max_output_size=1000,iou_threshold=iou_th,score_threshold=conf_th)
    
    boxes=tf.gather(boxes,indices)
    scores=tf.reshape(tf.gather(scores,indices),[-1,1])
    labels=tf.reshape(tf.gather(labels,indices),[-1,1])

    output=tf.concat([boxes,scores,labels],axis=-1)
    return output

def read_dict(param_path, feat_size):
    weights=open(param_path,'rb')
    params_dict = pickle.load(weights)
    grids = []
    for size in feat_size:
        ny, nx = size
        import torch
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid= torch.stack((xv, yv), 2).view((1, ny, nx,1, 2)).float().numpy()

        grid = tf.convert_to_tensor(grid, tf.float32)
        grids.append(grid)
    anchors = params_dict['model.24.anchors']
    anchor_gird = params_dict['model.24.anchor_grid']
    anchor_gird = np.transpose(anchor_gird, (0, 1, 3, 4, 2, 5))
    anchor_gird = anchor_gird.astype(np.float32)
    return params_dict, anchor_gird, grids
    
def get_tf_assign_op(params_dict):
    vars = tf.global_variables()
    list_layer = []
    total_layer = 0
    for params in params_dict.keys():
        if params.find("num_batches_tracked") == -1:
            total_layer += 1
            list_layer.append(params)


    sucessful = 0
    assign_ops = []
    dephe_convs = ['model.7.conv.3.weight',
                   'model.9.m.0.conv.conv.3.weight']
    for var in vars:
        name = var.name[:-2].replace("/",'.')
        try:
            params=params_dict[name]
            if len(params.shape) == 4:
                if name in dephe_convs:
                    params=np.transpose(params,(2,3,0,1))
                else:   
                    params=np.transpose(params,(2,3,1,0))

            # print(p.shape, var)
            assign_ops.append(tf.assign(var, params))
            sucessful += 1
            list_layer.remove(name)
        except:
            print("load wedits error:", name, ' ', var.shape)
    print("Sucessful:", sucessful, '/', total_layer)
    print(len(vars))
    print(list_layer)
    return assign_ops

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./embedded_yolo.pt')
    parser.add_argument('--param_path', type=str, default='./embedded_yolo.dict')
    parser.add_argument('--save_path', type=str, default='./TF')
    parser.add_argument('--type', type=int, default=1)
    
    #opt = parser.parse_args()
    opt, unparsed = parser.parse_known_args()
    if opt.type == 1:
        save_dtype = tf.float16
        
    model_path = opt.model_path
    param_path = opt.param_path

    
    generate_dict(model_path, param_path)
    params_dict, anchor_gird, grids = read_dict(param_path, feat_size)

    input=tf.placeholder(save_dtype, shape=[1, input_shape[0],input_shape[1],3],name='input')
    logits=yolov5(input,class_num)
    logit1=tf.identity(logits[0],'out_logit1')
    logit2=tf.identity(logits[1],'out_logit2')
    logit3=tf.identity(logits[2],'out_logit3')
    output=post_process(logits,grids,strides,anchor_gird,class_num)
    output=tf.identity(output,'output')

    assign_ops = get_tf_assign_op(params_dict)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(assign_ops)
        print("Done! Good job!")
        
        import shutil
        if os.path.isdir(os.path.join(opt.save_path, 'model')):
            shutil.rmtree(os.path.join(opt.save_path, 'model'))
            
        if os.path.isdir(os.path.join(opt.save_path, 'lite_model')):
            shutil.rmtree(os.path.join(opt.save_path, 'lite_model'))

            
        tf.saved_model.simple_save(sess, os.path.join(opt.save_path, 'model'), 
                                       inputs={"inputs": input},
                                       outputs={"output": output})
            
        tf.saved_model.simple_save(sess, os.path.join(opt.save_path, 'lite_model'), 
                                       inputs={"inputs": input},
                                       outputs = {"out_logit1": logit1, "out_logit2": logit2, "out_logit3": logit3})
        
        converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                  input_graph_def=sess.graph.as_graph_def(),
                                  output_node_names=["output"])
        
        with tf.gfile.GFile(os.path.join(opt.save_path, 'Frozen_model.pb'), "wb") as f:
            f.write(converted_graph_def.SerializeToString())

    if save_dtype == tf.float32:
        converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(os.path.join(opt.save_path, 'lite_model'))
        tflite_model = converter.convert()
        with open(os.path.join(opt.save_path, 'model.tflite'), 'wb') as f:
            f.write(tflite_model)



