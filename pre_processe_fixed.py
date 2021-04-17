import numpy as np
import json
from PIL import Image
import copy
import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import os
import xml.etree.ElementTree as ET
from os import getcwd
import shutil
from PIL import Image
from PIL import ImageEnhance
from utils.utils import check_dir
import argparse

CLASS = ['pedestrian', 'bicycle', 'vehicle', 'scooter']
msg_format = '%d %f %f %f %f \n'
trans_dict = {'bus': 2,
               'person':0,
               'bike':1,
               'truck': 2,
               'motor': 3,
               'car':2}

def plot_one_box(img, x, color=None, label=None):
    img_PIL = Image.fromarray(img)
    font = ImageFont.truetype('./font/FiraMono-Medium.otf', 26)
    draw = ImageDraw.Draw(img_PIL)
    text_size = draw.textsize(label, font)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    draw.text((c1[0], c1[1]-26), label, color, font=font)
    draw.rectangle((c1, c2), outline=color, width=5)
    draw.rectangle((c1[0], c1[1], c1[0] + text_size[0], c1[1] - text_size[1] - 3), outline=color, width=5)

    return np.array(img_PIL)

def cal_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
 
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h
    iou = area / (s1 + s2 - area)
    return iou

def combinbox(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xmin = min(xmin1, xmin2)
    ymin = min(ymin1, ymin2)
    xmax = max(xmax1, xmax2)
    ymax = max(ymax1, ymax2)

    return [xmin,ymin,xmax,ymax]

def xy2cxcywh(xmin, ymin, xmax, ymax, x_scale=1, y_scale=1):
    center_x = ((xmax + xmin) / 2.0) * x_scale
    center_y = ((ymax + ymin) / 2.0) * y_scale
    width = (xmax - xmin) * x_scale
    height = (ymax - ymin) * y_scale
    return center_x, center_y, width, height


def cxcywh2xy(cx, cy, w, h, x_scale=1, y_scale=1):
    xmin = int((cx - w / 2) * x_scale)
    ymin = int((cy - h / 2) * y_scale)
    xmax = int((cx + w / 2) * x_scale)
    ymax = int((cy + h / 2) * y_scale)
    return xmin, ymin, xmax, ymax

def xml_convert_yolo(xml_path, x_scale=1. / 1920., y_scale=1. / 1080.):
    tree = ET.parse(open(xml_path))
    root = tree.getroot()
    bboxes = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in CLASS:
            continue

        cls_id = CLASS.index(cls)
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        ymin = int(xmlbox.find('ymin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymax = int(xmlbox.find('ymax').text)

        center_x, center_y, width, height = xy2cxcywh(xmin, ymin, xmax, ymax,
                                                      x_scale, y_scale)

        if width < 0 or height < 0:
            continue

        
        bboxes.append((cls_id, center_x, center_y, width, height))
        
    return bboxes


def ivs_to_yolo(path, save_path, sampling_margin=200):
    img_dirs = os.listdir(path)
    img_save_path = os.path.join(save_path, 'images')
    label_save_path = os.path.join(save_path, 'labels')
    check_dir(img_save_path)
    check_dir(label_save_path)
    
    for img_dir in img_dirs:
        img_root = os.path.join(path, img_dir)
        filenames = os.listdir(img_root)

        for filename in filenames[0::sampling_margin]:
            source = os.path.join(img_root, filename)
            dest = os.path.join(img_save_path, filename)
            shutil.copyfile(source, dest)

            source = source.replace('JPEGImages', 'Annotations').replace('jpg', 'xml')
            bboxes = xml_convert_yolo(source)
            filename = filename.replace('jpg', 'txt')
            dest = os.path.join(label_save_path, filename)
            with open(dest, "w") as weiter:
                for bbox in bboxes:
                    msg = msg_format % bbox
                    weiter.write(msg)
    
    
def get_bdd_fixed_filename(all_labels):
    class_name = ['rider', 'motor', 'bike']
    names = [[], [], []]

    for index in range(len(class_name)):
        for labels in all_labels:
            file_name = labels['name']
            for label in labels['labels']:
                if label['category'] == class_name[index]:
                    names[index].append(file_name)
                    break
                    
    rider_names = names[0]
    motor_names = names[1]
    bike_names = names[2]
    return rider_names, motor_names, bike_names

def json_convert_yolo(labels, x_scale=1. / 1280., y_scale=1. / 720.):
    bboxes = []
    
    for label in labels['labels']:
        category = label['category']
        if category not in trans_dict.keys():
            continue

        cls_id = trans_dict[category]
        xmin = int(label['box2d']['x1'])
        ymin = int(label['box2d']['y1'])
        xmax = int(label['box2d']['x2'])
        ymax = int(label['box2d']['y2'])

        center_x, center_y, width, height = xy2cxcywh(xmin, ymin, xmax, ymax,
                                                      x_scale, y_scale)

        if width < 0 or height < 0:
            continue
        
        bboxes.append((cls_id, center_x, center_y, width, height))
        
    return bboxes

def bdd_to_yolo(path='./bdd100k', save_path='./bdd100k/val', mode='val'):
    labels_path = os.path.join(path, 'labels/bdd100k_labels_images_' + mode + '.json') 
    with open(labels_path) as f:
        all_labels = json.load(f)
    
    img_root = os.path.join(path, 'images/100k/' + mode)
    img_save_path = os.path.join(save_path, 'images')
    label_save_path = os.path.join(save_path, 'labels')
    check_dir(img_save_path)
    check_dir(label_save_path)
    rider_names, motor_names, bike_names = get_bdd_fixed_filename(all_labels)
    for labels in all_labels:
        filename = labels['name']
        if filename in rider_names:
            continue
        
        shutil.copyfile(os.path.join(img_root, filename), os.path.join(img_save_path, filename))
        
        bboxes = json_convert_yolo(labels)
           
        dest = os.path.join(label_save_path, filename[:-4] + ".txt")
        with open(dest,"w") as weiter:
            for bboxe in bboxes:
                msg = msg_format % bboxe
                weiter.write(msg)
                
def rider_iou_pair(bbox_labels):
    if not (len(bbox_labels['rider']) <= (len(bbox_labels['bike']) + len(bbox_labels['motor']))):
        return None
            
    datas = []
    for rider in bbox_labels['rider']:
        max_value = 0.001
        max_index = -1
        max_type = ''

        for cls in ['bike', 'motor']:
            if len(bbox_labels[cls]) == 0:
                continue

            for index, data in enumerate(bbox_labels[cls]):
                iou = cal_iou(rider, data)
                if iou > max_value:
                    max_index = index
                    max_value = iou
                    max_type = cls
                    
        if max_type == '':
            return None
        else :
            rider[3] = rider[3] - 20
            xmin,ymin,xmax,ymax = combinbox(rider, bbox_labels[max_type][max_index])
            datas.append([trans_dict[max_type], xmin,ymin,xmax,ymax])
            bbox_labels[max_type].pop(max_index)
            
    for label in ['bike', 'motor']:
        cls = trans_dict[label]
        for bbox in bbox_labels[label]:
            datas.append([cls] + bbox)
    
    return datas
                
    
def fixed_bdd_to_yolo(path='./bdd100k', save_path='./bdd100k/val', mode='val',
                      x_scale=1. / 1280., y_scale=1. / 720.):
    
    img_root = os.path.join(path, 'images/100k/' + mode)
    img_save_path = os.path.join(save_path, 'images')
    label_save_path = os.path.join(save_path, 'labels')
    check_dir(img_save_path)
    check_dir(label_save_path)
    
    labels_path = os.path.join(path, 'labels/bdd100k_labels_images_' + mode + '.json') 
    with open(labels_path) as f:
        all_labels = json.load(f)
    
    names = []
    rider_labels = []
    
    ######
    for labels in all_labels:
        filename = labels['name']
        for label in labels['labels']:
            if label['category'] == "rider":
                names.append(filename)
                rider_labels.append(labels)
                break
                
                
    #####
    index = 0
    while index < len(rider_labels):
        filename = names[index]
        labels = rider_labels[index]
        
        bbox_labels = {"rider" : [], "bike" : [], "motor":[], "normal": []}
        for label in labels['labels']:
            category = label['category']
            if category in trans_dict.keys() or category == 'rider':
                xmin = int(label['box2d']['x1'])
                ymin = int(label['box2d']['y1'])
                xmax = int(label['box2d']['x2'])
                ymax = int(label['box2d']['y2'])

                if category in bbox_labels.keys():
                    if category == 'rider':
                        bbox_labels[category].append([xmin,ymin,xmax,ymax + 20])
                    else:
                        bbox_labels[category].append([xmin,ymin,xmax,ymax])
                else:
                    bbox_labels['normal'].append([trans_dict[category], xmin, ymin, xmax, ymax])
                    
        datas = rider_iou_pair(bbox_labels)

        if datas == None:
            rider_labels.pop(index)
            names.pop(index)
            continue
            
        dest = os.path.join(label_save_path, filename[:-4] + ".txt")
        with open(dest, "w") as weiter:
            for bbox in datas + bbox_labels['normal']:
                bbox[1], bbox[2], bbox[3], bbox[4] = xy2cxcywh(bbox[1], bbox[2], bbox[3], bbox[4],
                                                              x_scale, y_scale)
                bbox = tuple(bbox)
                msg = msg_format % bbox
                weiter.write(msg)
        shutil.copyfile(os.path.join(img_root, filename), os.path.join(img_save_path, filename))
            
        index += 1

                
def get_mask_filename(path, pedestrian_ratio=0.4, vehicle_ratio=0.5):
    img_path = os.path.join(path, 'images')
    label_path = os.path.join(path, 'labels')
    filenames = os.listdir(img_path)
    vehicle_mask = []
    pedestrian_mask = []
    
    for filename in filenames:
        local_counts = np.zeros(len(CLASS))
        with open(os.path.join(label_path, filename[:-4] + ".txt") ,"r") as f:
            
            for line in f.readlines():
                data = line.split(' ')
                cls = int(data[0])
                local_counts[cls] += 1

        local_radio = np.array(local_counts)
        local_radio = local_radio / np.sum(local_radio)
        
        if local_radio[2] >= vehicle_ratio:
            vehicle_mask.append(filename)

        if local_radio[0] >= pedestrian_ratio:
            pedestrian_mask.append(filename)
            
    return vehicle_mask, pedestrian_mask


                
def transform_mask(img_path, label_path,
                   mask_pedestrian=True, mask_vehicle=True, background_color=(114, 114, 114)):
    img = Image.open(img_path)
    img = np.array(img)
    height, width, _ = img.shape
    mask_img = copy.deepcopy(img)
    mask_labels = []
    
    with open(label_path,"r") as f:
        lines = [line for line in f.readlines()]
        
    for line in lines:
        data = line.replace('\n', '').split(' ')
        cls_id = int(data[0])
        
        if (mask_vehicle and cls_id == 2) or \
            (mask_pedestrian and cls_id == 0):
            xmin, ymin, xmax, ymax = cxcywh2xy(float(data[1]), float(data[2]),
                                               float(data[3]), float(data[4]),
                                               width, height)
            mask_img[ymin:ymax,xmin:xmax] = np.array(background_color)


    for line in lines:
        data = line.replace('\n', '').split(' ')
        cls_id = int(data[0])

        if (mask_vehicle and cls_id == 2) or \
            (mask_pedestrian and cls_id == 0):
            continue

        xmin, ymin, xmax, ymax = cxcywh2xy(float(data[1]), float(data[2]),
                                           float(data[3]), float(data[4]),
                                           width, height)
        mask_img[ymin:ymax,xmin:xmax] = img[ymin:ymax,xmin:xmax]
        #mask_img = plot_one_box(mask_img, np.array([xmin, ymin, xmax, ymax]), label='M', color=(0, 255, 0))
        mask_labels.append(line)
        
    return mask_img, mask_labels
        
def yolo_to_mask_images(path, save_path, pedestrian_ratio=0.4, vehicle_ratio=0.5):
    vehicle_mask, pedestrian_mask = get_mask_filename(path, pedestrian_ratio, vehicle_ratio)
    all_mask = np.array(vehicle_mask + pedestrian_mask)
    all_mask = np.unique(all_mask)
    
    img_save_path = os.path.join(save_path, 'images')
    label_save_path = os.path.join(save_path, 'labels')
    check_dir(img_save_path)
    check_dir(label_save_path)
    
    img_root = os.path.join(path, 'images')
    label_root = os.path.join(path, 'labels')

    for filename in all_mask:
        img_path = os.path.join(img_root, filename)
        label_path = os.path.join(label_root, filename[:-4] + ".txt")
        mask_vehicle = filename in vehicle_mask
        mask_pedestrian = filename in pedestrian_mask
        mask_img, mask_labels = transform_mask(img_path, label_path,
                                               mask_pedestrian=True, mask_vehicle=True, background_color=(114, 114, 114))
        
        
        if len(mask_labels) == 0:
            continue
        
        save_name = filename[:-4] + '_mask'
        with open(os.path.join(label_save_path, save_name + ".txt"),"w") as weiter:
            for mask_label in mask_labels:
                weiter.write(mask_label)

        Image.fromarray(mask_img).save(os.path.join(img_save_path, save_name + '.jpg'))
    

def test_bdd_to_yolo(path='./bdd100k', save_path='./bdd100k/val', mode='val'):
    bdd_to_yolo(path, save_path, mode)
    
def test_ivs_to_yolo(path, save_path, sampling_margin=200):
    ivs_to_yolo(path, save_path, sampling_margin=sampling_margin)
    
def test_yolo_to_mask_images(path, save_path, pedestrian_ratio=0.4, vehicle_ratio=0.5):
    mask_vehicle, mask_pedestrian = get_mask_filename(path)
    yolo_to_mask_images(path, save_path, pedestrian_ratio, vehicle_ratio)
    
if __name__ == '__main__':
    #fixed_bdd_to_yolo(path='./bdd100k', save_path='./bdd100k/val_mask', mode='val',
    #                  x_scale=1. / 1280., y_scale=1. / 720.)
    #test_yolo_to_mask_images('./bdd100k/val_mask', './bdd100k/train2', pedestrian_ratio=0.4, vehicle_ratio=0.5)
    #test_ivs_to_yolo('./ivslab/ivslab_train/JPEGImages/All', './bdd100k_ivslab/train/', 200)
    #test_ivs_to_yolo('./ivslab/ivslab_train/JPEGImages/All', './bdd100k_ivslab/val/', 300)
    #test_bdd_to_yolo(path='./bdd100k', save_path='./bdd100k/val', mode='val')
    #test_yolo_to_mask_images('./bdd100k_ivslab/train', './bdd100k_ivslab/train2', pedestrian_ratio=0.4, vehicle_ratio=0.5)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--convert', type=str, default='ivs2yolo', help='convert type')
    parser.add_argument('--path', default='./ivslab/ivslab_train/JPEGImages/All', help='image path')
    parser.add_argument('--save_path', default='./bdd100k_ivslab/train/', help='convert save path')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--mode', default='val', help='bdd100k mode(train or val)')
    parser.add_argument('--sampling_margin', type=int, default=200, help='ivs sampling')
    parser.add_argument('--pedestrian_ratio', type=float, default=0.4, help='pedestrian mask ratio')
    parser.add_argument('--vehicle_ratio', type=float, default=0.5, help='vehicle mask ratio')
    #opt = parser.parse_args()
    opt, unparsed = parser.parse_known_args()

    convert = opt.convert
    
    if convert == 'ivs2yolo':
        ivs_to_yolo(opt.path, opt.save_path, opt.sampling_margin)
    elif convert == 'bdd2yolo':
        bdd_to_yolo(opt.path, opt.save_path, opt.mode)
    elif convert == 'yolo2mask':
        yolo_to_mask_images(opt.path, opt.save_path,
                            opt.pedestrian_ratio, opt.vehicle_ratio)
    elif convert == 'fixedbdd2yolo':
        fixed_bdd_to_yolo(opt.path, opt.save_path, mode=opt.mode,
                          x_scale=1. / 1280., y_scale=1. / 720.)