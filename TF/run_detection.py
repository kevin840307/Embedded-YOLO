import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
import os
import csv
import sys

reals = [2, 4, 1, 3]
img_size = 480 #416
input_shape=[288, 480]

#384, 640
#320, 512
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def xyxy2xywh(x):
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2
    return boxes
    
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords = clip_coords(coords, img0_shape)
    return coords

def load_image(path):
    global img_size
    img = cv2.imread(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    h0, w0 = img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not False else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

def main(image_list, save_filepath):
    with open(image_list,'r') as fp:
        lines = fp.readlines()
        paths = [ line.replace('\n', '')for line in lines]

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open('./Frozen_model.pb', "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        sess = tf.Session()
        #sess.run(tf.global_variables_initializer())

        input = sess.graph.get_tensor_by_name("input:0")
        output = sess.graph.get_tensor_by_name("output:0")

        with open(save_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_filename', 'label_id', 'x', 'y', 'w', 'h', 'confidence'])
            for path in paths:
                filename = os.path.basename(path)
                img, (h0, w0), (h, w) = load_image(path)
                img, ratio, pad = letterbox(img, input_shape, auto=False, scaleup=False)
                shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

                img = img[:, :, ::-1]
                img = np.ascontiguousarray(img)
                img_ = np.expand_dims(img, 0)
                if img_.max() > 1:
                    img_ = img_ / 255
                pred = sess.run(output, feed_dict={input: img_})


                box = pred[:, :4]  # xyxy
                box = scale_coords(input_shape, box, shapes[0], shapes[1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2
                pred[:, :4] = box
                for i in range(pred.shape[0]):
                    box = (pred[i, :4] + 0.5).astype(np.int32)
                    score = pred[i, 4]
                    label = int(pred[i, 5])
                    writer.writerow([filename, reals[label]] + list(box) + [round(score, 5)])

if __name__ == '__main__':
    assert len(sys.argv) == 3, "args error"
    image_list = sys.argv[1]
    save_filepath = sys.argv[2]
    if image_list.find('.txt') < 0:
        image_list = os.path.join(image_list, 'image_list.txt')
    if save_filepath.find('.csv') < 0:
        save_filepath = os.path.join(save_filepath, 'submission.csv')
    main(image_list, save_filepath)
    #main()