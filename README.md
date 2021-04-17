# Embedded YOLO : Faster and Lighter Object Detection
## Our Model
![architecture](https://user-images.githubusercontent.com/64062370/115116778-13406a00-9fce-11eb-8bb8-f650a427372e.png)

## Dynamic Interpolation Mosaic
![dynamic](https://user-images.githubusercontent.com/64062370/115116839-5ac6f600-9fce-11eb-86c4-98027602d71a.jpg)

## Ablation 
|  Model   | Filters Size | Dynamic Interpolation mosaic  | Two Stage | Student Features | Parameters | GLOPs(512x460) | mAP(IOU@50) |
|  ----  | ----  | ----  | ----  | ----  | ----  | ----  | ----  |
| (a)[YOLOv5](https://drive.google.com/drive/folders/1sC5jk4AqmgsAdOXjNKea6_IDuzIkIEOZ?usp=sharing)  | 512 | Y | Y | N  | 7.26M  | 6.8  | 0.460  |
| (b)[YOLOv5+DSC_CSP](https://drive.google.com/drive/folders/1C_65StiNxWNBVwZkMaWKZX95uA34nazt?usp=sharing)  | 512 | Y | Y | N  | 5.92M  | 6.4  | 0.456  |
| (c)[YOLOv5+DSC_CSP](https://drive.google.com/drive/folders/1RVEI88EbIG3nXL6IT-nMTg0OwZnbH1-n?usp=sharing)  | 256 | Y | Y | N  | 4.67M  | 6.0  | 0.453  |
| (d)[YOLOv5+DSC_CSP](https://drive.google.com/drive/folders/1NBgvyI1HDLgk7ZFw2_cdzzIYxbw1M97Z?usp=sharing)  | 128 | Y | Y | N  | 4.32M  | 5.9  | 0.446  |
| (e)[YOLOv5+DSC_CSP](https://drive.google.com/drive/folders/1_8ICZhjAhO9JTZN4yWNUjhU2Dml7E79y?usp=sharing)  | 256 | N | Y | N  | 4.67M  | 6.0  | 0.451  |
| (f)[YOLOv5+DSC_CSP](https://drive.google.com/drive/folders/1rKnTJmO4PGoLj5FTtfumSIEH59FLWdh7?usp=sharing)  | 256 | N | N | N  | 4.67M  | 6.0  | 0.437  |
| (g)[YOLOv5+DSC_CSP](https://drive.google.com/drive/folders/1U0BG7UFOwK4jETDhR6iahXrYrPlxoxHZ?usp=sharing)  | 256 | Y | Y | [6,10]  | 4.67M  | 6.0  | 0.451  |
| (h)[YOLOv5+DSC_CSP](https://drive.google.com/drive/folders/1TasLP3l6fjffdwtXLceVK5JJWNFYwNwA?usp=sharing)  | 256 | Y | Y | [4,6,10]  | 4.67M  | 6.0  | 0.456  |
| (i)[YOLOv5+DSC_CSP](https://drive.google.com/drive/folders/1l5-693VZVlPpUEN_s5YezcKIfqS3G4QY?usp=sharing)  | 256 | Y | Y | [4,6,10,14]  | 4.67M  | 6.0  | 0.451  |

## ICMR - Final Competition
* Model: using (h) model
* Speed: on the MediaTek's Dimensity 1000 platform

|  Model   | Speed (ms/frame) | Parameters | GLOPs(480x288) | mAP(IOU@50) | accuracy scooter | accuracy bicycle |
|  ----  | ----  | ----  | ----  | ----  | ----  | ----  |
| [YOLOv5+DSC_CSP](https://drive.google.com/file/d/12Y5hohyKyxf6lNBvEYWTJB7JnK4Hpf8_/view?usp=sharing) | 9.1 | 4.67M | 5.04  | 0.59 | 0.535 | 0.535 |

## ALL Weights(incude other weights)
[ALL Weights](https://drive.google.com/drive/folders/1qc982u2V7_uSptziKbcjbyxwLTJsGEbh?usp=sharing)

## Step
### 1. pre-process
*** Convert ivslab Dataset to YOLO ***
 > - python pre_processe.py --convert ivs2yolo --path ./ivslab/ivslab_train/JPEGImages/All --save_path ./bdd100k_ivslab/train/ 
 > 
***Convert bdd100k Dataset to YOLO***
 > - python pre_processe_fixed.py --convert bdd2yolo --path ./bdd100k --save_path ./bdd100k_ivslab/train/ --mode train
 > 
***Label conversion to ivas data and convert to YOLO***
 > - python pre_processe_fixed.py --convert fixedbdd2yolo --path ./bdd100k --save_path ./bdd100k_ivslab/train/ --mode train
 > ![bddconvertivs](https://user-images.githubusercontent.com/64062370/115118123-69b0a700-9fd4-11eb-9fad-2f272b9dcc3a.jpg)
 > 
***Generate mask dataset***
 > - python pre_processe.py --convert yolo2mask --path ./bdd100k_ivslab/train/ --save_path ./mask_bdd100k_ivslab/val/

### 2. train
 > - python train.py

### 3. inference(Competition format)
 > - python test.py --data ivslab.yaml --img 448 --conf 0.03 --iou 0.5 --weights embedded_yolo.pt --batch 4 --task test

## Convert Model to Tensorflow1.13.2
### 1. convert
 > - python pytorch_convert_tf.py --model_path ./embedded_yolo.pt --param_path ./embedded_yolo.dict --save_path ./TF --type 1 # type: 1 is float16, other is float32
 > 
### 2. inference(Competition format)
 > - cd ./TF
 > - python run_detection.py ./image_list.txt ./submission.csv
 
 
## Only inference using Tensorflow1.13.2
### requirements
> - numpy
> - opencv-python
> - tensorflow==1.13.2
> 
### inference(Competition format)
 > - cd ./TF
 > - python run_detection.py ./image_list.txt ./submission.csv

# Reference
1. [YOLOv5](https://github.com/ultralytics/yolov5)
2. [Pytorch to Tensorflow](https://zhuanlan.zhihu.com/p/345184106)
