Directory Structure:
-model:SaveModel
-run_detection.py:inference:input size=[1, 288, 480, 3] output size=[-1, 6]
-model.tflite(only detection model):input size=[1, 288, 480, 3] output list=[(1, 12, 20, 27), (1, 48, 80, 27), (1, 24, 40, 27)]
-requirements.txt:requirements package
-submission.csv:Qualification inference
-Frozn_model.pd



Detection:
python run_detection.py {image_list path} {submission.csv path}

Detection Example:
python run_detection.py ./ ./ #(image_list.txt、submission.csv)
python run_detection.py ./image_list.txt ./submission.csv
python run_detection.py ./image_list.txt ./ans.csv