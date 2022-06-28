# I3D-PyTorchVideo

Reference： https://github.com/tianyu0207/RTFM
https://pytorchvideo.org/docs/tutorial_torchhub_inference

# Download an vieo for utils/utils.py test. 
wget -c https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4

# Download UCF-Crime dataset.
Website: https://www.crcv.ucf.edu/projects/real-world/

The downloaded dataset is shown below:
```
├── Anomaly_Train.txt
├── Anomaly-Videos-Part-1.zip
├── Anomaly-Videos-Part-2.zip
├── Anomaly-Videos-Part-3.zip
├── Anomaly-Videos-Part-4.zip
├── config.yaml
├── Normal_Videos_for_Event_Recognition.zip
├── ReadMe-Anomaly-Detection.txt
├── Temporal_Anomaly_Annotation_for_Testing_Videos.txt
├── Testing_Normal_Videos.zip
├── Training-Normal-Videos-Part-1.zip
├── Training-Normal-Videos-Part-2.zip
└── UCF_Crimes-Train-Test-Split.zip
```
# After deleting  Normal_Videos_for_Event_Recognition.zip, unzip all the files and organize them, the directory tree is as follows:
```
.
├── Anomaly_Detection_splits
├── Anomaly_Train.txt
├── Anomaly-Videos-Part-1
├── Anomaly-Videos-Part-2
├── Anomaly-Videos-Part-3
├── Anomaly-Videos-Part-4
├── Temporal_Anomaly_Annotation_for_Testing_Videos.txt
├── Testing_Normal_Videos_Anomaly
├── Training-Normal-Videos-Part-1
└── Training-Normal-Videos-Part-2
```
# Instructions：
    Anomaly-Videos-Part-{1..4} is the abnormal dataset, which contains 950 anomaly videos.
        Anomaly_Train.txt : Division of the training set of anomalous videos.
        Temporal_Anomaly_Annotation_for_Testing_Videos.txt : Division of abnormal video testing set.

    Testing_Normal_Videos_Anomaly, Training-Normal-Videos-Part-1, and Training-Normal-Videos-Part-2 are the abnormal dataset, which contains 950 normal videos. The training and test datasets have been divided.

# Feature extract
```
python main.py
```


        
    





