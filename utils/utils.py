import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict
import ipdb


################################################################################
# Define device
################################################################################
def deviceSelect():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


################################################################################
# Load Model
# Reference: https://pytorchvideo.org/docs/tutorial_torchhub_inference
################################################################################
def loadModel(device):
    model_name = "i3d_r50"
    model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
    # 去掉类别的转换
    model.blocks[6].proj = torch.nn.Sequential()
    model = model.to(device)
    model = model.eval()

    return model


################################################################################
# Video transform
################################################################################
def videoTransform():
    ########################################
    # Define options
    ########################################
    side_size         = 224
    mean              = [0.45, 0.45, 0.45]
    std               = [0.225, 0.225, 0.225]
    crop_size         = 224
    num_frames        = 16
    sampling_rate     = 1
    frames_per_second = 30
    ########################################
    # Define transform
    ########################################
    transform =  ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x/255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(
                    size=side_size
                ),
                CenterCropVideo(crop_size),
            ]
        ),
    )
    ########################################
    # Caculate clip duration 
    ########################################
    clip_duration = (num_frames * sampling_rate)/frames_per_second 

    return transform, clip_duration

################################################################################
# Test
################################################################################
if __name__ == "__main__":

    device = deviceSelect()
    model = loadModel(device)
    
    video_path = "../archery.mp4"
    transform, clip_duration = videoTransform()
    # 指定视频开始与结束的时间
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    # video_data: {'video': [3, 16, 240, 320], 'audio': ...}
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec) # [3, 16, 240, 320]

    video_data = transform(video_data)  
    # print(video_data['video'].shape)  # torch.Size([3, 16, 224, 224])

    inputs = video_data["video"].unsqueeze(0).to(device)  # torch.Size([1, 3, 16, 224, 224])
    # print(inputs.shape)

    preds = model(inputs)
    # print(model)

    print(preds)
    print(preds.shape)


