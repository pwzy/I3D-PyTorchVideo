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


# Load model
# Device on which to run the model
# Set to cuda to load on GPU
device = "cuda:0"

# Pick a pretrained model and load the pretrained weights
model_name = "i3d_r50"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

# Set to eval mode and move to desired device
model = model.to(device)
model = model.eval()


####################
# SlowFast transform
####################

side_size = 224
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 224
num_frames = 16
sampling_rate = 2
frames_per_second = 30


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

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second 



# Run Inference
# Load the example video
video_path = "archery.mp4"
# Select the duration of the clip to load by specifying the start and end duration
# The start_sec should correspond to where the action occurs in the video
start_sec = 0
end_sec = start_sec + clip_duration

# Initialize an EncodedVideo helper class
video = EncodedVideo.from_path(video_path)

# Load the desired clip
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
# print(video_data.shape)  # video_data['video'] [3, 16, 240, 320]
# print(video_data['video'].shape)

# print(video_data['video'])
video_data = transform(video_data)

# print(video_data['video'].shape)  

# Move the inputs to the desired device
inputs = video_data["video"].unsqueeze(0).to(device)
# print(inputs.shape)


# print(preds.shape)

# print(model)


# print(net_structure)

# model = torch.nn.Sequential(*net_structure[:-2])

# model.blocks[6].dropout = torch.nn.Sequential()
model.blocks[6].proj = torch.nn.Sequential()
# model.blocks[6].output_pool = torch.nn.Sequential()

# print(model)


preds = model(inputs)
print(preds)
print(preds.shape)



