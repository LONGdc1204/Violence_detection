import torch
from torchvision.io import read_video
from torch.utils.data import Dataset
import torchvision.transforms as T

class Frame_Extraction_V2(Dataset):
    def __init__(self,
                 available_classes: list,
                 class_dict: dict,
                 data_dir_list: list,
                 num_frames_output: int,
                 data_transform: T.Compose,
                 device: torch.device):

        self.available_classes = available_classes
        self.class_dict = class_dict
        self.data_dir_list = data_dir_list
        self.num_frames_output = num_frames_output
        self.data_transform = data_transform
        self.device = device

    def _inference_ssd(self, video_dir: str):
        # get video tensor
        video_frames = read_video(video_dir, start_pts = 0,
                                  pts_unit ='sec',output_format="TCHW")[0]  # return tensor

        video_frames = self.data_transform(video_frames)


        video_frames = video_frames.to(self.device)   # cast to GPU

        # permute before feeding into 3D CNN model (B,C,H,W) -> (C,B,H,W)
        video_frames = video_frames.permute(1,0,2,3)

        return video_frames

    def __len__(self):
        return len(self.data_dir_list)


    def __getitem__(self, idx):
        # data
        current_video_dir = self.data_dir_list[idx]
        x_data = self._inference_ssd(current_video_dir)
        # label
        # get current class from video directory
        elements = current_video_dir.split('\\')
        class_str_name = [element for element in elements if element in self.available_classes][0]
        y_data = self.class_dict[class_str_name]

        return x_data, y_data