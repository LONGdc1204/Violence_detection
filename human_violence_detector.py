import torch
import random
import os
import glob
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from get_video import Frame_Extraction_V2
from get_model import get_model
from training_process import training_model


##### config parameters
data_dir = 'video_data'
resnet_pretrained_weight_path ='pretrained_weight\\r3d18_KM_200ep.pth'
val_train_ratio = 0.2
batch_size = 3
global crop_width
crop_width = 512
global crop_height
crop_height = 512

device = 0
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(fraction = 0.92, device = device)
else:
    device = torch.device('cpu')

# các classes có trong đường dẫn chứa data
available_classes = os.listdir(data_dir)

# dictionary chứa index của class dạng tensor
class_dict = {'threat some with a knife': torch.tensor(0),
              'hit someone with something': torch.tensor(1),
              'threat someone with a gun': torch.tensor(2),
              'stab someone with a knife': torch.tensor(3),
              'kicking someone': torch.tensor(4),
              'hold someone hostage': torch.tensor(5),
              'punching someone': torch.tensor(6),
              'chase someone': torch.tensor(7),
              'pushing someone':torch.tensor(8) }


# Tạo list chứa đường dẫn đến video cho train và validation
train_video_dir_list = []
vali_video_dir_list = []
for single_class in available_classes:
    curr_class_dirs = glob.glob(data_dir +'\\'+single_class+'\\*')
    ini_length = len(curr_class_dirs)
    new_curr_class_dirs= curr_class_dirs[0: 1*ini_length//3]
    random.shuffle(new_curr_class_dirs)
    length_val = int(len(new_curr_class_dirs)*val_train_ratio)
    in_class_val = new_curr_class_dirs[0:length_val]
    in_class_train = new_curr_class_dirs[length_val:len(new_curr_class_dirs)]
    train_video_dir_list.extend(in_class_train)
    vali_video_dir_list.extend(in_class_val)


random.shuffle(train_video_dir_list)
random.shuffle(vali_video_dir_list)

# khởi tạo transformation
video_transform = T.Compose(
    [T.Resize(size = (crop_width,crop_height), interpolation = InterpolationMode.NEAREST, antialias = True),
     T.ConvertImageDtype(torch.float32) ])


# tạo instance lấy dữ liệu
const_num_frames = 32
train_ds = Frame_Extraction_V2(available_classes,
                                class_dict,
                                train_video_dir_list,
                                const_num_frames,
                                video_transform,
                                device)
val_ds = Frame_Extraction_V2(available_classes,
                            class_dict,
                            vali_video_dir_list,
                            const_num_frames,
                            video_transform,
                            device)

# Create training and validation data loaders
train_dl = DataLoader(train_ds, batch_size= batch_size, shuffle=True, drop_last = True)
val_dl = DataLoader(val_ds, batch_size= batch_size, shuffle=True, drop_last = True)


# tạo mô hình
CNN3D_model = get_model(resnet_pretrained_weight_path, 6, device)


# huấn luyện mô hình 
train_acc_total, train_loss_total, val_acc_total, val_loss_total, val_cf_matrix = training_model(num_epochs = 40,
                                                                                                model_instance = CNN3D_model, 
                                                                                                train_dl = train_dl, 
                                                                                                val_dl = val_dl, 
                                                                                                device = device, 
                                                                                                learning_rate = 0.001)


# monitor output
cf_matrix = np.zeros(shape = (9,9),dtype = np.int8)
select_epoch_index = 12
for (cf_pred,cf_label) in val_cf_matrix[select_epoch_index]:
    for curr_pred, curr_label in zip(cf_pred.to('cpu').numpy(),cf_label.to('cpu').numpy()):
        cf_matrix[curr_label, curr_pred] +=1

# plot confusion matrix
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(cf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(cf_matrix.shape[0]): # for label
    for j in range(cf_matrix.shape[1]):  # for predict
        ax.text(x=j, y=i,s=cf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

plt.plot(train_acc_total,color = 'green')
plt.plot(val_acc_total,color = 'blue')
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.title("Traing/validation accuracy")
plt.show()

plt.plot(train_loss_total,color = 'green')
plt.plot(val_loss_total,color = 'blue')
plt.title("Traing/validation loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()