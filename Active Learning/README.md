# Active Learning for Remote Sensing

This project implements active learning for remote sensing data with temporal and spectral resolutions classification using a CNN-based model. It supports multiple query strategies (e.g., margin sampling, least confident, entropy, diversity) to help train models efficiently using fewer labeled samples.

---

## Features

- active-learning.ipynb notebook and active_learning_script.py python script
- Multiple active learning strategies:
  - `margin`
  - `least_confident`
  - `random`
  - `entropy`
  - `diversity`
  - `density`
  - `entropy_diversity`
- Multiple Data shape inputs
- Stores and Visualizes metrics suhc as Loss, Accuracy, F1, Confusion Matrix
- Checkpoints for saving and loading model state

---

## Requirements
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
tqdm

## File structure
```bash
/your/data/path/
├── tar_image_subset.npy     # shape: (N, Channels, Timesteps) can be changed according to model
├── label_target_subset.npy  # shape: (N,)
├── backboneSiteA2019.pth    # Pretrained CNN backbone
├── fcSiteA2019.pth          # Pretrained FC classifier head
```

## Configuration

Change these lines to configure them accoring to needs:

```python
#shape (Num, Channels, Timesteps)
Input_path = '/kaggle/input/active-learning-subset/tar_image_subset.npy' 
#shape (Num)
Labels_path = '/kaggle/input/active-learning-subset/label_target_subset.npy'

backbone_model_path = '/kaggle/input/active-learning-subset/backboneSiteA2019.pth'
fc_model_path = '/kaggle/input/active-learning-subset/fcSiteA2019.pth'

resume = True # True if you want to continue fomr the chekcpoint
checkpoint_path = "/kaggle/working/checkpoint_margin_samples400_round2_epoch1.pt"  # Replace with chekcpint file path to use
checkpoint_dir = "/kaggle/working/checkpoints" 

"""
----------------------------------------------------------------------------------------
Loss Function selection
Currently available:
"ce = cross entropy loss","ls = LabelSmoothingCrossEntropyLoss"
"""
loss_mode = "ls"

"""
---------------------------------------------------------------------------------------------
Active Learning Method(s) to use
Currently available:
"random", "entropy", "margin", "least_confident", "diversity", "entropy_diversity", "density"
"""
strategies = ['margin', ]

"""
----------------------------------------------------------------------------------------
Shape(s) of data to test on:
---(Initial samples, Iterations of AL loop, Query Sample Size)---
"""
data = [(100,5,10), (300,10,10)]

batch_size = 32
epochs = 15
variable_epochs = True # if True epochs will be 5 for first 5 rounds, 10 for next 5, and given 'epochs' for all else
test_size = 0.2
lr = 1e-1
weight_decay = 1e-4
label_smoothing = 0.1
ignore_index = -100
```
## Model
Current model is the following, which can be changed by replacing the following code, make sure the input and outputs match the input shape and number of classes, and the pretrained weigths are loaded correctly

```python 
################ CNN Backbone 
def conv_block(in_channels: int, out_channels: int, dropout=0.3) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 5, padding='same'),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(dropout),
    )

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = conv_block(6, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512)
        self.conv5 = conv_block(512, 1024)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x

 
################ Fully connected network
class FC(nn.Module):
    def __init__(self, input_dim):
        super(FC, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.fco = nn.Linear(input_dim, 3)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fco(x)
        return x



class FullModel(nn.Module):
    def __init__(self, backbone, fc):
        super().__init__()
        self.backbone = backbone
        self.fc = fc

    def forward(self, x):
        feat = self.backbone(x)
        out = self.fc(feat)
        return feat, out
        
```

Weights are loaded here

```python
#pretrained weights
map_location=torch.device(device)

backbone = cnn()
fc = FC(1024)

#loading backbone weights
state_dict = torch.load(backbone_model_path, map_location=map_location)
# Remove "module." prefix
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "")  # remove module. prefix
    new_state_dict[new_key] = v

backbone.load_state_dict(new_state_dict)

#loading fc weights
state_dict = torch.load(fc_model_path, map_location=map_location)
# Remove "module." prefix
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "")  # remove module. prefix
    new_state_dict[new_key] = v

fc.load_state_dict(new_state_dict)

model = FullModel(backbone, fc).to(device)
initial_state_dict = copy.deepcopy(model.state_dict())
```

## Adding Loss functions and Active Learning strategies

To add extra Loss Functions, modify the 'Loss' class and to add, more Active Learning strategies, modify the querysamples() function


