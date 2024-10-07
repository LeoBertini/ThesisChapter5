#on the NHM machine the following PyTorch installation should be used to invoke GPU compatibility
# the pytorch installation should be done inside the virtual environment or the default python environment
# py -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

import sys
sys.path.append('c:\\users\\ctlablovelace\\appdata\\local\\programs\\python\\python39\lib\\site-packages')
sys.path.append('C:\\Users\\ae20067\\.conda\\envs\\CoralWORMS\\lib\\site-packages')
import torch
print(f"PyTorch version: {torch.__version__}")


# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")