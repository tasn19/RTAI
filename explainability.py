import time
import os, sys
import torch
from Models.C19Xception import C19Xception
import matplotlib.pyplot as plt
from train import run_epoch, calculate_score
from tqdm import tqdm
import argparse
from torchvision import transforms
from data import MyTopCropTransform
from torch.utils.data import DataLoader
from data import CovidDataSet, ImageSortingDataSet
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, roc_curve, precision_score

sys.path.insert(1, "./grad-cam-pytorch-master")
import grad_cam

device = "cpu"

threshold = 0.35
model_path = "TrainedModels/xception-epochs_10-pretrained_True-batchsize_32-posweight_50-lr_0.003"

test_images_path = os.path.join("./Data/archive/", "test/")
test_metadata_path = os.path.join("./Data/archive/", "test.txt")

test_transform = transforms.Compose([
    transforms.ToTensor(),
    MyTopCropTransform(0.08),
    # transforms.Resize(size=image_size), transforms.CenterCrop(image_size), # keeps the aspect ratio, crops the image
    transforms.Resize(size=(299, 299))  # doesn't keep the aspect ratio
])

model = C19Xception(pretrained=False)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

test_dataset = CovidDataSet(test_metadata_path, test_images_path, test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 num_workers=0)


ims, labels = next(iter(test_dataloader))
gcam = grad_cam.GradCAM(model=model)

# probs, ids = gcam.forward(ims)
# gcam.backward(ids=torch.tensor([[0]]))
# print("possible target layers:")
# print(gcam.fmap_pool.keys())
# print(gcam.grad_pool.keys())

target_layers = ["backbone.conv1", 'backbone.bn2']

for target_layer in target_layers:

    probs, ids = gcam.forward(ims)
    gcam.backward(ids=torch.tensor([[0]]))
    regions = gcam.generate(target_layer=target_layer)

    # plt.imshow(ims.permute(2,3,1,0).squeeze())
    # plt.imshow(regions.permute(2,3,1,0).squeeze(), cmap='Reds', alpha=0.5)
    # plt.show()

    plt.subplot(1,2,1).imshow(ims.permute(2,3,1,0).squeeze())
    plt.subplot(1,2,2).imshow(regions.permute(2,3,1,0).squeeze(), cmap='jet')
    plt.show()