import os
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from Models.C19Xception import C19Xception
from data import CovidDataSet
from data import MyTopCropTransform

import cv2

sys.path.insert(1, "./grad-cam-pytorch-master")  # TODO: make it universal (windows addresses)
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

gcam = grad_cam.GradCAM(model=model)

# probs, ids = gcam.forward(ims)
# gcam.backward(ids=torch.tensor([[0]]))
# print("possible target layers:")
# print(gcam.fmap_pool.keys())
# print(gcam.grad_pool.keys())

target_layers = dict(model.named_modules()).keys()
target_layers = ['backbone.conv1', 'backbone.conv4']

im_num = 5

for i, (im, labels) in enumerate(test_dataloader):
    if i >= im_num:
        break
    for target_layer in target_layers:
        probs, ids = gcam.forward(im)
        gcam.backward(ids=ids)
        regions = gcam.generate(target_layer=target_layer)
        # regions = regions/max(regions)

        # plt.imshow(ims.permute(2,3,1,0).squeeze())
        # plt.imshow(regions.permute(2,3,1,0).squeeze(), cmap='Reds', alpha=0.5)
        # plt.show()

        im2 = im.permute(2, 3, 1, 0).squeeze()
        regions = regions.permute(2, 3, 1, 0).squeeze()
        cmap = plt.get_cmap('jet')
        combined = 0.5 * im2 + 0.5 * cmap(regions)[:, :, :3]
        plt.subplot(1, 3, 1).imshow(im2)
        plt.subplot(1, 3, 2).imshow(regions, cmap='jet')
        plt.subplot(1, 3, 3).imshow(combined)
        plt.show()
