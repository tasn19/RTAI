import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from train import train_model, calculate_score
from data import MyTopCropTransform, CovidDataSet
from Models.C19Xception import C19Xception
from Models.C19ResNet import C19ResNet

ap = argparse.ArgumentParser()
ap.add_argument("--image_size", type=int, default=299, help="resize images to this size")
ap.add_argument("--val_ratio", type=float, default=0.1, help="ratio of data to used for validation")
ap.add_argument("--lr", type=float, default=3e-3, help="learning rate")
ap.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
ap.add_argument("--model_name", type=str, default='xception', choices={"xception"}, help="the model architecture to "
                                                                                         "train")
ap.add_argument("--pretrained", type=bool, default=True, help="if True pre-trained weights on ImageNet will be used "
                                                              "for initialization")
ap.add_argument("--drop_out", type=bool, default=False, help="if True use drop_out before the last layer")
ap.add_argument("--BCE_pos_weight", type=int, default=50, help="the weight for the positive class error")
ap.add_argument("--train_batchsize", type=int, default=32, help="")
ap.add_argument("--models_folder", type=str, default="./", help="folder path to save the model")
ap.add_argument("--class_name", type=str, default=None, help="name to save the model, if none, the hyperparameters "
                                                             "will be used")
ap.add_argument("--num_workers", type=int, default=4, help="num_workers for dataloader")
ap.add_argument("--data_folder", type=str, default="Data/archive", help="address to the data folder")
args = vars(ap.parse_args())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
val_batchsize = 16
test_batchsize = 16


# TODO: try rescale and crop
# TODO: normalize to -1 and 1
# TODO: mix train and test for the final training(?)

# data path in computer


train_images_path = os.path.join(args['data_folder'], "train/")
train_metadata_path = os.path.join(args['data_folder'], "train.txt")
test_images_path = os.path.join(args['data_folder'], "test/")
test_metadata_path = os.path.join(args['data_folder'], "test.txt")


train_transform = transforms.Compose([
    transforms.ToTensor(),
    MyTopCropTransform(0.08),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    # transforms.Resize(size=image_size), transforms.CenterCrop(image_size), #keeps the aspect ratio
    transforms.Resize(size=(args['image_size'], args['image_size']))  # doesn't keep the aspect ratio
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    MyTopCropTransform(0.08),
    # transforms.Resize(size=image_size), transforms.CenterCrop(image_size), # keeps the aspect ratio, crops the image
    transforms.Resize(size=(args['image_size'], args['image_size']))  # doesn't keep the aspect ratio
])

train_dataset = CovidDataSet(train_metadata_path, train_images_path, train_transform)
val_dataset = CovidDataSet(train_metadata_path, train_images_path, test_transform)
test_dataset = CovidDataSet(test_metadata_path, test_images_path, test_transform)

# splitting the train dataset to train and validation
train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=args['val_ratio'])
train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

PATH = args['models_folder']
CLASS_NAME = args['class_name']
if CLASS_NAME is None:
    CLASS_NAME = args['model_name'] \
                 + "-epochs_" + str(args['num_epochs']) \
                 + "-pretrained_" + str(args['pretrained']) \
                 + "-batchsize_" + str(args['train_batchsize']) \
                 + "-posweight_" + str(args['BCE_pos_weight']) \
                 + "-lr_" + str(args['lr']) \
                 + "-drop_out_" + str(args['drop_out'])
    # saves the model with the model name WARNING: overwrites previous model is the same model name exists
model_path = os.path.join(PATH, CLASS_NAME)
history_path = os.path.join(PATH, CLASS_NAME + "-history.png")
details_history_path = os.path.join(PATH, CLASS_NAME + "-details_history.png")
CHECKPOINT_PATH = os.path.join(PATH, CLASS_NAME + "-Checkpoint")

train_dataloader = DataLoader(train_dataset, batch_size=args['train_batchsize'],
                              shuffle=True, num_workers=args['num_workers'])  # collate_fn=utils.collate_fn, pin_memory=True
val_dataloader = DataLoader(val_dataset, batch_size=val_batchsize,
                            shuffle=True, num_workers=args['num_workers'])  # collate_fn=utils.collate_fn, pin_memory=True
test_dataloader = DataLoader(test_dataset, batch_size=test_batchsize,
                             shuffle=False, num_workers=args['num_workers'])  # collate_fn=utils.collate_fn, pin_memory=True

dataloaders = {'train': train_dataloader,
               'val': val_dataloader,
               'test': test_dataloader}

logging_steps = {
    "train": len(dataloaders["train"]) // 10 + 1,
    "val": len(dataloaders["val"]) // 10 + 1,
    "test": len(dataloaders["test"]) // 10 + 1
}

dataset_sizes = {
    "train": len(train_dataset),
    "val": len(val_dataset),
    "test": len(test_dataset)
}

batch_sizes = {
    "train": args['train_batchsize'],
    "val": val_batchsize,
    "test": test_batchsize
}

if (args['model_name'] == "xception"):
    model = C19Xception(pretrained=args['pretrained'])
else:
    model = C19ResNet(model=args['model_name'], pretrained=args['pretrained'])

model.to(device)

optimizer = Adam(model.parameters(), lr=args['lr'])
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args['BCE_pos_weight']]).to(device), reduction='mean')
# criterion = nn.BCEWithLogitsLoss()
# can change pos_weight increasing pos_weight increases the sensitivity
#  of the positive class
# https://discuss.pytorch.org/t/weights-in-bcewithlogitsloss/27452

print("train size:", len(train_dataset))
print("val size:", len(val_dataset))
print("test size:", len(test_dataset))

model, loss_history, score_history = train_model(model, criterion, optimizer, args['num_epochs'],
                                                 dataloaders, logging_steps, dataset_sizes, batch_sizes,
                                                 device, CHECKPOINT_PATH)

plt.subplot(2, 2, 1)
plt.plot(score_history['val']['sp'], label="validation")
plt.plot(score_history['train']['sp'], label="train")
plt.legend()
plt.title("sp")
plt.subplot(2, 2, 2)
plt.plot(score_history['val']['sn'], label="validation")
plt.plot(score_history['train']['sn'], label="train")
plt.legend()
plt.title("sn")
plt.subplot(2, 2, 3)
plt.plot(score_history['val']['pp'], label="validation")
plt.plot(score_history['train']['pp'], label="train")
plt.legend()
plt.title("pp")
plt.subplot(2, 2, 4)
plt.plot(score_history['val']['pn'], label="validation")
plt.plot(score_history['train']['pn'], label="train")
plt.legend()
plt.title("pn")
plt.savefig(details_history_path)
plt.show()

plt.subplot(1, 2, 1)
plt.plot(score_history['val']['score'], label="validation")
plt.plot(score_history['train']['score'], label="train")
plt.legend()
plt.title("score")
plt.subplot(1, 2, 2)
plt.plot(loss_history['val'], label="validation")
plt.plot(loss_history['train'], label="train")
plt.legend()
plt.title("loss")
plt.savefig(history_path)
plt.show()

# Saving the model

torch.save(model.state_dict(), model_path)
