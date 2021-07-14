import torch
import os
from Models.C19Xception import C19Xception
import matplotlib.pyplot as plt
from train import run_epoch, calculate_score
import PIL.Image as Image
from tqdm import tqdm
import argparse
from torchvision import transforms
from data import MyTopCropTransform
from torch.utils.data import DataLoader
from data import CovidDataSet, ImageSortingDataSet
import torch.nn as nn
from torch.optim import Adam

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", type=str, default="./TrainedModels/xception-epochs_10-pretrained_True-batchsize_32"
                                                  "-posweight_50-lr_0.003", help="model to evaluate")
ap.add_argument("--threshold", type=float, default=0.35, help="probability threshold for the positive case")
ap.add_argument("--print_test", type=bool, default=False, help="print results on the provided test images")
ap.add_argument("--show_hist", type=bool, default=False, help="show histogram of the output probabilities")
ap.add_argument("--batch_size", type=int, default=32, help="batch_size for loading test images")
ap.add_argument("--num_workers", type=int, default=4, help="num_workers for the dataloader")
ap.add_argument("--image_size", type=int, default=299, choices={299}, help="image_size must match the model's input")
ap.add_argument("--test_data_path", type=str, default="./Data/archive/", help="folder with a folder containing test "
                                                                              "images (/test/) and test metadata ("
                                                                              "test.txt), only needed if print_test "
                                                                              "is True")
ap.add_argument("--competition_test_path", type=str, default="./Data/competition_test/", help="folder with "
                                                                                              "competition test "
                                                                                              "images")
args = vars(ap.parse_args())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

threshold = args['threshold']
model_path = args['model_path']

test_images_path = os.path.join(args['test_data_path'], "test/")
test_metadata_path = os.path.join(args['test_data_path'], "test.txt")

test_transform = transforms.Compose([
    transforms.ToTensor(),
    MyTopCropTransform(0.08),
    # transforms.Resize(size=image_size), transforms.CenterCrop(image_size), # keeps the aspect ratio, crops the image
    transforms.Resize(size=(args['image_size'], args['image_size']))  # doesn't keep the aspect ratio
])

model = C19Xception(pretrained=False)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# TODO clean the code for print_test: just evaluate the dataset results instead of calling run_epoch
if args['print_test']:
    test_dataset = CovidDataSet(test_metadata_path, test_images_path, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False,
                                 num_workers=args['num_workers'])  # collate_fn=utils.collate_fn, pin_memory=True

    dataloaders = {'test': test_dataloader}
    logging_steps = {"test": len(dataloaders["test"]) // 10 + 1}
    dataset_sizes = {"test": len(test_dataset)}
    batch_sizes = {"test": args['batch_size']}

    optimizer = Adam(model.parameters(), lr=3e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50]).to(device), reduction='mean')

    epoch_loss, all_labels, all_pred_probs = run_epoch(model, criterion, optimizer, 'test',
                                                       dataloaders, logging_steps, dataset_sizes, batch_sizes,
                                                       device)
    test_score, sp, sn, pp, pn = calculate_score(all_labels, (all_pred_probs > threshold).type(torch.uint8), return_separates=True)

    print("Loaded model test score: {:.6f}, sp: {:.6f}, sn: {:.6f}, pp:{:.6f}, pn:{:.6f}".format(test_score, sp, sn, pp, pn))
    if args['show_hist']:
        n, bins, patches = plt.hist(all_pred_probs, 100, facecolor='blue', alpha=0.5)
        plt.show()

competition_test_path = args['competition_test_path']
competition_dataset = ImageSortingDataSet(competition_test_path, test_transform)
competition_dataloader = DataLoader(competition_dataset, batch_size=args['batch_size'], shuffle=False,
                                 num_workers=args['num_workers'])  # collate_fn=utils.collate_fn, pin_memory=True

# L = os.listdir(competition_test_path)
# L.sort(key=lambda x: int(os.path.splitext(x)[0]))
model.to(device)
model.eval()
preds = []
for ims in tqdm(competition_dataloader):
    ims = ims.to(device)
    with torch.set_grad_enabled(False):
        preds.extend(model(ims).detach().clone().squeeze())
# for p in tqdm(L):
#     im_path = os.path.join(competition_test_path, p)
#     image = Image.open(im_path).convert("RGB")
#     image_tensor = test_transform(image)
#     image_tensor = torch.unsqueeze(image_tensor, 0).to(device)
#
#     preds.append(model.forward(image_tensor).detach().clone().squeeze())

probs = torch.tensor(preds).sigmoid()

if args['show_hist']:
    n, bins, patches = plt.hist(probs, 100, facecolor='blue', alpha=0.5)
    plt.show()

for p in probs > threshold:
    print(int(p))