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
from data import CovidDataSet

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", type=str, default="./TrainedModels/xception-epochs_10-pretrained_True-batchsize_32-posweight_50-lr_0.003", help="model to evaluate")
ap.add_argument("--threshold", type=float, default=0.35, help="model to evaluate")
ap.add_argument("--print_test", type=bool, default=False, help="print results on test data")
ap.add_argument("--image_size", type=int, default=299)
args = vars(ap.parse_args())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

threshold = args['threshold']
model_path = args['model_path']

test_images_path = "./Data/archive/test/"
test_metadata_path = "./Data/archive/test.txt"

test_transform = transforms.Compose([
    transforms.ToTensor(),
    MyTopCropTransform(0.08),
    # transforms.Resize(size=image_size), transforms.CenterCrop(image_size), # keeps the aspect ratio, crops the image
    transforms.Resize(size=(args['image_size'], args['image_size']))  # doesn't keep the aspect ratio
])
test_dataset = CovidDataSet(test_metadata_path, test_images_path, test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)  # collate_fn=utils.collate_fn, pin_memory=True

dataloaders = {'test': test_dataloader}
logging_steps = {"test": len(dataloaders["test"]) // 10 + 1}
dataset_sizes = {"test": len(test_dataset)}
batch_sizes = {"test": 16}

model = C19Xception(pretrained=False)
model.to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

if args['print_test']:
    epoch_loss, all_labels, all_pred_probs = run_epoch(model, None, None, phase='test')
    test_score, sp, sn, pp, pn = calculate_score(all_labels, (all_pred_probs > threshold).type(torch.uint8), return_separates=True)

    print("Loaded model test score: {:.6f}, sp: {:.6f}, sn: {:.6f}, pp:{:.6f}, pn:{:.6f}".format(test_score, sp, sn, pp, pn))
    n, bins, patches = plt.hist(all_pred_probs, 100, facecolor='blue', alpha=0.5)
    plt.show()

competition_test_path = "./Data/competition_test/"

L = os.listdir(competition_test_path)
L.sort(key=lambda x: int(os.path.splitext(x)[0]))
model.to(device)
model.eval()
preds = []
for p in tqdm(L):
    im_path = os.path.join(competition_test_path, p)
    image = Image.open(im_path).convert("RGB")
    image_tensor = test_transform(image)
    image_tensor = torch.unsqueeze(image_tensor, 0).to(device)

    preds.append(model.forward(image_tensor).detach().clone().squeeze())

probs = torch.tensor(preds).sigmoid()

# n, bins, patches = plt.hist(probs, 100, facecolor='blue', alpha=0.5)
# plt.show()

for p in probs > threshold:
    print(int(p))