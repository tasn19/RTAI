import torch
from Models.C19Xception import C19Xception
# from tqdm import tqdm
import argparse
from torchvision import transforms
from data import MyTopCropTransform
from torch.utils.data import DataLoader
from data import ImageListDataSet
import sys
import warnings

warnings.filterwarnings("ignore")


ap = argparse.ArgumentParser("AI against Covid eval")
ap.add_argument("ims_address", nargs="*", type=str)
ap.add_argument("model_path", type=str, help="model to evaluate")
ap.add_argument("--threshold", type=float, default=0.35, help="probability threshold for the positive case")
ap.add_argument("--batch_size", type=int, default=32, help="batch_size for loading test images")
ap.add_argument("--num_workers", type=int, default=0, help="num_workers for the dataloader")
args = vars(ap.parse_args())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.ToTensor(),
    MyTopCropTransform(0.08),
    # transforms.Resize(size=image_size), transforms.CenterCrop(image_size), # keeps the aspect ratio, crops the image
    transforms.Resize(size=(299, 299))  # doesn't keep the aspect ratio
])

model = C19Xception(pretrained=False)
model.to(device)
model.load_state_dict(torch.load(args['model_path'], map_location=device))

ims_address = sys.argv[1].replace('"', "").replace("'", "").replace(" ", "").strip(']').strip('[').split(',')

# print("ims address", ims_address)
# print("batch_size", args['batch_size'])
# print("num_workers", args['num_workers'])
# print("model_path", args['model_path'])
# print("threshold", args["threshold"])

competition_dataset = ImageListDataSet(ims_address, test_transform)
competition_dataloader = DataLoader(competition_dataset, batch_size=args['batch_size'], shuffle=False,
                                 num_workers=args['num_workers'])  # collate_fn=utils.collate_fn, pin_memory=True

# L = os.listdir(competition_test_path)
# L.sort(key=lambda x: int(os.path.splitext(x)[0]))
model.to(device)
model.eval()
preds = []
for ims in competition_dataloader:
    ims = ims.to(device)
    with torch.set_grad_enabled(False):
        if ims.size()[0] > 1:
            preds.extend(model(ims).detach().clone().squeeze())
        else:
            preds.extend(model(ims).detach().clone())
    # print(preds)
# for p in tqdm(L):
#     im_path = os.path.join(competition_test_path, p)
#     image = Image.open(im_path).convert("RGB")
#     image_tensor = test_transform(image)
#     image_tensor = torch.unsqueeze(image_tensor, 0).to(device)
#
#     preds.append(model.forward(image_tensor).detach().clone().squeeze())

probs = torch.tensor(preds).sigmoid()

# for p in probs > args['threshold']:
#     print(int(p))
# print([int(p) for p in probs > args['threshold']])
print("[", end="")
for p in probs > args['threshold']:
    print(int(p), end=",")
print("]", end="")
