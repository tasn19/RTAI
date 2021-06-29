import argparse
import time
import copy
from tqdm.auto import tqdm
import os

import torch

from sklearn.metrics import precision_score, recall_score


def calculate_score(y_true, y_pred, return_separates=False):
    score = 6 * recall_score(y_true, y_pred, pos_label=1) \
            + 5 * recall_score(y_true, y_pred, pos_label=0) \
            + 3 * precision_score(y_true, y_pred, pos_label=1) \
            + 2 * precision_score(y_true, y_pred, pos_label=0)

    if return_separates:
        return score, \
               recall_score(y_true, y_pred, pos_label=1), \
               recall_score(y_true, y_pred, pos_label=0), \
               precision_score(y_true, y_pred, pos_label=1), \
               precision_score(y_true, y_pred, pos_label=0)
    else:
        return score


def run_epoch(model, criterion, optimizer, phase,
              dataloaders, logging_steps, dataset_sizes, batch_sizes, device="cuda"):
    all_pred_probs = torch.tensor([])
    all_labels = torch.tensor([])
    running_loss = 0

    if phase == "train":
        model.train()
    else:
        model.eval()

    for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase]),
                                    leave=False,
                                    total=len(dataloaders[phase])):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            outputs = outputs.squeeze()

            loss = criterion(outputs, labels.float())

            if phase == "train":
                loss.backward()
                optimizer.step()

        all_pred_probs = torch.cat((all_pred_probs, outputs.detach().clone().sigmoid().cpu()))
        all_labels = torch.cat((all_labels, labels.detach().clone().cpu()))
        running_loss += loss.item() * inputs.size(0)

        if (i % logging_steps[phase] == 0):  # and (i>0):
            avg_loss = running_loss / ((i + 1) * batch_sizes[phase])

            print("[{}]: {} | loss : {:.4f} | score : {:.4f}"
                  .format(phase, i // logging_steps[phase],
                          avg_loss, calculate_score(all_labels, (all_pred_probs > 0.5).type(torch.int8))))

    epoch_loss = running_loss / dataset_sizes[phase]
    return epoch_loss, all_labels, all_pred_probs


def train_model(model, criterion, optimizer, num_epochs,
                dataloaders, logging_steps, dataset_sizes, batch_sizes,
                device="cuda", CHECKPOINT_PATH="./model_checkpoint"):
    since = time.time()

    if os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        startE = checkpoint['epoch']
    else:
        startE = -1

    best_model_wts = copy.deepcopy(model.state_dict())
    epoch_loss, all_labels, all_pred_probs = run_epoch(model, criterion, optimizer, 'test',
                                                       dataloaders, logging_steps, dataset_sizes, batch_sizes,
                                                       device)
    best_test_score, sp, sn, pp, pn = calculate_score(all_labels, (all_pred_probs > 0.5).type(torch.uint8),
                                                      return_separates=True)

    loss_history = {"val": [], "train": []}
    score_history = {"val": {'score': [], 'sp': [], 'sn': [], 'pp': [], 'pn': []},
                     "train": {'score': [], 'sp': [], 'sn': [], 'pp': [], 'pn': []}}

    for epoch in range(startE + 1, num_epochs):
        epoch_time = time.time()
        print("epoch {}/{}".format(epoch + 1, num_epochs))

        for phase in ["train", "val"]:
            epoch_loss, all_labels, all_pred_probs = run_epoch(model, criterion, optimizer, phase,
                                                               dataloaders, logging_steps, dataset_sizes, batch_sizes,
                                                               device)
            score, sp, sn, pp, pn, = calculate_score(all_labels, (all_pred_probs > 0.5).type(torch.int8),
                                                     return_separates=True)

            print(
                "---[{}] Epoch {}/{} | time: {:.4f} | Loss : {:.4f} | Score: {:.4f} [sp:{:.4f}-sn:{:.4f}-pp:{:.4f}-pn:{:.4f}]"
                    .format(phase, epoch + 1, num_epochs,
                            time.time() - epoch_time, epoch_loss, score, sp, sn, pp, pn))
            loss_history[phase].append(epoch_loss)
            score_history[phase]['score'].append(score)
            score_history[phase]['sp'].append(sp)
            score_history[phase]['sn'].append(sn)
            score_history[phase]['pp'].append(pp)
            score_history[phase]['pn'].append(pn)

        epoch_loss, all_labels, all_pred_probs = run_epoch(model, criterion, optimizer, phase='test')
        test_score, sp, sn, pp, pn = calculate_score(all_labels, (all_pred_probs > 0.5).type(torch.int8),
                                                     return_separates=True)
        if test_score >= best_test_score:
            print(
                "************** best model update: Test score: {:.4f} [sp:{:.4f}-sn:{:.4f}-pp:{:.4f}-pn:{:.4f}] "
                "**********".format(test_score, sp, sn, pp, pn))
            best_test_score = test_score
            best_model_wts = copy.deepcopy(model.state_dict())

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_model_state_dict': best_model_wts},
                   CHECKPOINT_PATH)

        print("------ Epoch {}/{} finished. Best test score: {:.4f} ----------------------"
              .format(epoch + 1, num_epochs, best_test_score))
        print()

    time_elapsed = time.time() - since
    print(f"training took {time_elapsed} seconds")
    print(f"best score: {best_test_score}")
    model.load_state_dict(best_model_wts)

    return model, loss_history, score_history
