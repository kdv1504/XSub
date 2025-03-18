# Importing packages
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import shap
import time
import random
from utils import config 
from utils.utils import robust_selection, find_topk_shap, find_golden_img, find_golden_shap_img, pixel_swap, pert_change
from dataprep import dataloader
from models import Trainer

opt = config.get_arguments().parse_args()

# Set up CUDA memory
use_cuda = True
print("CUDA: ", torch.cuda.is_available())
device = torch.cuda.device("cuda" if use_cuda else "cpu")
print("Using", torch.cuda.device_count(), "GPUs")

torch.manual_seed(opt.seed)

# Setting up classifier
if opt.dataset == 'breast_ultrasound':
    model = models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')
    for param in model.parameters():
        param.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, opt.label_count)
    model.to(opt.device)

model_path = opt.save_dir + 'models/model_' + opt.dataset + '_' + str(opt.epochs) + '.pt'
if not os.path.exists(model_path): # Train from scratch
    print("Training model from scratch.")

    train = dataloader(opt.data_root, opt.dataset, opt.bs, opt.num_workers, split=0.8, train=True)
    test = dataloader(opt.data_root, opt.dataset, opt.bs, opt.num_workers, split=0.8, train=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Fine-tuning the model
    trainer = Trainer(opt, model, opt.dataset, criterion, optimizer, 
                      train, test, lr_scheduler, opt.epochs, opt.save_period)
    trainer.train()
    trainer.test()

    # Free up memory
    del train
    del test

model.load_state_dict(torch.load(model_path))
model.to(opt.device)
print("Model loaded.")

# Load data for attack
data = dataloader(opt.data_root, opt.dataset, opt.bs, opt.num_workers)

# Generate SHAP values and store golden images
golden_img_path = opt.save_dir + 'golden_img_' + opt.dataset + '.pt'
golden_shap_path = opt.save_dir + 'golden_shap_' + opt.dataset + '.pt'
if not os.path.exists(golden_img_path):
    find_golden_shap_img(data, model, golden_img_path, golden_shap_path)
golden_image = torch.load(golden_img_path)
golden_shap = torch.load(golden_shap_path)
print("Golden image and SHAP values loaded.")
    
# Hyperparameters
alpha = [100]
beta = [100]
strategies = ['no clip'] #['clipping', 'no clip']
success_arr = []
start_time = time.time()

# Main attack loop
for a in alpha:
        for b in beta:
            for strat in strategies:
                t_success, u_success, total_samples, pert = 0, 0, 0, 0
                for batch_idx, (images, labels) in enumerate(data):
                    images, labels = images.to(opt.device), labels.to(opt.device)
                    total_samples += images.shape[0]
                    if batch_idx == 0:
                        e = shap.GradientExplainer(model, images)
                        print("Successfully initialized explainer.")
                    for i in range(images.shape[0]):
                        true_label = labels[i]
                        target_label = random.randint(0, opt.label_count-1)
                    while target_label == true_label:
                        target_label = random.randint(0, opt.label_count-1)
                    true_img = images[i]
                    target_img = golden_image[target_label]
                    true_shp = torch.sum(torch.tensor(e.shap_values(torch.unsqueeze(images[i], 0))), 1)[0,:,:,true_label]
                    target_shp = golden_shap[target_label]
                    im = pixel_swap(true_img, target_img, true_shp, target_shp, opt.topk, opt.topk, a, b, strat)
                    im = im.to(opt.device)
                    pert += pert_change(true_img, im)
                    _, im_pred = torch.max(model(torch.unsqueeze(im, 0)).data, 1)
                    if im_pred == target_label:
                        t_success +=1
                        u_success +=1
                    elif im_pred != true_label:
                        u_success +=1
                t_return = t_success / total_samples
                u_return = u_success / total_samples
                pert = pert / total_samples
                print(f'Success rate = {round(t_return, 4)} and {round(u_return, 4)}') 
                print(f'Average L2-norm perturbation: {round(pert, 4)}')
                success_arr.append([strat, a, b, t_return, u_return, pert])
                print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
            
save_path = opt.save_dir + 'results/' + opt.dataset + '_results.txt'
with open(save_path, 'w') as f:
    for line in success_arr:
        line = line[1:-1]
        f.write(f"{line}\n")
