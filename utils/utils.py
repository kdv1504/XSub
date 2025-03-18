import numpy as np
import torch
from utils import config
import shap
import os

opt = config.get_arguments().parse_args()

def robust_selection(data, shp, orig_labels, pred_labels):
    '''Function to select images that are correctly predicted.'''
    for i in range(orig_labels.shape[0]):
        if orig_labels[i] == pred_labels[i]:
            if i == 0:
                new_data = torch.unsqueeze(data[i], 0)
                new_labels = [orig_labels[i]]
                new_shp = torch.unsqueeze(shp[i], 0)
            else:
                new_data = np.append(new_data, torch.unsqueeze(data[i], 0), 0)
                new_labels.append(orig_labels[i])
                new_shp = np.append(new_shp, torch.unsqueeze(shp[i], 0), 0)
    return torch.tensor(new_data), torch.tensor(new_labels), torch.tensor(new_shp)

def find_topk_shap(shp, topk):
    '''
    Find top-k positions that have the highest SHAP values
    '''
    (rows, cols) = shp.shape
    topk_ids = []
    while topk > 0:
        max_shap = 0
        ids = [0, 0]
        for j in range(rows):
            for k in range(cols):
                if [j,k] not in topk_ids:
                    curr_max = shp[j,k]
                    if curr_max > max_shap:
                        max_shap = curr_max
                        ids = [j,k]
        topk_ids.append(ids)
        topk = topk - 1
    return topk_ids

def find_golden_img(shap_values, golden_shap, label_count=opt.label_count):
    golden_indices = []
    for label in range(label_count):
        golden_idx = 0
        for i in range(shap_values.shape[0]):
            if golden_shap is not None:
                current_max = torch.max(golden_shap[label, :, :])
            else: 
                current_max = torch.max(shap_values[golden_idx, :, :, label])
            if torch.max(shap_values[i, :, :, label]) > current_max:
                golden_idx = i
        golden_indices.append(golden_idx)
    return golden_indices

def find_golden_shap_img(data, model, golden_img_path, golden_shap_path):
    for batch_idx, (images, labels) in enumerate(data):
        images, labels = images.to(opt.device), labels.to(opt.device)

        # Generate SHAP values
        if batch_idx == 0:
            e = shap.GradientExplainer(model, images)
            print("Successfully initialized explainer.")

            # Load stored SHAP values
            if not os.path.exists(golden_shap_path):
                golden_shap = None
            else:
                golden_shap = torch.load(golden_shap_path)
        agg_shap = torch.sum(torch.tensor(e.shap_values(images)), 1)

        # Find and temporarily store golden images
        golden_shap = None
        golden_indices = find_golden_img(agg_shap, golden_shap)

        # Update golden SHAP values
        golden_shap = agg_shap[golden_indices]
        for i in range(golden_shap.shape[0]):
            if i == 0:
                to_save = torch.unsqueeze(golden_shap[i, :, :, i], 0)
            else:
                to_save = torch.cat((to_save, torch.unsqueeze(golden_shap[i, :, :, i], 0)), 0)

        # Store SHAP values and golden images
        torch.save(to_save, golden_shap_path)
        torch.save(images[golden_indices], golden_img_path)

# Main perturbation function
def pixel_swap(true_img, target_img, true_shp, target_shp, topk_true, topk_target, alpha, beta, strategy='clipping'):
    '''Strategy for K > 1'''
    true_topk = find_topk_shap(true_shp, topk_true)
    target_topk = find_topk_shap(target_shp, topk_target)
    to_return = torch.clone(true_img)
    for i in range(len(true_topk)):
        a = true_topk[i]
        b = target_topk[i]
        for j in range(true_img.shape[0]):
            subt = -alpha * to_return[j, a[0], a[1]]
            add = beta * target_img[j, b[0], b[1]]
            to_return[j, a[0], a[1]] = to_return[j, a[0], a[1]] + subt + add
            if strategy == 'clipping':
                if to_return[j, a[0], a[1]] > 1:
                    to_return[j, a[0], a[1]] = 1
                elif to_return[j, a[0], a[1]] < 0:
                    to_return[j, a[0], a[1]] = 0
    return to_return

# Calculate perturbation rate
def pert_change(orig, pert):
    '''
    Parameters:
    orig = original image
    pert = perturbed image
    '''

    pert_rate = round(torch.norm((pert.cpu()- orig.cpu()).cuda(), 2).item(),2)

    return pert_rate