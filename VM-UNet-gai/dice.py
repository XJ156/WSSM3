import os
import csv
import numpy as np
import cv2

def dice_coef(truth, pred):
    intersection = np.sum(np.logical_and(truth, pred))
    total_pixels = np.sum(truth) + np.sum(pred)
    dice = (2. * intersection + 1.) / (total_pixels + 1.)
    return dice

def calculate_dice_coef(folder_truth, folder_pred, output_csv):
    truth_files = os.listdir(folder_truth)
    pred_files = os.listdir(folder_pred)

    dices = []
    with open(output_csv, mode='w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File', 'Dice Coefficient'])
        for truth_file in truth_files:
            if truth_file.endswith('.jpg') or truth_file.endswith('.png'):
                truth_path = os.path.join(folder_truth, truth_file)
                pred_file = truth_file.split('.')[0] + '.png'
                pred_path = os.path.join(folder_pred, pred_file)
                print('Truth:', truth_path)
                print('Pred:', pred_path)
                truth = cv2.imread(truth_path, cv2.IMREAD_GRAYSCALE) > 0
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) > 0
                dice = dice_coef(truth, pred)
                writer.writerow([truth_file, dice])
                dices.append(dice)

    mean_dice = np.mean(dices)
    return mean_dice

folder_truth = '/disk/sdb/zhangqian0620/VM-UNet-main/data/olp/val/masks'
folder_pred = '/disk/sdb/zhangqian0620/VM-UNet-main/output_olp_300_test'
output_csv = 'vmunet_olp.csv'
mean_dice = calculate_dice_coef(folder_truth, folder_pred, output_csv)
print('Mean Dice Coefficient:', mean_dice)
