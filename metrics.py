import logging
import os
import re

import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import cv2
from skimage import color
from PIL import Image
import seaborn as sns
import scipy.stats as stats
from skimage import measure
import tensorflow as tf
import binary_metric as bm

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)


def compute_metrics_on_patient(input_fold):
    data = h5py.File(os.path.join(input_fold, 'pred.hdf5'), 'r')
    file_names = []
    # measures per structure:
    dices_list = []
    hausdorff_list = []
    prec_list = []
    sens_list = []
    vol_list = []
    vol_err_list = []
    vol_gt_list = []
    for paz in np.unique(data['paz'][()]):
        pred_arr = []  # predizione del modello
        mask_arr = []  # ground truth
        flag = 1
        for i in np.where(data['paz'][()] == paz)[0]:
            pred_arr.append(data['pred'][i])
            mask_arr.append(data['mask'][i])
            if flag:
                px_size = data['pixel_size'][i]
                flag = 0

        pred_arr = np.transpose(np.asarray(pred_arr, dtype=np.uint8), (1, 2, 0))
        mask_arr = np.transpose(np.asarray(mask_arr, dtype=np.uint8), (1, 2, 0))

        gt_binary = (mask_arr == 1) * 1
        pred_binary = (pred_arr == 1) * 1
        volpred = pred_binary.sum() * (float(px_size[0]) * float(px_size[1])) * float(px_size[2]) / 1000.
        volgt = gt_binary.sum() * (float(px_size[0]) * float(px_size[1])) * float(px_size[2]) / 1000.

        vol_list.append(volpred)  # volume predetto CNN
        vol_err_list.append(volpred - volgt)
        vol_gt_list.append(volgt)  # volume reale

        temp_dice = 0
        hd_max = 0
        temp_rec = 0
        temp_prec = 0
        count = 0
        for zz in range(gt_binary.shape[2]):
            slice_pred = np.squeeze(pred_binary[:,:,zz])
            slice_gt = np.squeeze(gt_binary[:,:,zz])

            if slice_gt.sum() == 0 and slice_pred.sum() == 0:
                temp_dice += 1
                hd_value = 0
                temp_rec += 1
                temp_prec += 1
            elif slice_pred.sum() == 0 and slice_gt.sum() > 0:
                temp_dice += 0
                hd_value = 1
                temp_rec += 0
                temp_prec += 0
            elif slice_pred.sum() != 0 and slice_gt.sum() != 0:
                temp_dice += bm.dc(slice_pred, slice_gt)
                hd_value = bm.hd95(slice_gt, slice_pred, (float(px_size[0]), float(px_size[1])), connectivity=2)
                temp_rec += bm.recall(slice_pred, slice_gt)
                temp_prec += bm.precision(slice_pred, slice_gt)
            if hd_max < hd_value:
                hd_max = hd_value
            count += 1
        dices_list.append(temp_dice / count)
        hausdorff_list.append(hd_max)
        sens_list.append(temp_rec / count)
        prec_list.append(temp_prec / count)
        file_names.append(paz)

    df = pd.DataFrame({'dice': dices_list, 'hd': hausdorff_list,
                        'vol': vol_list, 'vol_gt': vol_gt_list, 'vol_err': vol_err_list,
                        'paz': file_names, 'recall': sens_list, 'prec': prec_list})
    data.close()
    return df


def compute_metrics_on_slice(input_fold):
    data = h5py.File(os.path.join(input_fold, 'pred.hdf5'), 'r')
    file_names = []
    # measures per structure:
    dices_list = []
    hausdorff_list = []
    prec_list = []
    sens_list = []
    vol_list = []
    vol_err_list = []
    vol_gt_list = []
    for i in range(len(data['pred'][()])):
        pred_binary = (data['pred'][i] == 1) * 1
        gt_binary = (data['mask'][i] == 1) * 1
        px_size = data['pixel_size'][i]

        areapred = pred_binary.sum() * (float(px_size[0]) * float(px_size[1])) * 1 / 1000.
        areagt = gt_binary.sum() * (float(px_size[0]) * float(px_size[1])) * 1 / 1000.

        vol_list.append(areapred)  # volume predetto CNN
        vol_err_list.append(areapred - areagt)
        vol_gt_list.append(areagt)  # volume reale

        slice_pred = np.squeeze(pred_binary)
        slice_gt = np.squeeze(gt_binary)

        if slice_gt.sum() == 0 and slice_pred.sum() == 0:
            dices_list.append(1)
            hausdorff_list.append(0)
            sens_list.append(1)
            prec_list.append(1)
        elif slice_pred.sum() == 0 and slice_gt.sum() > 0:
            dices_list.append(0)
            hausdorff_list.append(1)
            sens_list.append(0)
            prec_list.append(0)
        elif slice_pred.sum() != 0 and slice_gt.sum() != 0:
            dices_list.append(bm.dc(slice_pred, slice_gt))
            hausdorff_list.append(bm.hd95(slice_gt, slice_pred, (float(px_size[0]), float(px_size[1])), connectivity=2))
            sens_list.append(bm.recall(slice_pred, slice_gt))
            prec_list.append(bm.precision(slice_pred, slice_gt))
        file_names.append(data['paz'][i])

    df = pd.DataFrame({'dice': dices_list, 'hd': hausdorff_list,
                        'vol': vol_list, 'vol_gt': vol_gt_list, 'vol_err': vol_err_list,
                        'paz': file_names, 'recall': sens_list, 'prec': prec_list})
    data.close()
    return df


def print_latex_tables(path, file_name, file_out):
    file = os.path.join(path, file_name)
    out_file = os.path.join(path, file_out)
    data = pd.read_pickle(file)
    dice = data['dice']
    hd = data['hd']
    rec = data['recall']
    prec = data['prec']
    with open(out_file, "w") as text_file:
        text_file.write('dice mean:%0.3f (%0.3f)\n' % (np.mean(dice), np.std(dice, ddof=1)))
        text_file.write('hd mean:%0.3f (%0.3f)\n' % (np.mean(hd), np.std(hd, ddof=1)))
        text_file.write('rec mean:%0.3f (%0.3f)\n' % (np.mean(rec), np.std(rec, ddof=1)))
        text_file.write('prec mean:%0.3f (%0.3f)\n' % (np.mean(prec), np.std(prec, ddof=1)))
        q50, q75, q25 = np.percentile(dice, [50, 75, 25])
        text_file.write('dice median:%0.3f (%0.3f - %0.3f)\n' % (q50, q25, q75))
        q50, q75, q25 = np.percentile(hd, [50, 75, 25])
        text_file.write('hd median:%0.3f (%0.3f - %0.3f)\n' % (q50, q25, q75))
        q50, q75, q25 = np.percentile(rec, [50, 75, 25])
        text_file.write('rec median:%0.3f (%0.3f - %0.3f)\n' % (q50, q25, q75))
        q50, q75, q25 = np.percentile(prec, [50, 75, 25])
        text_file.write('prec median:%0.3f (%0.3f - %0.3f)\n' % (q50, q25, q75))
    return 0


def main(path_pred):
    print(path_pred)
    if os.path.exists(os.path.join(path_pred, 'pred.hdf5')):
        path_eval = os.path.join(path_pred, 'evaluation')
        if not os.path.exists(path_eval):
            tf.io.gfile.makedirs(path_eval)
            print(path_eval)
            df = compute_metrics_on_patient(path_pred)
            df.to_pickle(os.path.join(path_eval, 'df_paz.pkl'))
            df.to_excel(os.path.join(path_eval, 'excel_df_paz.xlsx'))
            print_latex_tables(path_eval, 'df_paz.pkl', 'paz_tables.txt')

            df = compute_metrics_on_slice(path_pred)
            df.to_pickle(os.path.join(path_eval, 'df_slice.pkl'))
            df.to_excel(os.path.join(path_eval, 'excel_df_slice.xlsx'))
            print_latex_tables(path_eval, 'df_slice.pkl', 'slice_tables.txt')

    print('saving images...')
    pdf_path = os.path.join(path_eval, 'plt_imgs.pdf')
    data = h5py.File(os.path.join(path_pred, 'pred.hdf5'), 'r')
    figs = []
    for i in range(len(data['img_raw'])):
        img_raw = cv2.normalize(src=data['img_raw'][i], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                dtype=cv2.CV_8U)
        mask = data['mask'][i].astype(np.uint8)
        pred = data['pred'][i].astype(np.uint8)
        fig = plt.figure(figsize=(14, 14))
        ax1 = fig.add_subplot(131)
        ax1.set_axis_off()
        ax1.imshow(img_raw, cmap='gray')

        ax2 = fig.add_subplot(132)
        ax2.set_axis_off()
        ax2.imshow(
            color.label2rgb(mask, img_raw, colors=[(255, 0, 0), (0, 255, 0), (255, 255, 0)], alpha=0.0008, bg_label=0,
                            bg_color=None))
        ax3 = fig.add_subplot(133)
        ax3.set_axis_off()
        ax3.imshow(
            color.label2rgb(pred, img_raw, colors=[(255, 0, 0), (0, 255, 0), (255, 255, 0)], alpha=0.0008, bg_label=0,
                            bg_color=None))
        ax1.title.set_text('Raw_img')
        ax2.title.set_text('Ground truth')
        ax3.title.set_text('Automated')
        txt = str(data['paz'][i] + '_' + str(i))
        plt.text(0.1, 0.65, txt, transform=fig.transFigure, size=18)
        figs.append(fig)
        #plt.show()
    data.close()

    with PdfPages(pdf_path) as pdf:

        for fig in figs:
            pdf.savefig(fig)
            plt.close()
