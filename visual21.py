from glob import glob
from skimage import color
from skimage.exposure import adjust_gamma
from skimage.util import img_as_float
import SimpleITK as sitk
import os
from skimage import io
import matplotlib.pyplot as plt
import imageio
from imgaug import augmenters as iaa
import numpy as np


def show_segmented_image(orig_img, pred_img):
    # Show the prediction over the original image
    # INPUT:
    #     1)orig_img: the test image, which was used as input
    #     2)pred_img: the prediction output
    # OUTPUT:
    #     segmented image rendering

    orig_img = sitk.GetArrayFromImage(sitk.ReadImage(orig_img))
    # orig_img = orig_img[0,:]
    pred_img = sitk.GetArrayFromImage(sitk.ReadImage(pred_img))
    orig_img = orig_img / np.max(orig_img)
    ones = np.argwhere(pred_img == 1)
    twos = np.argwhere(pred_img == 2)
    fours = np.argwhere(pred_img == 4)
    # pred_img =
    gray_img = img_as_float(orig_img)
    gray_pred = img_as_float(pred_img)
    image = adjust_gamma(color.gray2rgb(gray_img), 1)
    pred = adjust_gamma(color.gray2rgb(gray_pred), 1)
    sliced_image = image.copy()
    sliced_pred = pred.copy()
    red_multiplier = [1, 0.2, 0.2]
    yellow_multiplier = [1, 1, 0.25]
    green_multiplier = [0.35, 0.75, 0.25]

    for i in range(len(ones)):
        sliced_image[ones[i][0]][ones[i][1]][ones[i][2]] = red_multiplier
        sliced_pred[ones[i][0]][ones[i][1]][ones[i][2]] = red_multiplier
    for i in range(len(twos)):
        sliced_image[twos[i][0]][twos[i][1]][twos[i][2]] = green_multiplier
        sliced_pred[twos[i][0]][twos[i][1]][twos[i][2]] = green_multiplier
    for i in range(len(fours)):
        sliced_image[fours[i][0]][fours[i][1]][fours[i][2]] = yellow_multiplier
        sliced_pred[fours[i][0]][fours[i][1]][fours[i][2]] = yellow_multiplier
    return sliced_image,sliced_pred
def save_visual_examples(input_dir,input_lab_dir ,pre_u_dir,pre_final_dir,output_dir, case_number, z_number, z,save):
    # save_dir = os.path.join(output_dir, 'case_images/case_{}'.format(case_number))
    save_dir = os.path.join(output_dir, 'case_images/case_'+z)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("ddd")


    img= sitk.GetArrayFromImage(sitk.ReadImage(input_dir+'.nii.gz'))
    print(img.shape)


    # gt ,gt_s= show_segmented_image("/home/user/4TB/datasets/MICCAI_BraTS2020_TrainingData/"+case_number+"/"+case_number+'_flair.nii.gz',"/home/user/4TB/datasets/MICCAI_BraTS2020_TrainingData/"+case_number+"/"+case_number+ '_seg.nii.gz')
    gt, gt_s = show_segmented_image(
        "/home/user/4TB/datasets/RSNA_ASNR_MICCAI_BraTS2021_trainall/BraTS2021_" + case_number + "/BraTS2021_" + case_number + '_flair.nii.gz',
        "/home/user/4TB/datasets/RSNA_ASNR_MICCAI_BraTS2021_trainall/BraTS2021_" + case_number + "/BraTS2021_" + case_number + '_seg.nii.gz')
    pred_u, pred_u_s = show_segmented_image(
        "/home/user/4TB/datasets/RSNA_ASNR_MICCAI_BraTS2021_trainall/BraTS2021_" + case_number + "/BraTS2021_" + case_number + '_flair.nii.gz',
        pre_u_dir + '.nii.gz')
    pred_final, pred_final_s = show_segmented_image(
        "/home/user/4TB/datasets/RSNA_ASNR_MICCAI_BraTS2021_trainall/BraTS2021_" + case_number + "/BraTS2021_" + case_number + '_flair.nii.gz',
        pre_final_dir + '.nii.gz')
    # gt = sitk.GetArrayFromImage(sitk.ReadImage(input_lab_dir + '_seg.nii.gz'))
    print(gt.shape)
    if z =='sagittal':
        for i in range(4):
            input = img[i]
            # print(input.shape)
            input = input / np.max(input)
            gray_img = img_as_float(input)
            gray_img = adjust_gamma(color.gray2rgb(gray_img), 1)
            # slice = gray_img[z_number,:,:]

            slice = gray_img[:, :, z_number]
            # slice = gray_img[:,  z_number,:]
            io.imsave(save_dir + '/L{}_input{}_s.png'.format(fname,i), slice)
        gt = gt[:, :,z_number]
        io.imsave(save_dir + '/L{}_gt_s.png'.format(fname), gt)
        pred_u = pred_u[ :, :,z_number]
        io.imsave(save_dir + '/L{}_pred_u_s.png'.format(fname), pred_u)
        pred_final = pred_final[:, :,z_number]
        io.imsave(save_dir + '/L{}_unet_pred_f_s.png'.format(fname), pred_final)
    if z =='horizontal':
        for i in range(4):
            input = img[i]
            print(input.shape)
            input = input / np.max(input)
            gray_img = img_as_float(input)
            gray_img = adjust_gamma(color.gray2rgb(gray_img), 1)
            slice = gray_img[z_number,:,:]
            #
            # slice = gray_img[:, :, z_number]
            # slice = gray_img[:,  z_number,:]
            io.imsave(save_dir + '/L{}_input{}_h.png'.format(fname,i), slice)
        gt = gt[z_number,:, :]
        io.imsave(save_dir + '/L{}_gt_h.png'.format(fname), gt)
        gt_s = gt_s[z_number, :, :]
        io.imsave(save_dir + '/L{}_gt_s_h.png'.format(fname), gt_s)
        pred_u = pred_u[ z_number,:, :]
        io.imsave(save_dir + '/L{}_pred_u_h.png'.format(fname), pred_u)
        pred_final = pred_final[z_number,:, :]
        io.imsave(save_dir + '/L{}_unet_pred_f_h.png'.format(fname), pred_final)
    if z =='coronal':
        for i in range(4):
            input = img[i]
            print(input.shape)
            input = input / np.max(input)
            gray_img = img_as_float(input)
            gray_img = adjust_gamma(color.gray2rgb(gray_img), 1)
            slice = gray_img[:,  z_number,:]
            io.imsave(save_dir + '/L{}_input{}_c.png'.format(fname,i), slice)
        gt = gt[:,  z_number,:]
        io.imsave(save_dir + '/L{}_gt_c.png'.format(fname), gt)
        pred_u = pred_u[:,  z_number,:]
        io.imsave(save_dir + '/L{}_pred_u_c.png'.format(fname), pred_u)
        pred_final = pred_final[:,  z_number,:]
        io.imsave(save_dir + '/L{}_unet_pred_f_c.png'.format(fname), pred_final)



if __name__ == "__main__":

    data = sorted(glob("/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/all/*.nii.gz"))

    z = ["horizontal","sagittal","coronal"]

    for i in range(len(data)):
        fname = data[i].split("/")[-1].split(".")[0]
        fname = fname.split("_")[-1]
        print(fname)
        simple = r'/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2021_train/images/BraTS2021_' + fname
        simple_lab = r'/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2021_train/labels/BraTS2021_' + fname
        pre_dir_u = r'/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/all/BraTS21_' + fname
        pre_dir_final = r'/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/all/BraTS21_' + fname
        save_file = '/home/user/4TB/Chenbonian/medic-segmention/isnet/visualization/'
        save_visual_examples(simple, simple_lab, pre_dir_u, pre_dir_final, save_file, fname, 97, z[0], False)