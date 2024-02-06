import os
from glob import glob
from subprocess import call

import nibabel as nib
import numpy as np
from scipy.ndimage import label


def to_lbl(pred, p,v):
    enh = pred[2]
    c1, c2, c3 = pred[0] > p[0], pred[1] > p[1], pred[2] > p[2]
    pred = (c1 > 0).astype(np.uint8)
    pred[(c2 == False) * (c1 == True)] = 2
    pred[(c3 == True) * (c1 == True)] = 4

    components, n = label(pred == 4)
    for et_idx in range(1, n + 1):
        _, counts = np.unique(pred[components == et_idx], return_counts=True)
        if 1 < counts[0] and counts[0] < 8 and np.mean(enh[components == et_idx]) < 0.9:
            pred[components == et_idx] = 1

    et = pred == 4
    if 0 < et.sum() and et.sum() < v and np.mean(enh[et]) < 0.90:  # 500 150 73 35 15  0.9
        pred[et] = 1

    pred = np.transpose(pred, (2, 1, 0)).astype(np.uint8)
    return pred
def no_to_lbl(pred, p,v):
    enh = pred[2]
    c1, c2, c3 = pred[0] > 0.5, pred[1] > 0.5, pred[2] > 0.5
    pred = (c1 > 0).astype(np.uint8)
    pred[(c2 == False) * (c1 == True)] = 2
    pred[(c3 == True) * (c1 == True)] = 4
    #
    # components, n = label(pred == 4)
    # for et_idx in range(1, n + 1):
    #     _, counts = np.unique(pred[components == et_idx], return_counts=True)
    #     if 1 < counts[0] and counts[0] < 8 and np.mean(enh[components == et_idx]) < 0.9:
    #         pred[components == et_idx] = 1
    #
    # et = pred == 4
    # if 0 < et.sum() and et.sum() < v and np.mean(enh[et]) < 0.90:  # 500 150 73 35 15  0.9
    #     pred[et] = 1

    pred = np.transpose(pred, (2, 1, 0)).astype(np.uint8)
    return pred

def dataset_choise(fname,p,years,bast):
    if years==2019:
        img = nib.load(
            f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2019_val/images/{fname}.nii.gz")  # BraTS2020_00001.nii.gz
        # print(fname)
        fname = fname.split("_")
        # print('ddd',fname)
        nib.save(
            nib.Nifti1Image(p, img.affine, header=img.header),
            os.path.join(save_path, "BraTS19_" + fname[1]+"_" +fname[2]+"_"+fname[3]+ ".nii.gz"),
        )
    if years==2020:
        if bast==2:
            img = nib.load(
                    f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_train/images/{fname}.nii.gz")
            fname = fname.split("_")[-1]
            nib.save(
                nib.Nifti1Image(p, img.affine, header=img.header),
                os.path.join(save_path, "BraTS20_Training_" + fname + ".nii.gz"),)
        else:
            img = nib.load(
                f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_val/images/{fname}.nii.gz")
        # img = nib.load(
        #     f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2020_train/images/{fname}.nii.gz")
            fname = fname.split("_")[-1]
            # print(fname[2:],fname)
            nib.save(
                nib.Nifti1Image(p, img.affine, header=img.header),
                os.path.join(save_path, "BraTS20_Validation_" + fname + ".nii.gz"),)
            # os.path.join(save_path, "BraTS20_Training_" + fname + ".nii.gz"),)
    if years == 2021:
        # print("dd")
        img = nib.load(f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2021_train/images/{fname}.nii.gz")
        fname = fname.split("_")[-1]
        nib.save(
            nib.Nifti1Image(p, img.affine, header=img.header),
            os.path.join(save_path, "BraTS21_" + fname + ".nii.gz"), )
    if years == 2023:
        img = nib.load(f"/home/user/4TB/Chenbonian/medic-segmention/BraTS_dataset/BraTS2023_train/images/{fname}.nii.gz")
        # fname = fname.split("_")[-1]
        nib.save(
            nib.Nifti1Image(p, img.affine, header=img.header),
            os.path.join(save_path, fname + ".nii.gz"), )
def prepare_preditions(e,p,v,years,post,bast):
    fname = e[0].split("/")[-1].split(".")[0]
    preds = [np.load(f) for f in e]
    if post == 1:
        p = to_lbl(np.mean(preds, 0),p,v)
    else: p = no_to_lbl(np.mean(preds, 0),p,v)

    dataset_choise(fname,p,years,bast)
    # p =



post =1
if post == 1:
    # ps=[[0.5,0.5,0.5],[0.7,0.3,0.4]]
    ps = [ [0.7, 0.3, 0.4]]
    # ps = [[0.45,0.4,0.45]]
else : ps=[[0,0,0]]
# ps = [[0.5,0.5,0.5]]
vs = [500]
versions = [58]
years = 2021
fold = 0
bast = 0



for version in versions:
    for p in ps:
        for v in vs:
            if bast==1:
                save_path = f"/home/user/4TB/Chenbonian/medic-segmention/isnet/results/new-bast-last-v{version}-1000-{v}-{p[0]}-{p[1]}-{p[2]}-tta-{years}"
            elif bast==2:
                save_path = f"/home/user/4TB/Chenbonian/medic-segmention/isnet/results/new-bast-last-v{version}t-1000-{v}-{p[0]}-{p[1]}-{p[2]}-tta-{years}"

            else:
                save_path = f"/home/user/4TB/Chenbonian/medic-segmention/isnet/results/new-last-v{version}-1000-{v}-{p[0]}-{p[1]}-{p[2]}-tta-{years}"

            os.makedirs(save_path)


            if years==2020:
                preds = sorted(glob(f"/home/user/4TB/Chenbonian/medic-segmention/isnet/predictions_last-v{version}_task=19_fold={fold}_tta"))
                # preds = sorted(glob(
                # f"/home/user/4TB/Chenbonian/medic-segmention/isnet/predictions_epoch=609-dice=89_36_task=19_fold=0_tta"))

            elif years==2019:
                preds = sorted(glob(
                    f"/home/user/4TB/Chenbonian/medic-segmention/results/predictions_last-v{version}_task=18_fold={fold}_tta"))
            elif years==2021:
                # print("ddd")
                preds = sorted(glob(
                    f"/home/user/4TB/Chenbonian/medic-segmention/isnet/predictions_last-v{version}_task=15_fold={fold}_tta"))

            elif years==2023:
                preds = sorted(glob(
                    f"/home/user/4TB/Chenbonian/medic-segmention/isnet/predictions_last-v{version}_task=19_fold={fold}_tta"))
            # print(preds)
            examples = list(zip(*[sorted(glob(f"{p}/*.npy")) for p in preds]))
            print(f"Preparing final predictions v{version}-1000-{v}-{p[0]}-{p[1]}-{p[2]}-tta",len(examples))
            print(f"Preparing save_path {save_path.split('/')[-1]}")

            for e in examples:
                prepare_preditions(e,p,v,years,post,bast)
                print("Finished:",e)
            print("Finished!")
