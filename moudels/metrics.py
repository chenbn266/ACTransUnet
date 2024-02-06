# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from monai.metrics import compute_meandice, do_metric_reduction, HausdorffDistanceMetric,compute_dice
from monai.networks.utils import one_hot
from torchmetrics import Metric


class Dice(Metric):
    def __init__(self, n_class, brats):
        super().__init__(dist_sync_on_step=False)
        self.n_class = n_class
        self.brats = brats
        self.HD95 = HausdorffDistanceMetric(percentile=95)
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros((n_class,)), dist_reduce_fx="sum")
        self.add_state("hd95", default=torch.zeros((n_class,)), dist_reduce_fx="sum")

    def update(self, p, y, l):
        if self.brats:
            p = (torch.sigmoid(p) > 0.5).int()
            y_wt, y_tc, y_et = y > 0, ((y == 1) + (y == 3)) > 0, y == 3
            y = torch.stack([y_wt, y_tc, y_et], dim=1)

        else:
            p, y = self.ohe(torch.argmax(p, dim=1)), self.ohe(y)

        self.steps += 1
        self.loss += l
        self.dice += self.compute_metric(p, y, compute_dice, 1, 0)
        # self.hd95 = self.compute_metric_HD(p,y,0,373.12866)
        if self.brats:
            pwt, ptc, pet = torch.split(p, split_size_or_sections=1, dim=1)
            ywt, ytc, yet = torch.split(y, split_size_or_sections=1, dim=1)
            pwt[pwt>0],ptc[ptc>0],pet[pet>0] = 1,1,1
            ywt[ywt>0],ytc[ytc>0],yet[yet>0] = 1,1,1

            if ywt.sum() > 0:
                if pwt.sum() > 0:
                    hd95_wt = self.HD95(pwt, ywt)
                    hd95_wt = hd95_wt[0, 0]
                else:
                    hd95_wt = 373.1287
            else:
                if pwt.sum() > 0:
                    hd95_wt = 373.1287
                else:
                    hd95_wt = 0
            if ytc.sum()>0:
                if ptc.sum()>0:
                    hd95_tc = self.HD95(ptc, ytc)
                    hd95_tc=hd95_tc[0, 0]
                else: hd95_tc = 373.1287
            else:
                if ptc.sum()>0:
                    hd95_tc=  373.1287
                else: hd95_tc = 0


            if yet.sum()>0:
                if pet.sum()>0:
                    hd95_et = self.HD95(pet, yet)
                    hd95_et= hd95_et[0, 0]
                else:hd95_et = 373.1287
            else:
                if pet.sum()>0:
                    hd95_et = 373.1287
                else: hd95_et = 0
            self.hd95[0] += hd95_wt
            self.hd95[1] += hd95_tc
            self.hd95[2] += hd95_et
        # else:
        #     for i in range(len(p)):
        #         p[i][p[i] > 0]= 1
        #         y[i][y[i] > 0]= 1
        #         if y[i].sum()>0:
        #             self.hd95[i] += self.HD95(p, y)
        #         else:
        #             self.hd95[i]+=0
        # self.hd95+= [hd95_wt,hd95_tc,hd95_et]
        # self.hd95 = torch.cat([hd95_wt[0],hd95_tc[0],hd95_et[0]],dim=0)
        # print(f"hd{self.steps}", hd95_et[0, 0], self.hd95)
        # self.hd95+=self.hd95

    def compute(self):
        return 100 * self.dice / self.steps, self.loss / self.steps, self.hd95/self.steps

    def ohe(self, x):
        return one_hot(x.unsqueeze(1), num_classes=self.n_class + 1, dim=1)

    def compute_metric(self, p, y, metric_fn, best_metric, worst_metric):
        metric = metric_fn(p, y, include_background=self.brats)
        metric = torch.nan_to_num(metric, nan=worst_metric, posinf=worst_metric, neginf=worst_metric)
        metric = do_metric_reduction(metric, "mean_batch")[0]

        for i in range(self.n_class):
            if (y[:, i] != 1).all():
                metric[i - 1] += best_metric if (p[:, i] != 1).all() else worst_metric
        # print("dice",metric)
        return metric
    # def compute_metric_HD(self,p,y,best_metric,worst_metric):
    #     pwt, ptc, pet = torch.split(p, split_size_or_sections=1, dim=1)
    #     ywt, ytc, yet = torch.split(y, split_size_or_sections=1, dim=1)
    #     # pwt[pwt>0],ptc[ptc>0],pet[pet>0] = 1,1,1
    #     # ywt[ywt>0],ytc[ytc>0],yet[yet>0] = 1,1,1
    #     hd95_wt = self.HD95(pwt, ywt)
    #     hd95_tc = self.HD95(ptc, ytc)
    #     hd95_et = self.HD95(pet, yet)
    #     # self.hd = self.HD95(p,y)
    #
    #     # metric = [hd95_wt[[0]],hd95_tc[[0]],hd95_et[[0]]]
    #     print("hd", hd95_wt, hd95_tc, hd95_et)
    #     return hd95_wt, hd95_tc, hd95_et
