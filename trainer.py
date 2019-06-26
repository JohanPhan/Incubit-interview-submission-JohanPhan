#from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn

from model import *
from progress_bar import progress_bar
import torch.nn.functional as F
from utils import *
import numpy as np
class Trainer(object):
    def __init__(self, model, config, training_loader, testing_loader):
        super(Trainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda:1' if self.GPU_IN_USE else 'cpu')
        self.model = model
        self.lr = config["lr"]
        self.nEpochs = config["nEpochs"]
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.upscale_factor = config["upscale_factor"]
        self.training_loader = training_loader
        self.testing_loader = testing_loader
    def build_model(self):
        self.model = self.model.to(self.device)     
        self.model.weight_init(mean=0.0, std=0.02)
        self.criterion = torch.nn.L1Loss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20, 40, 60, 80, 100], gamma=0.5)

    def save(self, path):
        model_out_path = path
        torch.save(self.model, model_out_path).to(self.device)  
        print("Checkpoint saved to {}".format(model_out_path))
    def load(self, path):
        model_out_path = path
        self.model = torch.load(model_out_path).to(self.device)  
        print("Loaded Checkpoint from {}".format(model_out_path))
    def cMSE_lost(self, lr, hr):
        diff = hr-lr
        b = torch.mean(diff)
        diff -= b
        cMSE = torch.mean(diff * diff)
        return cMSE
    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, sm, target) in enumerate(self.training_loader):
            data, target = data.float().to(self.device), target.float().to(self.device)
            self.optimizer.zero_grad()
            SR_output = self.model(data)
            loss = self.criterion(SR_output, target)
            loss2 = self.cMSE_lost(SR_output, target)
            train_loss += loss.item() + 100 * loss2.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))
        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, sm, target) in enumerate(self.testing_loader):
                data, target = data.float().to(self.device), target.float().to(self.device)
                prediction = self.model(data)
                loss = self.cMSE_lost(prediction,target )
                psnr = 10 * log10(loss.item())
                avg_psnr += -psnr
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save()
    def rerun(self):
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save()                