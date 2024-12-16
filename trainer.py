import os

import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from util.tools import loginfo
from util.tsne import tsne_visual
import pylab
from torch.functional import F

class LocalTrainer(object):
    def __init__(self, args, dataset, device, model):
        super(LocalTrainer, self).__init__()
        self.dataset = dataset
        self.args = args
        self.device = device
        self.model = model

        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        sampler = DistributedSampler(dataset) if args.distributed else RandomSampler(dataset)
        self.dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size,
                                  num_workers=args.num_worker, pin_memory=True, drop_last=True)
        self.criterion = torch.nn.L1Loss().to(device)

    def train(self, comm_round, cid):
        self.model.train()
        round_loss_gather = []
        for epoch in range(1, self.args.epochs+1):
            epoch_loss_gathrer = []
            for batch_idx, batch in enumerate(self.dataloader):
                x = batch[0]
                target = batch[1]
                x = x.float().to(self.device)
                target = target.float().to(self.device)

                output, _ = self.model(x)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % self.args.print_freq == 0 :
                    loginfo('Round: {:2d} CID: {:2d} Epoch: {:2d} [{:6d}/{:6d} ({:3.0f}%)]\t Loss: {:.6f}'.format(comm_round, cid,
                        epoch, batch_idx, len(self.dataloader), 100. * batch_idx / len(self.dataloader), loss.detach().item()))
                epoch_loss_gathrer.append(loss.detach().item())

            epoch_loss_avg = sum(epoch_loss_gathrer) / len(epoch_loss_gathrer)
            round_loss_gather.append(epoch_loss_avg)

        loginfo('>> Round:{} clint:{} training complete'.format(comm_round, cid))
        for epoch, loss in enumerate(round_loss_gather):
            loginfo('Epoch: {}  Avg_loss: {:.6f}'.format(epoch+1, loss))

        return self.model.state_dict(), sum(round_loss_gather) / len(round_loss_gather)


class HorizontalTrainer(object):
    def __init__(self, args, dataset_dict, device, models, classifier, labels):
        super(HorizontalTrainer, self).__init__()
        self.dataset_dict = dataset_dict
        self.args = args
        self.device = device

        self.models = models
        self.classifier = classifier

        self.model_optimizers = [torch.optim.Adam(models[i].parameters(), lr=args.lr) for i in range(len(args.train_modals))]
        self.classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

        self.dataloaders = []
        for idx, dataset in dataset_dict.items():
            sampler = DistributedSampler(dataset[1]) if args.distributed else RandomSampler(dataset[1])
            self.dataloaders.append(DataLoader(dataset[1], sampler=sampler, batch_size=args.batch_size,
                                      num_workers=args.num_worker, pin_memory=True, drop_last=True))

        self.l1_criterion = torch.nn.L1Loss().to(device)
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(device)
        self.labels = labels
        self.iters = min([len(dataloader) for dataloader in self.dataloaders])


    def minimize_similarity(self, z_i, z_s):
        N = z_i.shape[0]
        loss = -torch.norm(z_i - z_s, p=1)
        loss = loss / N
        return loss


    def reset_optimizer_grad(self):

        for optimizer in self.model_optimizers:
            optimizer.zero_grad()

        self.classifier_optimizer.zero_grad()

    def optimzer_step(self):
        for optimizer in self.model_optimizers:
            optimizer.step()

        self.classifier_optimizer.step()

    def train(self, comm_round, epoch):
        for model in self.models:
            model.train()
        self.classifier.train()
        for batch_id, batchs in enumerate(zip(*self.dataloaders)):
            zs = None
            ys = None
            l1_loss = 0
            intra_loss = 0
            for client_id, batch in enumerate(batchs):
                x = batch[0]
                target = batch[1]
                x = x.float().to(self.device)
                target = target.float().to(self.device)
                output, z_i, z_s = self.models[client_id](x)
                y = self.labels[client_id]
                zs = torch.cat([zs, z_s], dim=0) if zs is not None else z_s
                ys = torch.cat([ys, y], dim=0) if ys is not None else y
                l1_loss += self.l1_criterion(output, target)
                #calculate similarity

                intra_loss += self.minimize_similarity(z_i, z_s)


            #classify loss
            y = self.classifier(zs)
            scores = F.log_softmax(y, dim=-1)
            ce_loss = -torch.sum(scores * ys) / ys.shape[0]

            loss = l1_loss + intra_loss * self.args.u1 + ce_loss * self.args.u2
            self.reset_optimizer_grad()
            loss.backward()
            self.optimzer_step()

            if batch_id % self.args.print_freq == 0 :
                loginfo('Round: {:2d} Epoch: {:2d} [{:6d}/{:6d} ({:3.0f}%)]\t Loss: {:.6f} Sup_Loss: {:.6f} intra_Loss: {:.6f} ce_Loss: {:.6f} '.format(comm_round,
                    epoch, batch_id, self.iters, 100. * batch_id / self.iters, loss.detach().item(), l1_loss.detach().item(), intra_loss.detach().item(), ce_loss.detach().item()))

        return [model.state_dict() for model in self.models]


class VerticalTrainer(object):
    def __init__(self, args, dataset_dict, device, models, classifier, labels):
        super(VerticalTrainer, self).__init__()
        self.dataset_dict = dataset_dict
        self.args = args
        self.device = device

        self.models = models
        self.classifier = classifier

        self.model_optimizers = [torch.optim.Adam(models[i].parameters(), lr=args.lr) for i in
                                 range(len(args.train_modals))]
        self.classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

        self.dataloaders = []
        for idx, dataset in dataset_dict.items():
            sampler = DistributedSampler(dataset[0], shuffle=False) if args.distributed else SequentialSampler(dataset[0])
            self.dataloaders.append(DataLoader(dataset[0], sampler=sampler, batch_size=args.batch_size,
                                               num_workers=args.num_worker, pin_memory=True, drop_last=True))

        self.l1_criterion = torch.nn.L1Loss().to(device)
        self.ce_criterion = torch.nn.CrossEntropyLoss().to(device)
        self.labels = labels
        self.iters = min([len(dataloader) for dataloader in self.dataloaders])


    def reset_optimizer_grad(self):

        for optimizer in self.model_optimizers:
            optimizer.zero_grad()

        self.classifier_optimizer.zero_grad()

    def optimzer_step(self):
        for optimizer in self.model_optimizers:
            optimizer.step()

        self.classifier_optimizer.step()

    def maxmize_similarity(self, x, y):

        N = x.shape[0]
        loss = torch.norm(x - y, p=1)
        loss = loss / N
        return loss

    def minimize_similarity(self, z_i, z_s):
        N = z_i.shape[0]
        loss = -torch.norm(z_i - z_s, p=1)
        loss = loss / N
        return loss

    def train(self, comm_round, epoch):
        for model in self.models:
            model.train()
        self.classifier.train()
        for batch_id, batchs in enumerate(zip(*self.dataloaders)):
            zs = None
            ys = None
            l1_loss = 0
            intra_loss = 0

            zilist = []

            for client_id, batch in enumerate(batchs):
                x = batch[0]

                target = batch[1]
                x = x.float().to(self.device)
                target = target.float().to(self.device)
                output, z_i, z_s = self.models[client_id](x)
                y = self.labels[client_id]
                zs = torch.cat([zs, z_s], dim=0) if zs is not None else z_s
                ys = torch.cat([ys, y], dim=0) if ys is not None else y
                zilist.append(z_i)
                l1_loss += self.l1_criterion(output, target)
                # calculate similarity
                intra_loss += self.minimize_similarity(z_i, z_s)

            inter_loss = 0
            S = len(zilist)
            for i in range(S-1):
                for j in range(i+1, S):
                    inter_loss += self.maxmize_similarity(zilist[i], zilist[j])

            # classify loss
            y = self.classifier(zs)
            scores = F.log_softmax(y, dim=-1)
            ce_loss = -torch.sum(scores * ys) / ys.shape[0]

            loss = l1_loss + intra_loss * self.args.u1 + ce_loss * self.args.u2 + inter_loss * self.args.u3
            self.reset_optimizer_grad()
            loss.backward()
            self.optimzer_step()

            if batch_id % self.args.print_freq == 0:
                loginfo(
                    'Round: {:2d} Epoch: {:2d} [{:6d}/{:6d} ({:3.0f}%)]\t Loss: {:.6f} Sup_Loss: {:.6f} intra_Loss: {:.6f} inter_Loss: {:.6f} ce_Loss: {:.6f} '.format(
                        comm_round,
                        epoch, batch_id, self.iters, 100. * batch_id / self.iters, loss.detach().item(),
                        l1_loss.detach().item(), intra_loss.detach().item(), inter_loss.detach().item(), ce_loss.detach().item()))

        return [model.state_dict() for model in self.models]







