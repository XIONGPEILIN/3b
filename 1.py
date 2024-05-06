import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from tqdm import tqdm
import torch
import random
from torchvision import transforms
import torchvision
from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import time

os.environ["http_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["https_proxy"] = "http://proxy.uec.ac.jp:8080/"


def dataload(batch_size,resize=None,norm=False,type="single"):
    trans = [transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor()]
    if norm:
        normlize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        trans.insert(-1,normlize)
    if resize:
        trans.insert(0,transforms.Resize(resize))

    trans = transforms.Compose(trans)
    train_data = torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=trans,download=True)
    if type =="multi":
        sampler = DistributedSampler(train_data)
        data_loader = DataLoader(train_data,batch_size,shuffle=False,sampler=sampler)
    elif type == "single":
        data_loader = DataLoader(train_data,batch_size,shuffle=False)
    return data_loader
def multi_train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    device = torch.device('cuda',local_rank)
    dist.init_process_group(backend="nccl")

    batch_size = 128
    train_loader = dataload(batch_size,resize=(224,224),type="multi")
    

    model = resnet50.cuda(local_rank)
    ddp_model = DDP(model,[local_rank],find_unused_parameters=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(),lr=0.001,momentum=0.9)

    num_epoch = 2

    start_time = time.time()
    epochs_time_list = []
    for epoch in range(num_epoch):
        ddp_model.train()
        for imgs,val in train_loader:
            imgs = imgs.to(device)
            val = val.to(device)
            y_hat = ddp_model(imgs)
            loss = loss_fn(y_hat,val)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        epochs_time_list.append(time.time() - start_time)

    dist.destroy_process_group()
    return np.sum(epochs_time_list)

def single_train():
    batch_size = 128
    train_loader = dataload(batch_size,resize=(224,224),type="single")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet50.parameters(),lr=0.001,momentum=0.9)

    num_epoch = 2
    start_time = time.time()
    epochs_time_list = []
    
    for epoch in range(num_epoch):
        resnet50.train()
        count = 0
        for imgs,val in train_loader:
            imgs = imgs.to(device)
            val = val.to(device)
            y_hat = resnet50(imgs)
            loss = loss_fn(y_hat,val)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            count+=1
        epochs_time_list.append(time.time() - start_time)

    return np.sum(epochs_time_list)


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

resnet50 = models.resnet50(pretrained=True,progress=True)
resnet50.add_module('extra',nn.Linear(resnet50.fc.out_features,10))
device = torch.device('cuda')
resnet50 = resnet50.to(device)


if __name__ == "__main__":
    # print("multi time: ",multi_train())
    print('single time:',single_train())

    