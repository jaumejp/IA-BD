import os
import numpy as np
import pandas as pd

from datetime import datetime
import time

from PIL import Image, ExifTags, ImageOps
# control image max-size 
Image.MAX_IMAGE_PIXELS = 201326592
Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

from src import bbox2tlbr, sqrbbox

import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

import timm

# toDO:
# orientation
# train/test split
# log file
# adjust imgSize, epochs, bSize, lRate, gamma

class maDataset(Dataset):
    
    def __init__(self, csvDataFile):
        
        self.df = pd.read_csv(csvDataFile)

        self.transform = transforms.Compose([
                transforms.RandomResizedCrop(_imgSize, scale = (0.90, 1.00)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomAffine(180, translate = (0.1, 0.1), scale = (0.75, 1.0))], 0.25),
                transforms.ToTensor(),
                transforms.Normalize(_imgNorm[0], _imgNorm[1]),
            ])

        # self.transform = transforms.Compose([
        #         transforms.Resize((_imgSize, _imgSize)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(_imgNorm[0], _imgNorm[1])
        #     ])

    def imgOpen(self, imgPath):

        pilImg = Image.open(imgPath).convert('RGB')

        img_exif = pilImg.getexif()
        orientation = 0
        if img_exif is not None:
            for key, val in img_exif.items():
                if key in ExifTags.TAGS and key == 274:
                    orientation = val
        if orientation > 1:
            pilImg = ImageOps.exif_transpose(pilImg)

        return pilImg
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
    
        # open image file
        row = self.df.iloc[idx]
        pilImg = self.imgOpen('%s/%s' %(_imgRoot, row.img_fName))
        # crop image
        bbox = sqrbbox([(row.bbx_xtl, row.bbx_ytl), (row.bbx_xbr, row.bbx_ybr)], pilImg.size)
        pilImg = pilImg.crop(bbox)
        # transform to torch tensor image
        torchImg = self.transform(pilImg)

        return {'img_fName': row.img_fName, 'image' : torchImg, 'label': [_classes.index(row.class_label)]}


def main(args):

    # device
    _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dataset
    _dataSet = maDataset(_csvdata)

    # dataloader
    _dataLoader = DataLoader(_dataSet, batch_size = args.bSize, shuffle = True, num_workers = args.workers)

    # model
    _model = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained = True,
            num_classes = len(_classes),
            global_pool = 'avg'
        )    
    _model = _model.to(_device)

    # loss function
    _loss_function = torch.nn.CrossEntropyLoss()

    # optimizer
    _optimizer = optim.Adam(_model.parameters(), lr = args.lRate)

    # optimizer
    _scheduler = optim.lr_scheduler.StepLR(_optimizer, step_size = 1, gamma = args.gamma)

    # callback
    runId = datetime.now().strftime("%Y%m%d%H%M")
    logdir = os.path.join(args.log, runId)
    os.makedirs(logdir)
    _sWriter = SummaryWriter(logdir)

    # +++ train
    _model.train()
    torch.set_grad_enabled(True)

    for epoch in range(args.epochs):

        train_loss, train_match = .0, .0
        start_time = time.time()
        
        for batch in _dataLoader:

            # +++ forward pass
            _optimizer.zero_grad()
            inputs = batch['image'].to(_device)
            output = _model(inputs)
            
            # +++ loss
            labels = torch.cat(tuple(batch['label']), dim = 0).to(_device)
            batch_loss = _loss_function(output, labels)

            # +++ backpropagation
            batch_loss.backward()
            _optimizer.step()

            # +++ evaluation
            _, preds = torch.max(torch.nn.functional.softmax(output, dim = 1), dim = 1)
            train_loss += batch_loss.data * inputs.shape[0]
            train_match += torch.sum(preds.data == labels.data)

        _sWriter.add_scalar('lRate', _scheduler.get_last_lr()[0], epoch)
        _sWriter.add_scalar('train_loss', train_loss /_dataSet.__len__(), epoch)
        _sWriter.add_scalar('train_acc', train_match /_dataSet.__len__(), epoch)

        print('+++ epoch {:3d}, {:6.4f}s Train- Loss: {:.4f} Acc: {:.4f}\n'.format(epoch, (time.time() -start_time), train_loss.item() /_dataSet.__len__(), train_match.item() /_dataSet.__len__()), end = '')

    torch.save(_model, os.path.join(args.pth, '%s.pth' %runId))


# +++++++ main 

_imgRoot = '/home/jgarriga/projects/ma24/data/test/images'
_csvdata = '/home/jgarriga/projects/ma24/data/test/phase2_test.csv'
_classes = ['aegypti', 'albopictus', 'anopheles', 'culex', 'culiseta', 'japonicus-koreicus']

_imgSize = 320 # pretrained at 384
_imgNorm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

_log = '/home/jgarriga/models/log'
_pth = '/home/jgarriga/models/pth'

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description = 'train/evaluate model')

    parser.add_argument('--seed', default = 1234, help = 'random state for train/val/test split')
    parser.add_argument('--epochs', type = int, default = 2, help = 'training epochs')
    parser.add_argument('--lRate', type = float, default = 0.00005, help = 'learning rate')
    parser.add_argument('--gamma', type = float, default = 0.995, help = 'learning rate decay')
    parser.add_argument('--bSize', type = int, default = 8, help = 'batch size')
    parser.add_argument('--workers', type = int, default = 8, help = 'number of workers')

    parser.add_argument('--log', default = _log, help = 'tensorboard log folder')
    parser.add_argument('--pth', default = _pth, help = 'model save folder')

    args = parser.parse_args()
    main(args)
