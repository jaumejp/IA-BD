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
        
'''
Abans només feiem unes transformacións, mida de la xarxa i pasar a tensor. El que es fa habitualment, tenim el dataloader, tenim el shuffle
Tenim les img, per tenir batch diferents i errors diferents, ajuda a la xarxa faci tant over fitting. 

Per potenciar més això, fem aquestes transformacions una mica més elaborades. 

Tenim. Filp Horitzontal, Vertical, random resize (amb la escala de 90 a 100% de la img), retalla una mica (entre 90 i 100 random) perd una 
mica però no molt, per no machacar el mosquit, retallada petita per canviar una mica la imatge. 

La redimensionem a la mateixa mida que volem, la hem de tornar a fer més gran. (random resize crop)
Documentació de torch vision i hi ha molts tipuus de transformacions. 

Altres interessants, que comporten cert perill, però poden fer entrenament més robust. 

Podem regularitzar (mesures per evitar over fitting) capa de drop out (apagar determinades neurones aleatoriament) amb això fem que la xarxa
de cop i volta ha de ser més robusta perquè li desapareix info que podia ser clau. 

Una altre transformació que es en aquest sentit, cut out o cut holes, fa forats a la imatge (sería com un drop out però abans).

Cada vegada que cridem una imatge, cada vegada li pasem una imatge diferent a la xarxa (perque hi ha el parametre probabilitat i són transforms diferents. 

Posar mica de soroll, borrosa, etc.

Com menys imatges tenim, més important és això. 

=====

Altre novetat: funció image Open:

'''

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
    
'''
En comptes de obrir amb el image open per defecte, ho fem amb el self (que es la nostre que estem definit a dalt)

Quan s'obre, es necessita el bounding box. com que són fotos agafades amb movil, a les metadades hi guarda la horientació, 180º. 

En cas de que estiguin rotades, pill l'obre tal cual, però open CV, que es un altre modul, n'hi ha un que es mira aquesta info. 

I, quan obre, ja la gira, les veus totes com si s'hagues agafat amb el movil a 0 (apaisat). Pill si hem fet vertical, la obre i la gira. 

Això és important, perquè si no saps com s'han tret els bounding boxes. Si quan entrenem no tenim en compte això. 

Si tenim unes coordenades amb vertical i després nosaltres la posem en horitzontal, els bbx no cuadraran. 

Per això sempre es treballa amb horientación 0. 
'''
    
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

'''
__getitem__ : generem el item que li pasarem el dataloader, que aquest s'encarregara de passar-ho a la gpu.
'''
    

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

_imgRoot = '/home/jaume/Notebook/mosquits/phase2_test/test/final/'
_csvdata = '/home/jaume/Notebook/mosquits/phase2_test.csv'
_classes = ['aegypti', 'albopictus', 'anopheles', 'culex', 'culiseta', 'japonicus-koreicus']

_imgSize = 320 # pretrained at 384
_imgNorm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

_log = '/home/jaume/models/log'
_pth = '/home/jaume/models/pth'

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
    
'''
argparse: per afegir arguments, per defecte són strings. 
Sheduler, modifica el learning rate, per en joan és millor treballar amb un learning rate que creixi, learning rate al principi petit, anem fent mica en mica, a mida que els errors van disminuint, incrementem el error. Això ho controla el gamma. Si es 1 és constant.
'''

'''
Callback, maneres de tenir informació de com evoluciona el proccés d'entrenament. No interessa només la foto final, sino que també interesa
veure si al llarg del entrenament, el que hauríem d'esperar es que la funció de loss vagi decreixent, i la funció d'accuracy, la presició
que es van classificant els exemples, una curba creixent puja molt rapid i s'estabilitza, i la funció de los baixa molt rapid i s'estabilitza. 

Si el de validació fa una U, vol dir que hi hagi un overfitting, perque fa molt bé les de train però les de test no en sap perque hi ha overfitting. 

L'accuraci faria la u al revés perque puja bé l'accuracy però quan fa l'overfitting l'accuracy comença a baixar. 

- Callback, és una crida al proccés d'entrenament. 3 maneres diferents d'implementar-los.

La que fa servir en Joan: fa serivr el tensor work, i una classe que es el SummaryWriter, on volem fer el log (carpeta)

Genera una id, de cada run (tot un entrenament) python train.py --ddd

Genera un nom de subfolder on guardarà els logs (events) que guarda el summary writer els gugardarà en aquella sub carpeta i els pots comparar.

.addScalar, genera una serie d'events, segons el scheduler i gama inicial,
una altre serie per la funció de loss. (una serie es una llista). 

Cada .addScalar, afegeix a la serie.
Per cada epoch, guardar acc, loss, 
'''