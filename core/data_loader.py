"""
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
from munch import Munch
from PIL import Image
import random
import glob
import copy
import torch
from torch.utils import data
from torchvision import transforms

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames
class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None
    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.samples)

class TrainFaceDataSet(data.Dataset):
    def __init__(self, data_path_list, transform=None, transform_seg=None):
        self.datasets = []
        self.num_per_folder =[]
        self.lm_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_lm_images/'
        self.mask_image_path = data_path_list[0][:data_path_list[0].rfind('/')+1] \
                             + data_path_list[0][data_path_list[0].rfind('/')+1:] + '_mask_images/'
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            self.datasets.append(image_list)
            self.num_per_folder.append(len(image_list))
        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, item):
        idx = 0
        while item >= self.num_per_folder[idx]:
            item -= self.num_per_folder[idx]
            idx += 1
        image_path = self.datasets[idx][item]
        souce_lm_image_path = self.lm_image_path  + image_path.split('/')[-1]
        souce_mask_image_path = self.mask_image_path  + image_path.split('/')[-1]
        source_image = Image.open(image_path).convert('RGB')
        source_lm_image = Image.open(souce_lm_image_path).convert('RGB')
        source_mask_image = Image.open(souce_mask_image_path).convert('L')
        if self.transform is not None:
            source_image = self.transform(source_image)
            source_lm_image = self.transform(source_lm_image)
            source_mask_image = self.transform_seg(source_mask_image)
        #choose ref from the same folder image
        temp = copy.deepcopy(self.datasets[idx]) 
        temp.pop(item)
        reference_image_path = temp[random.randint(0, len(temp)-1)]
        reference_lm_image_path = self.lm_image_path + reference_image_path.split('/')[-1]
        reference_mask_image_path = self.mask_image_path  + reference_image_path.split('/')[-1]
        reference_image = Image.open(reference_image_path).convert('RGB')
        reference_lm_image = Image.open(reference_lm_image_path).convert('RGB')
        reference_mask_image = Image.open(reference_mask_image_path).convert('L')
        if self.transform is not None:
            reference_image = self.transform(reference_image)
            reference_lm_image = self.transform(reference_lm_image)
            reference_mask_image = self.transform_seg(reference_mask_image)
        outputs=dict(src=source_image, ref=reference_image, src_lm=source_lm_image, ref_lm=reference_lm_image,
                      src_mask=1-source_mask_image, ref_mask=1-reference_mask_image)
        return outputs
    def __len__(self):
        return sum(self.num_per_folder)

class TestFaceDataSet(data.Dataset):
    def __init__(self, data_path_list, test_img_list, transform=None, transform_seg=None):
        self.source_dataset = []
        self.reference_dataset = []
        self.data_path_list = data_path_list
        self.lm_image_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_lm_images/'
        self.mask_image_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_mask_images/'
        self.biseg_parsing_path = data_path_list[:data_path_list.rfind('/')+1] \
                             + data_path_list[data_path_list.rfind('/')+1:] + '_parsing_images/'
        f=open(test_img_list,'r')
        for line in f.readlines():
            line.split(' ')
            self.source_dataset.append(line.split(' ')[0])
            self.reference_dataset.append(line.split(' ')[1])
        f.close()
        self.transform = transform
        self.transform_seg = transform_seg
    def __getitem__(self, item):
        source_image_path = self.data_path_list  + '/' + self.source_dataset[item]
        try:
            source_image = Image.open(source_image_path).convert('RGB')
        except:
            print('fail to read %s.jpg'%source_image_path)
        souce_lm_image_path = self.lm_image_path + self.source_dataset[item]
        souce_mask_image_path = self.mask_image_path  + self.source_dataset[item]
        source_parsing_image_path = self.biseg_parsing_path + self.source_dataset[item]
        source_lm_image = Image.open(souce_lm_image_path).convert('RGB')
        source_mask_image = Image.open(souce_mask_image_path).convert('L')
        source_parsing_image = Image.open(source_parsing_image_path).convert('L')
        if self.transform is not None:
            source_image = self.transform(source_image)
            source_lm_image = self.transform(source_lm_image)
            source_mask_image = self.transform_seg(source_mask_image)
            source_parsing = self.transform_seg(source_parsing_image)
        reference_image_path = self.data_path_list + '/' + self.reference_dataset[item][0:-1]
        try:
            reference_image = Image.open(reference_image_path).convert('RGB')
        except:
            print('fail to read %s.jpg' %reference_image_path)
        reference_lm_image_path = self.lm_image_path  + self.reference_dataset[item][0:-1]
        reference_mask_image_path = self.mask_image_path + self.reference_dataset[item][0:-1]
        reference_parsing_image_path = self.biseg_parsing_path + self.reference_dataset[item][0:-1]
        reference_lm_image = Image.open(reference_lm_image_path).convert('RGB')
        reference_mask_image = Image.open(reference_mask_image_path).convert('L')
        reference_parsing = Image.open(reference_parsing_image_path).convert('L')
        if self.transform is not None:
            reference_image = self.transform(reference_image)
            reference_lm_image = self.transform(reference_lm_image)
            reference_mask_image = self.transform_seg(reference_mask_image)
            reference_parsing = self.transform_seg(reference_parsing)
        outputs=dict(src=source_image, ref=reference_image, src_lm=source_lm_image, ref_lm=reference_lm_image, src_mask=1-source_mask_image,
                      ref_mask=1-reference_mask_image, src_parsing=source_parsing, ref_parsing=reference_parsing,
                      src_name=self.source_dataset[item], ref_name=self.reference_dataset[item])
        return outputs
    def __len__(self):
        return len(self.source_dataset)

def get_train_loader(root, img_size=256,
                     batch_size=8, num_workers=4):
    print('Preparing dataLoader to fetch images during the training phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transform_seg = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
    ])
    train_dataset = TrainFaceDataSet(root, transform, transform_seg)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    return train_loader

def get_test_loader(root, test_img_list, img_size=256,
                     batch_size=8, num_workers=4):
    print('Preparing dataLoader to fetch images during the testing phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transform_seg = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
    ])
    test_dataset = TestFaceDataSet(root, test_img_list, transform, transform_seg)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, drop_last=True)
    return test_loader

class InputFetcher:
    def __init__(self, loader, mode=''):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
    def _fetch_inputs(self):
        try:
            inputs_data = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            inputs_data= next(self.iter)
        return inputs_data
    def __next__(self):
        t_inputs = self._fetch_inputs()
        inputs = Munch(src=t_inputs['src'], tar=t_inputs['ref'], src_lm=t_inputs['src_lm'],
                       tar_lm=t_inputs['ref_lm'], src_mask=t_inputs['src_mask'], tar_mask=t_inputs['ref_mask'])
        if self.mode=='train':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()})
        elif self.mode=='test':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()}, src_parsing=t_inputs['src_parsing'].to(self.device),
                           tar_parsing=t_inputs['ref_parsing'].to(self.device), src_name=t_inputs['src_name'],tar_name=t_inputs['ref_name'])
        return inputs

