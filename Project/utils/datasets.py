import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import random
from shutil import copyfile
import shutil
import os 
from torchvision.transforms import ToTensor#用于把图片转化为张量
import numpy as np#用于将张量转化为数组，进行除法




class CustomImageDataset(Dataset):
    '''
        自定义数据集类，用于加载图像数据集。
        参数：
            data_dir (str): 数据集的根目录。
            classes (list): 类别名称列表。
            transform (torchvision.transforms.Compose, optional): 数据预处理操作。默认为None。
        返回：
            image (torch.Tensor): 图像张量。
            class_idx (int): 类别索引。
    '''
    def __init__(self, data_dir, classes,transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}#根据类别获取索引
        self.img_paths = self.get_image_paths()#每个图像和对应的索引对
        #self.check_datadir(self.check_datadir)
    def check_datadir(self,dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(self.data_dir)
    def get_image_paths(self):
        img_paths = []
        for cls in self.classes:
            class_dir = os.path.join(self.data_dir, cls)#对应类别的数据集图像路径
            class_idx = self.class_to_idx[cls]#对应类别的索引
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                img_paths.append(
                        (
                            img_path, class_idx#每个图像和对应的索引对
                        )
                    )
        return img_paths
    def __len__(self):
        return len(self.img_paths)#数据集大小

    def __getitem__(self, idx):
        '''
            获取图像张量和索引
            参数：
            idx (int): 索引。
            返回：
            image (torch.Tensor): 图像张量。
            class_idx (int): 类别索引。
        '''
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self.img_paths))
            ls = self.img_paths[start:stop]
            res=[]
            for imgpath in ls:
                img_path, class_idx=imgpath
                image = Image.open(img_path).convert('RGB')
        
                if self.transform is not None:
                    image = self.transform(image)
                res.append((image, class_idx))#图像向量和类别索引
            return res
        else:
            img_path, class_idx = self.img_paths[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image, class_idx#


def split_data(data_dir,target_dir,label,sep=[0.7,0.25,0.05]):
    '''
        对每个类别为一个 图像文件夹的数据，进行数据集划分，并保存到对应的文件夹（target_dir)中
        参数：
            data_dir (str): 数据集的根目录。
            target_dir (str): 划分后的数据集的根目录。
            label (str): 类别名称。
            sep (list): 分割比例，列表形式，包含训练集、验证集和测试集的比例。默认为[0.7, 0.25, 0.05]。
    '''
    os.makedirs(os.path.join(target_dir, 'train',label), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val',label), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test',label), exist_ok=True)
    import glob

    image_files = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    random.shuffle(image_files)

    # 计算分割点
    total_images = len(image_files)
    train_split = int(sep[0] * total_images)
    val_split = int(sep[1] * total_images)

    # 将图片复制到对应的目标文件夹
    for i, image_file in enumerate(image_files):
        if i < train_split:
            dest_dir = os.path.join(target_dir, 'train',label)
        elif i < train_split + val_split:
            dest_dir = os.path.join(target_dir, 'val',label)
        else:
            dest_dir = os.path.join(target_dir, 'test',label)
        
        # 获取图片文件名并复制到目标文件夹
        image_filename = os.path.basename(image_file)
        copyfile(image_file, os.path.join(dest_dir, image_filename))



def clear_folder(folder_path):
    '''
        遍历删除文件夹结构下所有文件但保留文件夹
        参数：
            folder_path (str): 文件夹路径。
    '''
    # 检查目标文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历目标文件夹内的所有文件和子文件夹
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # 如果是文件，则直接删除
            if os.path.isfile(item_path):
                os.remove(item_path)
            # 如果是子文件夹，则递归清除子文件夹内的内容
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)



from torchvision.datasets import ImageFolder#用于导入图片数据集
def get_mean_std(dir):
    '''
        计算数据集的均值和标准差
        参数：
            dir (str): 数据集的根目录。
        返回：
            mean (numpy.ndarray): 均值。
            std (numpy.ndarray): 标准差。
    '''
    means = [0,0,0]
    std = [0,0,0]#初始化均值和方差
    transform=ToTensor()#可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
    dataset=ImageFolder(dir,transform=transform)#导入数据集的图片，并且转化为张量
    num_imgs=len(dataset)#获取数据集的图片数量
    for img,a in tqdm(dataset,desc='get_mean_std'):#遍历数据集的张量和标签
        for i in range(3):#遍历图片的RGB三通道
            # 计算每一个通道的均值和标准差
            means[i] += img[i, :, :].mean()
            std[i] += img[i, :, :].std()
    mean=np.array(means)/num_imgs
    std=np.array(std)/num_imgs#要使数据集归一化，均值和方差需除以总图片数量
    return mean,std #打印出结果


def get_transform(
        chance='train',
        resize_size=337,
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ):

    '''
        获取预设的数据处理方法，包含数据增强
        参数：
            chance (str): 数据处理方式，默认为train。
            resize_size (int): 缩放尺寸。
            mean (list): 均值。
            std (list): 标准差。
        返回：
            transforms.Compose: 数据处理方法。
    '''
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(resize_size, scale=(0.6, 1.0), ratio=(0.75, 1.33), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(224,scale=(0.5, 1.0), ratio=(3./4., 4./3.)),
            # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.4),  # 10%的概率将图像转换为灰度

            # transforms.RandomAffine(degrees, translate=None, scale=None, shear=None),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        'val': transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        "null": transforms.Compose([
            transforms.Resize(resize_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
    return data_transforms[chance]

