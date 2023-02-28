import os
import random
import torch
import numpy as np
from torch.utils import data
import cv2
import torch.nn as nn
import torchvision
import pandas as pd
import monai
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve,classification_report, confusion_matrix, plot_confusion_matrix, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt

def set_random_seed(seed = 73):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_model(model_name, weight=None):
    if model_name == 'inceptionv3':
        model = torchvision.models.inception_v3(pretrained=True,aux_logits=False)
        model.dropout = nn.Dropout(0.4)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    elif model_name == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.features.norm5 = nn.AdaptiveAvgPool2d(1)
        model.classifier = nn.Sequential( nn.Dropout(p=0.4), nn.Linear(model.classifier.in_features, 2))
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    elif model_name == 'efficientnet_b2':
        model = torchvision.models.efficientnet_b2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    elif model_name == 'regnet':
        model = torchvision.models.regnet_y_1_6gf(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    elif model_name == 'resnext':
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        if weight != None:
            model.load_state_dict(torch.load(weight))
        return model
    else:
        pass

class Ensemble(nn.Module):
    def __init__(self, *models):
        super(Ensemble, self).__init__()
        self.models = []
        for model in models:
            self.models.append(model)
        self.linear = nn.Linear(2*len(models),2)
        
       
    def forward(self, x):
        out = []
        for model in self.models:
            out.append(model(x))
            
        out = self.linear(torch.cat(out,1))
        
        return out

def get_loss(loss_name):
    if loss_name == 'CE':
        criterion = nn.CrossEntropyLoss()
        return criterion
    elif loss_name == 'BCE':
        criterion = nn.BCELoss()
        return criterion
    elif loss_name == 'FL':
        criterion = monai.losses.FocalLoss(to_onehot_y=True)
        return criterion
    else:
        pass

def get_optimizer(optimizer_name, param,lr, momentum,weight_decay):
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(param, lr=lr, weight_decay=weight_decay)
        return optimizer
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(param, lr=lr, weight_decay=weight_decay)
        return optimizer
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(param, lr=lr, momentum=momentum, weight_decay=weight_decay)
        return optimizer
    else:
        return None

def get_scheduler(scheduler_name, optimizer, max_epoch):
    if scheduler_name == 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        return scheduler
    if scheduler_name == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 5e-6, 5e-2, step_size_up=50, step_size_down=None)
        return scheduler
    if scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
        return scheduler
    if scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(max_epoch/2))
        return scheduler


def split_data():
    csv = pd.read_csv('../../data/classification/NTUH_1519.csv')
    img_dir = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/'
    label_data = []
    for i in range(csv.shape[0]):
        if csv.iloc[i,3] != 1:
            label_data.append([csv.iloc[i,0], csv.iloc[i,2]])
   
    print(f"Training images: #{len(label_data)}")
    # split data
    set_random_seed() 
    train, val = train_test_split(label_data, test_size=0.2)
    # val, test = train_test_split(val, test_size=0.5)
    
    return train, val


class Dataset(data.Dataset):
    def __init__(self, data, transform=None):
        self.img_dir = '/home/u/woody8657/data/C426_Pneumothorax_preprocessed/image/preprocessed/images/'
        self.data = data
        self.transform = transform
        
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.img_dir,self.data[index][0])+'.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 


        label = self.data[index][1]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.data)

class NTUH_20_Dataset(data.Dataset):
    def __init__(self, transform=None):
        self.label_data = []
        label_path = '../../data/classification/labels_NTUH_20'
        for pid in os.listdir(label_path):
            if os.path.isfile(os.path.join(label_path, pid, 'label.json')):
                self.label_data.append([pid, 1])
            else:
                self.label_data.append([pid, 0])
        self.img_dir = '../../data/images_NTUH_20'
        self.transform = transform
        
    
    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.img_dir,self.label_data[index][0])+'.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB ) 

        label = self.label_data[index][1]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.label_data)
    
class Evaluation:
    def __init__(self, label, pred):
        '''
        label: binary
        pred: probability
        '''
        self.label = label
        self.pred = pred
    def eval(self, threshold):
        AUROC = roc_auc_score(self.label, self.pred)
        AUPRC = average_precision_score(self.label, self.pred)

        pred_binary = self.thresholding(self.pred, threshold)
        tn, fp, fn, tp = confusion_matrix(self.label, pred_binary).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        NPV = tn / (tn + fn)
        PPV = tp / (tp + fp)
        F1 = 2 * sensitivity * PPV / (sensitivity + PPV)
        acc = (tp + tn) / (tp + tn + fp + fn)
        output = {
            "AUROC": AUROC,
            "AUPRC": AUPRC,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "PPV": NPV,
            "NPV": NPV,
            "F1": F1,
            "Accuracy": acc
        }
        return output

    def thresholding(self, pred, threshold):
        output = [0] * len(pred)
        for i in range(len(pred)):
            if pred[i] > threshold:
                output[i] = 1
            else:
                output[i] = 0
        return output

    def plot_confusion_matrix(self, threshold=0.5, filename="confusion_matrix.png"):
        pred_binary = self.thresholding(self.pred, threshold)
        tn, fp, fn, tp = confusion_matrix(self.label, pred_binary).ravel()
        ax = sns.heatmap([[tn, fp], [fn, tp] ], annot=True, cmap='Blues', cbar=False, fmt="d")

        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(['None','Pneumothorax'])
        ax.yaxis.set_ticklabels(['None','Pneumothorax'])

        plt.savefig(filename)

# Youden's index
def sensivity_specifity_cutoff(y_true, y_score):
    '''Find data-driven cut-off for classification
    
    Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
    
    Parameters
    ----------
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    References
    ----------
    
    Ewald, B. (2006). Post hoc choice of cut points introduced bias to diagnostic research.
    Journal of clinical epidemiology, 59(8), 798-801.
    
    Steyerberg, E.W., Van Calster, B., & Pencina, M.J. (2011). Performance measures for
    prediction models and markers: evaluation of predictions and classifications.
    Revista Espanola de Cardiologia (English Edition), 64(9), 788-794.
    
    Jiménez-Valverde, A., & Lobo, J.M. (2007). Threshold criteria for conversion of probability
    of species presence to either–or presence–absence. Acta oecologica, 31(3), 361-369.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

def model_complexity(model):
    from torchstat import stat
    input_size = (3, 1024, 1024)
    stat(model, input_size)