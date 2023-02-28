import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils import data
import torchvision.transforms as transforms
import torchvision
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from utils import *
import json

def load_model_lit2torch(model_name, ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name)
    model_dict = torch.load(ckpt)['state_dict']
    for key in model_dict.keys():        
        # model.state_dict()[key.replace('network.', '')] = model_dict[key]
        model.state_dict()[key.replace('network.', '')].copy_(model_dict[key])
        # print(model_dict[key])
        # print(model.state_dict()[key.replace('network.', '')])
        # raise
    model.to(device)
    model.eval()
    return model

class myEnsemble(nn.Module):
    def __init__(self, *models):
        super(myEnsemble, self).__init__()
        # self.models = models
        self.models = []
        for model in models:
            self.models.append(model)
       
    def forward(self, x):
        out = []
        for model in self.models:
            out.append(model(x))
        
        out = sum(out) / len(out)
        
        return out

def main(opt):

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.img_size,opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    # Dataset
    test_dataset = NTUH_20_Dataset(transform=test_transform)
    # Dataloader
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = load_model_lit2torch('efficientnet_b2', './logs/efficientnet_b2/checkpoints/pneumothorax-epoch83-val_AP0.74.ckpt')
    model2 = load_model_lit2torch('inceptionv3', './logs/inceptionv3/checkpoints/pneumothorax-epoch75-val_AP0.70.ckpt')
    model3 = load_model_lit2torch('densenet121', './logs/densenet121/checkpoints/pneumothorax-epoch84-val_AP0.71.ckpt')
    model = myEnsemble(model1, model2, model3).to(device)  

    criterion = nn.CrossEntropyLoss()
    # These are used to record information in validation.
    test_loss = []
    test_accs = []
    f = nn.Softmax(dim=1)
    prob_all = []
    label_all = []
    pred = []
    for images, labels  in tqdm.tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            prob = f(outputs)
        prob_all.extend(prob[:,1].cpu().numpy())
        label_all.extend(labels.cpu().numpy())
        pred.extend(prob.argmax(dim=-1).cpu().numpy())
        # We can still compute the loss (but not the gradient).-
        loss = criterion(outputs, labels)
        # Compute the accuracy for current batch.
        acc = (outputs.argmax(dim=-1) == labels).float().mean()
        # Record the loss and accuracy.
        test_loss.append(loss.item())
        test_accs.append(acc)
        
    # The average loss and accuracy for entire validation set is the average of the recorded values.
    test_loss = sum(test_loss) / len(test_loss)
    print(f"Test loss: {test_loss}")
    evaluation = Evaluation(label_all,prob_all)
    # # threshold = 0.23281845450401306
    # # threshold = 0.029015876352787018
    threshold = 0.5
    print(evaluation.eval(threshold))
    evaluation.plot_confusion_matrix(threshold=threshold)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=1024, help='image sizes')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for per GPUs')
    opt = parser.parse_args()
    
    main(opt)
