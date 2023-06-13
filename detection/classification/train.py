import os
import argparse    
import json

from torch.utils import data
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix, average_precision_score
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from  utils import *

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
    
        # define network, loss, dataset
        self.network = get_model(config['model'])
        self.loss = get_loss(config['loss'])
        self.train_data, self.val_data = split_data()
        self.transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['img_size'],config['img_size'])),
        # transforms.ColorJitter(), 
        transforms.RandomAdjustSharpness(2, p=0.5),
        # transforms.RandomAutocontrast(p=0.5), 
        transforms.RandomRotation(180/24),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['img_size'],config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.train_dataset = Dataset(self.train_data, transform=self.transform)
        self.val_dataset = Dataset(self.val_data, transform=self.test_transform)
        
        # initialize some variables
        self.best_val_loss = None
        self.best_auprc = 0
        self.best_epoch = None
        
    def train_dataloader(self):
        train_loader = data.DataLoader(dataset=self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=16, pin_memory=True)
        return train_loader
    
    def val_dataloader(self):
        val_loader = data.DataLoader(dataset=self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
        return val_loader
    
    def configure_optimizers(self):
        optimizer = get_optimizer(self.config['optimizer'], self.parameters(), self.config['lr'], 0, 5e-4)
        self.lr_scheduler = get_scheduler(self.config['scheduler'], optimizer, self.config['epochs'])
        return optimizer
    
    def forward(self, x):
        return self.network(x)    
    
    def training_step(self, batch, batch_idx):
        prob_all = []
        label_all = []
        pred = []
        
        images, labels = batch
        outputs = self.network(images)
        loss = self.loss(outputs, labels)
        prob = F.softmax(outputs,dim=1)
        prob_all.extend(prob[:,1].cpu().detach().numpy())
        label_all.extend(labels.cpu().detach().numpy())
        pred.extend(outputs.argmax(dim=-1).cpu().detach().numpy())
        acc = (outputs.argmax(dim=-1) == labels).float().mean()

        return {'loss': loss,'acc': acc, 'prob': prob_all, 'label': label_all, 'pred': pred}
        
    def training_epoch_end(self, training_step_outputs):
        s = 0
        acc = 0
        prob_all = []
        label_all = []
        pred = []
        for t in training_step_outputs:
            s += t['loss']
            acc += t['acc']
            prob_all += t['prob']
            label_all += t['label']
            pred += t['pred']
           
        s /= len(training_step_outputs)
        acc /= len(training_step_outputs)
        
        evaluation = Evaluation(label_all,prob_all)
        threshold = 0.5 
        performance = evaluation.eval(threshold)

        self.log('step',self.trainer.current_epoch)
        if self.lr_scheduler:
            self.lr_scheduler.step()
            self.log('lr', self.lr_scheduler.get_last_lr()[0], on_step=False, on_epoch=True)
        else:
            self.log('lr', self.config['lr'], on_step=False, on_epoch=True)
        
        self.log('train/loss', s, on_step=False, on_epoch=True)
        self.log('train/AUROC', performance["AUROC"], on_step=False, on_epoch=True)
        self.log('train/AUPRC', performance["AUPRC"], on_step=False, on_epoch=True)
        self.log('train/TP', performance["TP"], on_step=False, on_epoch=True)
        self.log('train/FP', performance["FP"], on_step=False, on_epoch=True)
        self.log('train/FN', performance["FN"], on_step=False, on_epoch=True)
        self.log('train/TN', performance["TN"], on_step=False, on_epoch=True)
        self.log('train/Sensitivity', performance["Sensitivity"], on_step=False, on_epoch=True)
        self.log('train/Specificity', performance["Specificity"], on_step=False, on_epoch=True)
        self.log('train/PPV', performance["PPV"], on_step=False, on_epoch=True)
        self.log('train/NPV', performance["NPV"], on_step=False, on_epoch=True)
        self.log('train/F1', performance["F1"], on_step=False, on_epoch=True)
        self.log('train/Accuracy', acc, on_step=False, on_epoch=True)
     
        
    
    def validation_step(self, batch, batch_idx):
        prob_all = []
        label_all = []
        pred = []

        images, labels = batch
        outputs = self.network(images)
        loss = self.loss(outputs, labels)
        prob = F.softmax(outputs,dim=1)
        prob_all.extend(prob[:,1].cpu().detach().numpy())
        label_all.extend(labels.cpu().detach().numpy())
        pred.extend(outputs.argmax(dim=-1).cpu().detach().numpy())
        acc = (outputs.argmax(dim=-1) == labels).float().mean()

        return {'loss': loss,'acc': acc, 'prob': prob_all, 'label': label_all, 'pred': pred}
     
    def validation_epoch_end(self, outputs):
        s = 0
        acc = 0
        prob_all = []
        label_all = []
        pred = []
        for t in outputs:
            s += t['loss']
            acc += t['acc']
            prob_all += t['prob']
            label_all += t['label']
            pred += t['pred']
        s /= len(outputs)
        acc /= len(outputs)
        
        evaluation = Evaluation(label_all,prob_all)
        threshold = 0.5 
        performance = evaluation.eval(threshold)

        self.log('step',self.trainer.current_epoch)
        self.log('val/loss', s, on_step=False, on_epoch=True)
        self.log('val/AUROC', performance["AUROC"], on_step=False, on_epoch=True)
        self.log('val/AUPRC', performance["AUPRC"], on_step=False, on_epoch=True)
        self.log('val/TP', performance["TP"], on_step=False, on_epoch=True)
        self.log('val/FP', performance["FP"], on_step=False, on_epoch=True)
        self.log('val/FN', performance["FN"], on_step=False, on_epoch=True)
        self.log('val/TN', performance["TN"], on_step=False, on_epoch=True)
        self.log('val/Sensitivity', performance["Sensitivity"], on_step=False, on_epoch=True)
        self.log('val/Specificity', performance["Specificity"], on_step=False, on_epoch=True)
        self.log('val/PPV', performance["PPV"], on_step=False, on_epoch=True)
        self.log('val/NPV', performance["NPV"], on_step=False, on_epoch=True)
        self.log('val/F1', performance["F1"], on_step=False, on_epoch=True)
        self.log('val/Accuracy', acc, on_step=False, on_epoch=True)

        if performance["AUPRC"] > self.best_auprc:
            self.best_epoch = self.current_epoch
            self.best_auprc = performance["AUPRC"]
        
        self.log('hp_metric', self.best_auprc, on_step=False, on_epoch=True)
        self.log('best_epoch', self.best_epoch, on_step=False, on_epoch=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='training config')
    parser.add_argument('--savedir', help='log dir')
    parser.add_argument('--gpu', help='gpu id')
    opt = parser.parse_args()
    
    with open(opt.config, 'r') as file:
        config = json.load(file)
    print(config)
    

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val/AUPRC',
        filename='pneumothorax-epoch{epoch:02d}-val_AUPRC{val/AUPRC:.2f}',
        mode='max',
        auto_insert_metric_name=False
    )
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{opt.savedir}/")
    seed_everything(73, workers=True)
    
    trainer = pl.Trainer(
        gpus=len(str(opt.gpu).split(',')),
        max_epochs=config['epochs'],
        deterministic=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=tb_logger
    )
    # set_random_seed()
    trainer.fit(LitModel(config))