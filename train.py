"""
@author: Cuong Tran <cgtran2003@gmail.com>
"""
from dataset import NaturalSceneDataset
from model import MyCNN

import torch
from torchvision.transforms import Compose, ToTensor, Resize, RandomAffine, ColorJitter
from torchsummary import summary
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import argparse, shutil
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description='Train a simple CNN model for natural scene classification')
    parser.add_argument('--data-path', type=str, default='./natural_scenes')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--board-path', type=str, default='TensorBoard')
    parser.add_argument('--checkpoint-path', type=str, default='Save_Model')
    parser.add_argument('--pretrained-path', type=str, default='Save_Model/last.pt')
    return parser.parse_args()

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
        RandomAffine(
            degrees=(-5, 5),
            translate=(0.15, 0.15),
            scale=(0.85, 1.1),
            shear=10
        ),
        ColorJitter(
            brightness=0.25,
            contrast=0.5,
            saturation=0.5,
            hue=0.05
        ),
    ])
    valid_transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size))
    ])

    # DATASET
    train_dataset = NaturalSceneDataset(root=args.data_path, train=True, transform=train_transform)
    valid_dataset = NaturalSceneDataset(root=args.data_path, train=False, transform=valid_transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=False,
        shuffle=False,
    )

    # MODEL
    model = MyCNN(num_classes=6).to(device)
    # summary(model, (3,args.image_size, args.image_size))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = args.epochs
    
    # TensorBoard 
    tb_path = args.board_path
    if os.path.isdir(tb_path):
        shutil.rmtree(tb_path)
    os.makedirs(tb_path)
    writer = SummaryWriter(tb_path)

    # PRETRAINED MODEL 
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if os.path.exists(args.pretrained_path):
        checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
    else:
        start_epoch = 0
        best_accuracy = 0
        
    for epoch in range(start_epoch, epochs):
        # MODEL TRAINING
        model.train()
        progress_bar = tqdm(train_dataloader, colour = 'green')
        for iter, (image, label) in enumerate(progress_bar):
            # Forward
            image = image.to(device)
            label = label.to(device)
            prediction = model(image)
            loss = criterion(prediction, label)
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Epoch: {epoch}/{epochs}, Loss: {loss.item():.4f}')
            writer.add_scalar('Train/Loss:', loss.item(), epoch * len(train_dataloader) + iter)    
        
        # MODEL VALIDATION
        model.eval()
        all_predictions = []
        all_labels = []
        all_loss = []
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour = 'cyan', desc=f'Validation Epoch: {epoch}/{epochs}')
            for image, label in progress_bar:
                image = image.to(device)
                label = label.to(device)
                prediction = model(image) 
                loss = criterion(prediction, label)
                
                prediction = torch.argmax(prediction, dim=1) 
                all_predictions.extend(prediction.tolist())
                all_labels.extend(label.tolist())
                all_loss.append(loss.item())
        
        avg_loss = np.mean(all_loss)
        accuracy = accuracy_score(all_labels, all_predictions)
        writer.add_scalar('Validation/Accuracy:', accuracy, epoch)
        writer.add_scalar('Validation/Loss:', avg_loss, epoch)
        print(f'Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}')

        # Write progress
        with open('train_progress.txt', 'a') as f:
            f.write(f'Epoch: {epoch}/{epochs}, Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}\n')

        # CHECKPOINT 
        checkpoint = {
            'epoch': epoch + 1,
            'best_accuracy': best_accuracy,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, args.checkpoint_path + '/last.pt')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(checkpoint, args.checkpoint_path + '/best.pt')

if __name__ == '__main__':
    args = get_args()
    train(args)