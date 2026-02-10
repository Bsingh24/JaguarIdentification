from Resnet import ResNet, Block, create_pairs, create_tensors
import random
import pandas as pd
import numpy as np
import os
import torch
from torch.nn import TripletMarginLoss
from torchvision.io import decode_image, ImageReadMode
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision.transforms import v2
from torch.optim import Adam
from collections import defaultdict
import mlflow
import argparse

"""
creates batches with a size of 15, ensuring we are using all samples provided at random.
"""

def create_batches(label_mapping, n_classes, n_samples):
    all_classes = list(label_mapping.keys())
    batches = []
    label_mapping_copy = label_mapping.copy()
    while True:
        available_classes = [c for c in all_classes if len(label_mapping_copy[c]) > 0] # Keep only classes if they are still available.
        # Exit if we have less
        if len(available_classes) < n_classes: 
            break

        # Ensures that we will at least have 1 example of every label pair
        selected_classes = random.sample(available_classes, n_classes)
        batch = []

        for cls in selected_classes:
            indices = label_mapping_copy[cls]
            # If there are not enough examples to sample from, we re-select a sample from the shortened list. Empty the list after
            if len(indices) >= n_samples:
                chosen = indices[:n_samples]
                label_mapping_copy[cls] = indices[n_samples:]
            else:
                chosen = indices + random.choices(indices, k=(n_samples - len(indices)))
                label_mapping_copy[cls] = []

            batch.extend(chosen)

        batches.append(batch)

    return batches


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.n_classes = n_classes 
        self.n_samples = n_samples  

        self.label_mapping = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_mapping[int(label)].append(idx)

    def __iter__(self):
        for cls in self.label_mapping:
            random.shuffle(self.label_mapping[cls])

        batches = create_batches(self.label_mapping, self.n_classes, self.n_samples)
        for batch in batches:
            yield batch

    def __len__(self):
        batches = create_batches(self.label_mapping, self.n_classes, self.n_samples)
        return len(batches)


class ImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_label = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_label.iloc[idx, 0])
        # image = decode_image(img_path, mode=ImageReadMode.RGB)
        image = decode_image(img_path)
        label = self.img_label.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)

        return image, label

        
def load_training_params(lr=1e-4):
    resnet_model = ResNet(Block, [3, 4, 6, 3], image_channels=4)
    loss_fn = TripletMarginLoss()
    optimizer = Adam(resnet_model.parameters(), lr=lr)
    return resnet_model, loss_fn, optimizer


def preprocess():
    transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True)])
    
    train_data = ImageDataset('Data/train_data.csv', 'Data/train/', transform)
    valid_data = ImageDataset('Data/valid_data.csv', 'Data/train/', transform)
    test_data = ImageDataset('Data/test_data.csv', 'Data/train/', transform)

    df = pd.read_csv('Data/train_data.csv')
    label_tensor = torch.tensor(df['encoded_ground_truth'].values)
    sampler = BalancedBatchSampler(label_tensor, 5, 4)
    dataloader = DataLoader(train_data, batch_sampler=sampler)

    return train_data, valid_data, test_data, dataloader


def train_mlflow(model, loss_fn, optimizer, dataloader, valid_data):
    mlflow.set_experiment("Resnet Scratch Test")
    mlflow.pytorch.autolog()
    mlflow.enable_system_metrics_logging()
    
    with mlflow.start_run():
        params = {'learning_rate': 1e-4, 'epochs': 50, 'output_size': 256, 'batch_size': 15, 'channel_dim': 4}
        mlflow.log_params(params)
        
        for i in range(params['epochs']):
            avg_loss = []
            for batch, (X, y) in enumerate(dataloader):
                pred = model(X)
                pairs = create_pairs(y, pred)
                anchor_tensor, positive_tensor, negative_tensor = create_tensors(pairs)
                loss = loss_fn(anchor_tensor, positive_tensor, negative_tensor)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                avg_loss.append(loss.item())
            with torch.no_grad():
                val_output = []
                val_label = []
                for j in range(valid_data.__len__()):
                    X, y = valid_data[j]
                    reshaped_X = torch.reshape(X, (1, 4, 224, 224))
                    val_pred = model(reshaped_X)
                    val_output.append(torch.flatten(val_pred))
                    val_label.append(y)
                pairs = create_pairs(val_label, val_output)
                anchor_tensor, positive_tensor, negative_tensor = create_tensors(pairs)
                val_loss = loss_fn(anchor_tensor, positive_tensor, negative_tensor)
            print(f"Epoch {i+1}: Avg training loss - {np.mean(avg_loss)}, Avg validation loss - {val_loss}")
            mlflow.log_metric("train_loss", np.mean(avg_loss), step=i+1)
            mlflow.log_metric("valid_loss", val_loss, step=i+1)


def train_non_mlflow(model, loss_fn, optimizer, dataloader, valid_data):            
    for i in range(params['epochs']):
        avg_loss = []
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            pairs = create_pairs(y, pred)
            anchor_tensor, positive_tensor, negative_tensor = create_tensors(pairs)
            loss = loss_fn(anchor_tensor, positive_tensor, negative_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            avg_loss.append(loss.item())
        with torch.no_grad():
            val_output = []
            val_label = []
            for j in range(valid_data.__len__()):
                X, y = valid_data[j]
                reshaped_X = torch.reshape(X, (1, 4, 224, 224))
                val_pred = model(reshaped_X)
                val_output.append(torch.flatten(val_pred))
                val_label.append(y)
            pairs = create_pairs(val_label, val_output)
            anchor_tensor, positive_tensor, negative_tensor = create_tensors(pairs)
            val_loss = loss_fn(anchor_tensor, positive_tensor, negative_tensor)
        print(f"Epoch {i+1}: Avg training loss - {np.mean(avg_loss)}, Avg validation loss - {val_loss}")    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Choose between Resnet-50 pretrained, Resnet-50 non-pretrained, and other**TODO')
    parser.add_argument('--epochs', help='Choose how many epochs you want the selected model to run for', type=int, default=50)    
    parser.add_argument('--mlflow', help="Track metrics via MLFlow. It will be stored locally.", action='store_true')
    args = parser.parse_args()

    model = args.model
    epochs = args.epochs
    use_mlflow = args.mlflow
    model, loss_fn, optimizer = load_training_params()
    train_data, valid_data, test_data, dataloader = preprocess()
    
    if use_mlflow:
        print("Beginning training with MLFlow tracking")
        print("####################################")
        train_mlflow(model, loss_fn, optimizer, dataloader, valid_data)
        print("####################################")
        print("Finished Training")
        
    else:
        print("Beginning training")
        print("####################################")
        train_non_mlflow(model, loss_fn, optimizer, dataloader, valid_data)
        print("####################################")
        print("Finished Training")


if __name__ == "__main__":
    main()