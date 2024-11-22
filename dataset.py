import os
import glob
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class OrientationDataset(Dataset):
    def __init__(self, assets_dir, split_csv=None, transform=None, verbose=False):
        self.assets_dir = assets_dir
        self.verbose = verbose
        self.classes = {'back': 0, 'front': 1, 'left': 2, 'right': 3}
        self.transform = transform
        self.split_df = None if split_csv is None else pd.read_csv(split_csv)
        self.split = os.path.basename(split_csv).split('.')[0] if split_csv is not None else None
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filepath, position = self.data[idx]
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.classes[position]

        # One-hot encode the label
        label = np.zeros(len(self.classes))
        label[self.classes[position]] = 1

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label
    
    def load_data(self):
        # Load data from files in self.assets_dir
        avail_assets = os.listdir(self.assets_dir)
        data = list()

        if self.split_df is not None:
                asset_splits = self.split_df[self.split_df['split'] == self.split]['asset_id'].values
                avail_assets = [asset for asset in avail_assets if asset in asset_splits]
    
        for asset in avail_assets:
            asset_path = os.path.join(self.assets_dir, asset)
            asset_positions = self.load_asset_positions(asset_path)
            if asset_positions:
                for position, filepath in asset_positions.items():
                    data.append((filepath, position))
    
        return data

    def load_asset_positions(self, asset_path):
        positions = ['front', 'back', 'left', 'right']
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

        asset_positions = dict()

        for position in positions:
            for ext in extensions:
                # Build a search pattern using glob
                search_pattern = os.path.join(asset_path, f'*{position}*{ext}')
                # print(f"Searching for {position} in {asset_path} using pattern: {search_pattern}")
                matched_files = glob.glob(search_pattern)
                
                if len(matched_files) == 1:
                    asset_positions[position] = matched_files[0]
                    for position, file_path in asset_positions.items():
                        asset_positions[position] = file_path
                elif len(matched_files) > 1:
                    if self.verbose:
                        print(f"Warning: Multiple files found for position '{position}' in {asset_path}")
                else:
                    if self.verbose:
                        print(f"Warning: No files found for position '{position}' in {asset_path}")
        
        return asset_positions
    
    def _build_csv(self, output_dir, train_ratio=0.7, val_ratio=0.15):
        os.makedirs(output_dir, exist_ok=True)
        avail_assets = os.listdir(self.assets_dir)
        train_val, test = train_test_split(avail_assets, test_size=(1 - train_ratio))
        train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio))

        # Create a DataFrame to store the asset ids and their splits
        split_data = []

        for asset in train:
            split_data.append({'asset_id': asset, 'split': 'train'})
        for asset in val:
            split_data.append({'asset_id': asset, 'split': 'val'})
        for asset in test:
            split_data.append({'asset_id': asset, 'split': 'test'})

        split_df = pd.DataFrame(split_data)

        # Save CSV files for train, validation, and test splits
        split_df[split_df['split'] == 'train'].to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        split_df[split_df['split'] == 'val'].to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        split_df[split_df['split'] == 'test'].to_csv(os.path.join(output_dir, 'test.csv'), index=False)

        print(f"CSV files created in {output_dir}: train.csv, val.csv, test.csv")