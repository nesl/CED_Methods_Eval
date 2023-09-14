import os
import glob
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
import torch
from abc import ABC
import configparser


package_dir, _ = os.path.split(os.path.abspath(__file__))

config = configparser.ConfigParser()
for fName in ['config_local.ini', 'config.ini']:
    f_path = join(package_dir, fName)
    if os.path.isfile(f_path):
        config.read(f_path)
        continue


class WISDM(Dataset, ABC):
    """Abstract class for the WISDM datasets accessible with pytorch"""

    def __init__(self,
                 accel_files,
                 gyro_files,
                 root,
                 time_window=5,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 normalize_acc=False):
        """
        Args:
            root (string): Root directory of dataset where directory ``wisdm`` exists
            accel_files (string): name of the accelartion file
            gyro_files (string): name of the gyroscope file
            folds (array, int): folds to call for.
            time_window (int): The time window of input (unit: second).
            normalize_acc (bool, optional): if true then normalize accleration data by 9.8 (gravity offset)
            overwrite (bool): overwrite existing npz file
        """

        # Dataset generation
        self.sample_rate = 20 # 20Hz
        self.input_size = int(self.sample_rate * time_window)
        self.root = root
        
        self.accel_files = accel_files
        self.gyro_files = gyro_files
        self.accel_file_paths = glob.glob(join(root, accel_files))
        self.gyro_file_paths = glob.glob(join(root, gyro_files))
        self.accel_file_paths.sort()
        self.gyro_file_paths.sort()

        # Process root and related values
        self.db_path = join(root, 'raw/watch/{}_{}.npz'.format(
            type(self).__name__, self.input_size))

        # Read the csv and merge df with folds
        if 'df' not in self.__dir__():
            dfs = []
            for index, (accel_file, gyro_file) in enumerate(zip(self.accel_file_paths, self.gyro_file_paths)):
                accel_df = pd.read_csv(accel_file, names=['userID', 'label', 'timestamp', 'accel_x', 'accel_y', 'accel_z'], header=0)
                gyro_df = pd.read_csv(gyro_file, names=['timestamp', 'gyro_x', 'gyro_y', 'gyro_z'], header=0)
                df = pd.merge(accel_df, gyro_df,on="timestamp")
                df['accel_z'] = df['accel_z'].str.replace(';','').astype(float)
                df['gyro_z'] = df['gyro_z'].str.replace(';','').astype(float)
                if index < 1/5 * len(self.accel_file_paths): df['fold'] = 1 # data from every 1/5 users are marked as a folder
                elif index < 2/5 * len(self.accel_file_paths): df['fold'] = 2
                elif index < 3/5 * len(self.accel_file_paths): df['fold'] = 3
                elif index < 4/5 * len(self.accel_file_paths): df['fold'] = 4
                else: df['fold'] = 5
                dfs.append(df)
            self.df = pd.concat(dfs, axis=0, ignore_index=True)
        self.classes = sorted(list(set(self.df['label'])))
        self.nClasses = len(self.classes)

        # Maybe create dataset
        if not os.path.isfile(self.db_path) or overwrite:
            self._create_dataset()

        # Get item processing
        self.folds = folds
        self.normalize_acc = normalize_acc
        self.get_db_folds()  # Fils self.accels and self.labels

    def __len__(self):
        return len(self.imus)

    def get_db_folds(self):
        full_db = np.load(self.db_path, allow_pickle=True)
        self.imus = []
        self.labels = []
        self.folds_nb = []
        for fold in self.folds:
            fold_name = 'fold{}'.format(fold)
            print('loading ', fold_name)
            imus = full_db[fold_name].item()['imus']
            labels = full_db[fold_name].item()['labels']
            self.imus.extend(imus)
            self.labels.extend(labels)
            self.folds_nb.extend([fold, ] * len(labels))

        self.imus = [self.preprocess(torch.tensor(i, dtype=torch.float)) for i in self.imus] # PyTorch expects the input tensor and model parameters to have the same dtype, since the model parameters are initialized as FloatTensors, we need to change the input to torch.float.
        self.labels = [torch.tensor(l).int() for l in self.labels]

    def preprocess(self, imu):
        # normalize accel 
        # imu[:3] *= 9.80665 # TODO: change this to np.max(), by changeing self.accels and self.labels into tensors (instead of list of tensors)

        if self.normalize_acc:
            imu[:3] = imu[:3] / 9.80665 # gravity offset
        return imu

    def get_label_mapping(self):
        classes = self.classes
        mapping = {}
        for c in classes:
            id = classes.index(c)
            mapping[c] = id
        return mapping
    
    def __getitem__(self, idx):
        imu = self.imus[idx]
        label = self.labels[idx]
        return imu, label

    def _create_dataset(self):
        df = self.df
        input_size = self.input_size
        
        # Create npz file
        print('Creating corresponding npz file...')
        dataset = {}
        for fold in range(1, 6):
            fold_name = 'fold{}'.format(fold)
            dataset[fold_name] = {}
            dataset[fold_name]['imus'] = []
            dataset[fold_name]['labels'] = []

        pairs = set(zip(df.userID, df.label))
        for pair in pairs: 
            # each activity is collected for about 3 min = 3x60x20 = 3600 frames
            df_temp = df[(df['userID'] == pair[0]) & (df['label'] == pair[1])]
            num_sequence = len(df_temp) // input_size
            for i in range(num_sequence):
                rows = df_temp.iloc[i * input_size : (i + 1) * input_size]
                imu = rows[['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']].values.T
                label = self.classes.index(pair[1])
                fold = df_temp.iloc[i * input_size].fold
                fold_name = 'fold{}'.format(fold)
                dataset[fold_name]['imus'].append(imu)
                dataset[fold_name]['labels'].append(label)
        print('Saving')
        np.savez(self.db_path, **dataset)



class WISDMSelect(WISDM):
    """Class for the partial WISDM datasets with selected"""

    def __init__(self,
                 accel_files='raw/watch/accel/*.txt',
                 gyro_files='raw/watch/gyro/*.txt',
                 root=config['Paths']['WISDM'],
                 class_file='activity_key.txt',
                 class_selected=['walking', 'jogging', 'sitting', 'standing', 'typing', 'teeth', 'pasta', 'drinking'],
                 time_window=5,
                 overwrite=False,
                 folds=[1, 2, 3, 4],
                 normalize_acc = False):
        """
        Args:
            root (string): Root directory of dataset where directory ``wisdm`` exists
            accel_files (string): name of the accelartion file
            gyro_files (string): name of the gyroscope file
            class_file (string): name of the activity label file
            class_selected (list, string): activities selected for the customized wisdm dataset
            folds (array, int): folds to call for.
            time_window (int): The time window of input (unit: second).
            normalize_acc (bool, optional): if true then normalize accleration data by 9.8 (gravity offset).
            overwrite (bool): overwrite existing npz file
        """

        # Filter df with selected activities
        df_key = pd.read_csv(join(root, class_file), sep=' = ',names=['activity', 'label'], engine='python')
        self.label_dict = dict(df_key.values)
        self.class_selected = class_selected
        label_selected = [self.label_dict[c] for c in class_selected]

        accel_file_paths = glob.glob(join(root, accel_files))
        gyro_file_paths = glob.glob(join(root, gyro_files))
        accel_file_paths.sort()
        gyro_file_paths.sort()

        dfs = []
        for index, (accel_file, gyro_file) in enumerate(zip(accel_file_paths, gyro_file_paths)):
            accel_df = pd.read_csv(accel_file, names=['userID', 'label', 'timestamp', 'accel_x', 'accel_y', 'accel_z'], header=0)
            gyro_df = pd.read_csv(gyro_file, names=['timestamp', 'gyro_x', 'gyro_y', 'gyro_z'], header=0)
            df = pd.merge(accel_df, gyro_df,on="timestamp")
            df = df[df['label'].isin(label_selected)]
            df['accel_z'] = df['accel_z'].str.replace(';','').astype(float)
            df['gyro_z'] = df['gyro_z'].str.replace(';','').astype(float)
            if index < 1/5 * len(accel_file_paths): df['fold'] = 1 # data from every 1/5 users are marked as a folder
            elif index < 2/5 * len(accel_file_paths): df['fold'] = 2
            elif index < 3/5 * len(accel_file_paths): df['fold'] = 3
            elif index < 4/5 * len(accel_file_paths): df['fold'] = 4
            else: df['fold'] = 5
            dfs.append(df)
        self.df  = pd.concat(dfs, axis=0, ignore_index=True)

        super().__init__(
            accel_files=accel_files,
            gyro_files=gyro_files,
            root=root,
            time_window=time_window,
            overwrite=overwrite,
            folds=folds,
            normalize_acc=normalize_acc)

    def get_label_mapping(self):
        label_dict_swap = {letter: name for name, letter in self.label_dict.items()}
        classes = self.classes
        mapping = {}
        for c in classes:
            id = classes.index(c)
            mapping[label_dict_swap[c]] = id
        return mapping


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torch import nn
    from torch import optim

    time_window = 1

    imu_train_set = WISDM(
        accel_files='raw/watch/accel/*.txt',
        gyro_files='raw/watch/gyro/*.txt',
        root=config['Paths']['WISDM'],
        time_window=time_window,
        overwrite=False,
        folds=[1, 2, 3, 4],
    )

    imu_train_loader = DataLoader(imu_train_set, batch_size=16, 
                              shuffle=True, num_workers=4)

    # Load a sample network
    net = nn.Sequential(
        nn.Conv1d(6, 32, 3, 1), nn.ReLU(), nn.BatchNorm1d(32),
        nn.Conv1d(32, 32, 3, 1), nn.ReLU(), nn.BatchNorm1d(32),
        nn.Conv1d(32, imu_train_set.nClasses, 5, 1), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    imus = imu_train_loader.dataset.imus
    labels = imu_train_loader.dataset.labels

    # test = np.concatenate([s.unsqueeze(0) for s in imus])
    # print(test.shape)
    # print(max(np.max(test), -np.min(test)))
    
    # Training loop
    # n_epochs = 5
    # summary = {'loss': [[] for _ in range(n_epochs)], 'acc': [[] for _ in range(n_epochs)]}
    # for e in range(n_epochs):
    #     for i, (imus, labels) in enumerate(imu_train_loader):
    #         # Zero the grads
    #         optimizer.zero_grad()

    #         # Run the Net
    #         x = net(imus)
    #         x = x.view(x.shape[:-1])

    #         # Optimize nettpr
    #         loss = criterion(x, labels.long())
    #         loss.backward()
    #         optimizer.step()
    #         summary['loss'][e].append(loss.item())

    #         # Calculat accuracy
    #         _, pred = x.data.topk(1, dim=1)
    #         pred = pred.view(pred.shape[:-1])
    #         acc = torch.sum(pred == labels)/x.shape[0]
    #         summary['acc'][e].append(acc.item())
            
    #     print('Loss: {}, Accuracy: {}'.format(np.mean(summary['loss'][e]), np.mean(summary['acc'][e])))
        
        
    # imu_test_set = WISDMSelect(
    #     folds=[5],
    #     time_window=time_window,
    #     overwrite=False)

    # imu_test_loader = DataLoader(imu_test_set, batch_size=1, 
    #                           shuffle=True, num_workers=4)

    # test_accuracy = []
    # for i, (imus, labels) in enumerate(imu_test_loader):
    #         # Run the Net
    #         x = net(imus)
    #         x = x.view(x.size()[:-1])

    #         # loss = criterion(x, labels.long())
    #         # summary['loss'][e].append(loss.item())
    #         # Calculat accuracy
    #         _, pred = x.data.topk(1, dim=1)
    #         pred = pred.view(pred.shape[:-1])
    #         acc = torch.sum(pred == labels)/x.shape[0]
    #         summary['acc'][e].append(acc.item())
    # print(np.mean(summary['acc'][e]))