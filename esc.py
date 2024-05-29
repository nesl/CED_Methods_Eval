import os
import wavio
import shutil
import random
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
from torchaudio import transforms
import torch
import librosa
import subprocess
from abc import ABC
import configparser


package_dir, _ = os.path.split(os.path.abspath(__file__))

config = configparser.ConfigParser()
for fName in ['config_local.ini', 'config.ini']:
    f_path = join(package_dir, fName)
    if os.path.isfile(f_path):
        config.read(f_path)
        continue


def compute_mfcc(sound, rate, frame=512):
    '''MFCC computation with default settings
    (2048 FFT window length, 512 hop length, 128 bands)'''
    melspectrogram = librosa.feature.melspectrogram(sound,
                                                    sr=rate,
                                                    hop_length=frame)
    logamplitude = librosa.amplitude_to_db(melspectrogram)
    mfcc = librosa.feature.mfcc(S=logamplitude, n_mfcc=13).transpose()
    return mfcc
 

def group(iterator, count):
    '''Group an iterator (like a list) in chunks of <count>'''
    itr = iter(iterator)
    while True:
        yield tuple([next(itr) for i in range(count)])


def compute_zcr(sound, frame_size=512):
    '''Compute zero crossing rate'''
    zcr = []
    for frame in group(sound, frame_size):
        zcr.append(np.nanmean(0.5 * np.abs(np.diff(np.sign(frame)))))

    zcr = np.asarray(zcr)
    return zcr


def convert_ar(src_path, dst_path, ar):
    if not os.path.isfile(dst_path):
        cmd = 'ffmpeg -i "{}" -ac 1 -ar {} -loglevel error -y "{}"'.format(
            src_path, ar, dst_path)
        subprocess.call(cmd, shell=True)


def random_crop(sound, size):
    org_size = len(sound)
    start = random.randint(0, org_size - size)
    return sound[start: start + size]


def normalize(factor):
    def f(sound):
        return sound / factor
    return f


class ESC(Dataset, ABC):
    """Abstract class for the ESC datasets accessible with pytorch"""

    def __init__(self,
                 csv_file,
                 root,
                 time_window,
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        """
        Args:
            root (string): Root directory of dataset where directory
                ``esc`` exists
            csv_file (string): name of the csv_file
            folds (array, int): folds to call for.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            use_bc_learning (bool): Use the between classes learning approch
            audio_rate (int): audio rate to use for the learning
            overwrite (bool): overwrite existing npz file
        """

        # Dataset generation
        self.root = root
        self.audio_rate = audio_rate
        self.input_size = int(self.audio_rate * time_window)
        self.threshold_sound = threshold_sound
        self.csv_file = csv_file
        self.transforms = transforms

        # Process root and related values
        if type(root) is list:
            root = root[0]
        self.db_path = join(root, './audio/{}_{}.npz'.format(
            type(self).__name__, self.input_size))

        # Read the csv and process it
        if 'df' not in self.__dir__():
            self.df = pd.read_csv(join(root, csv_file))
        self.dfs = self.df
        if type(self.df) is list:
            self.df = self.df[0]._append(self.df[1:])
            # self.df = pd.concat([self.df[0], self.df[1:]])

        self.classes = self._ordered_classes()
        self.nClasses = len(self.classes)

        # Maybe create dataset
        if not os.path.isfile(self.db_path) or overwrite:
            self._create_dataset()

        # Get item processing
        self.folds = folds
        self.use_bc_learning = use_bc_learning
        self.strongAugment = strong_augment
        self.get_db_folds()  # Fils self.sounds and self.labels

        # Maybe compute mfcc and zcr
        if compute_features:
            self.compute_features()

    def compute_features(self):
        print('Computing features...')
        self.mfcc = []
        self.zcr = []
        for s in self.sounds:
            s = s / float(2 ** 16 / 2)
            self.mfcc.append(compute_mfcc(s, self.audio_rate, 512))
            self.zcr.append(compute_zcr(s, 512))

    def _ordered_classes(self):
        '''Retrieve classes from df ignoring target if there is more than
        one category name per target'''
        df = self.df
        if(set(zip(df.target, df.category)) > set(df.target)):
            # More than a category per target, create new ordering (alphabetic)
            classes = sorted(set(df.category))
        else:
            # No problem in target and category, create a list of classes
            # in odered by target value
            classes = set(zip(df.target, df.category))
            classes = sorted(classes, key=lambda x: x[0])
            classes = [c[1] for c in classes]
        return classes

    def __len__(self):
        return len(self.sounds)

    def get_db_folds(self):
        full_db = np.load(self.db_path, allow_pickle=True)
        self.sounds = []
        self.labels = []
        self.folds_nb = []
        for fold in self.folds:
            fold_name = 'fold{}'.format(fold)
            print('loading ', fold_name)
            sounds = full_db[fold_name].item()['sounds']
            labels = full_db[fold_name].item()['labels']
            self.sounds.extend(sounds)
            self.labels.extend(labels)
            self.folds_nb.extend([fold, ] * len(labels))
        
        self.sounds = [self.preprocess(torch.tensor(s, dtype=torch.float).unsqueeze(0)) for s in self.sounds] # PyTorch expects the input tensor and model parameters to have the same dtype, since the model parameters are initialized as FloatTensors, we need to change the input to torch.float.
        self.labels = [torch.tensor(l).int() for l in self.labels]

    def preprocess(self, sound):
        # normalize sound 
        factor = 32768.0 # TODO: change this to np.max(), by changeing self.sounds and self.labels into tensors (instead of list of tensors)
        sound = normalize(factor)(sound)

        if self.transforms != []:
            sound = self.transforms(sound)
        return sound
    
    def get_label_mapping(self):
        classes = self.classes
        mapping = {}
        for c in classes:
            id = classes.index(c)
            mapping[c] = id
        return mapping

    def __getitem__(self, idx):
        sound = self.sounds[idx]
        label = self.labels[idx]
        return sound, label
    
    def _create_dataset(self):
        # Root and df to lists
        roots = self.root
        dfs = self.dfs
        if type(roots) is not list:
            roots = [roots]
        if type(dfs) is not list:
            dfs = [dfs]

        # Convert audio
        print('Converting sounds to {}Hz...'.format(
            self.audio_rate))
        for root, df in zip(roots, dfs):
            for idx, row in df.iterrows():
                src_path = join(root, row.path)
                dst_path = src_path.replace('audio/', 'tmp/')
                os.makedirs(join(root, 'tmp'), exist_ok=True)
                convert_ar(src_path, dst_path, self.audio_rate)
                # TODO: check torchaudio resample

        # Create npz file
        print('Creating corresponding npz file...')
        dataset = {}
        for fold in range(1, 6):
            fold_name = 'fold{}'.format(fold)
            dataset[fold_name] = {}
            dataset[fold_name]['sounds'] = []
            dataset[fold_name]['labels'] = []
            for root, df in zip(roots, dfs):
                for idx, row in df[df.fold == fold].iterrows():
                    wav_file = row.path.replace('audio/', 'tmp/')
                    wav_file = join(root, wav_file)
                    org_sound = wavio.read(wav_file).data.T[0]
                    n_crop = len(org_sound) // self.input_size
                    if n_crop <= 1:
                        sound = org_sound
                        label = self.classes.index(row.category)
                        dataset[fold_name]['sounds'].append(sound)
                        dataset[fold_name]['labels'].append(label)
                    else:
                        for i in range(n_crop): # randomly crop samples from one sound file
                            sound = random_crop(org_sound, self.input_size)
                            label = self.classes.index(row.category)
                            dataset[fold_name]['sounds'].append(sound)
                            dataset[fold_name]['labels'].append(label)
                    

        print('Saving')
        np.savez(self.db_path, **dataset)
        for root in roots:
            shutil.rmtree(join(root, 'tmp'))


class ESC50(ESC):
    def __init__(self,
                 csv_file='meta/esc50.csv',
                 root=config['Paths']['ESC50'],
                 time_window=5,
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        # Fix path in df
        self.df = pd.read_csv(join(root, csv_file))
        self.df['path'] = 'audio/' + self.df.filename

        super().__init__(
            csv_file=csv_file,
            root=root,
            time_window=time_window,
            threshold_sound=threshold_sound,
            audio_rate=audio_rate,
            overwrite=overwrite,
            folds=folds,
            transforms=transforms,
            use_bc_learning=use_bc_learning,
            strong_augment=strong_augment,
            compute_features=compute_features)


class ESC10(ESC):
    def __init__(self,
                 csv_file='meta/esc50.csv',
                 root=config['Paths']['ESC10'],
                 time_window=5,
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        # Fix path in df
        self.df = pd.read_csv(join(root, csv_file))
        self.df = self.df[self.df.esc10 == True]
        self.df['path'] = 'audio/' + self.df.filename

        super().__init__(
            csv_file=csv_file,
            root=root,
            time_window=time_window,
            threshold_sound=threshold_sound,
            audio_rate=audio_rate,
            overwrite=overwrite,
            folds=folds,
            transforms=transforms,
            use_bc_learning=use_bc_learning,
            strong_augment=strong_augment,
            compute_features=compute_features)


class ESC70(ESC):
    def __init__(self,
                 csv_file=['meta/esc50.csv',
                           'kitchen20.csv'],
                 root=[config['Paths']['ESC50'],
                       config['Paths']['KITCHEN20']],
                 time_window=5,
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        # Array of 2 dataframes for ESC50 and kitchen20
        dfs = []
        for r, f in zip(root, csv_file):
            df = pd.read_csv(join(r, f))
            if 'path' not in df.columns:
                df['path'] = 'audio/' + df.filename
            dfs.append(df)
        self.df = dfs

        super().__init__(
            csv_file=csv_file,
            root=root,
            time_window=time_window,
            threshold_sound=threshold_sound,
            audio_rate=audio_rate,
            overwrite=overwrite,
            folds=folds,
            transforms=transforms,
            use_bc_learning=use_bc_learning,
            strong_augment=strong_augment,
            compute_features=compute_features)


class Kitchen20(ESC):
    def __init__(self,
                 csv_file='kitchen20.csv',
                 root=config['Paths']['KITCHEN20'],
                 time_window=5,
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        super().__init__(
            csv_file=csv_file,
            root=root,
            time_window=time_window,
            threshold_sound=threshold_sound,
            audio_rate=audio_rate,
            overwrite=overwrite,
            folds=folds,
            transforms=transforms,
            use_bc_learning=use_bc_learning,
            strong_augment=strong_augment,
            compute_features=compute_features)


class ESC70Select(ESC):
    def __init__(self,
                 csv_file=['meta/esc50.csv',
                           'kitchen20.csv',
                           'silent_sound.csv'],
                 root=[config['Paths']['ESC50'],
                       config['Paths']['KITCHEN20'],
                       config['Paths']['SILENTSOUND']],
                 class_selected = ['footsteps', 'drinking_sipping', 'brushing_teeth', 'mouse_click', 'keyboard_typing', 'toilet_flush', \
                                   'blender', 'stove-burner', 'clean-dishes', 'chopping', 'drawer', 'water-flowing', 'peel', 'eat', 'no_sound'],
                 time_window=5,
                 threshold_sound=0,
                 audio_rate=16000,
                 overwrite=False,
                 folds=[1, 2, 3, 4, 5],
                 transforms=[],
                 use_bc_learning=False,
                 strong_augment=False,
                 compute_features=False):
        # Array of 2 dataframes for ESC50 and kitchen20 and only select classes of interest
        dfs = []
        for r, f in zip(root, csv_file):
            print(join(r, f))
            df = pd.read_csv(join(r, f))
            df = df[df['category'].isin(class_selected)]
            if 'path' not in df.columns:
                df['path'] = 'audio/' + df.filename
            dfs.append(df)
        self.df = dfs

        super().__init__(
            csv_file=csv_file,
            root=root,
            time_window=time_window,
            threshold_sound=threshold_sound,
            audio_rate=audio_rate,
            overwrite=overwrite,
            folds=folds,
            transforms=transforms,
            use_bc_learning=use_bc_learning,
            strong_augment=strong_augment,
            compute_features=compute_features)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torch import nn
    from torch import optim

    time_window = 1
    audio_rate = 16000
    input_length = int(audio_rate * time_window)

    audio_set = ESC70Select(
        # root='./Audio/kitchen20/',
        time_window=time_window,
        folds=[1, 2, 3, 4],
        # transforms=transforms.Compose([
        #     transforms.RandomStretch(1.25),
        #     transforms.Scale(2 ** 16 / 2),
        #     transforms.Pad(input_length // 2),
        #     transforms.RandomCrop(input_length),
        #     transforms.RandomOpposite()]),
        # transforms=lambda x : x[:,:20000], 
        transforms=lambda x : nn.functional.pad(x, ((input_length - x.shape[1]) // 2, (input_length - x.shape[1]) // 2)) if (x.shape[1] % 2) == 0 \
            else nn.functional.pad(x, ((input_length - x.shape[1]) // 2, (input_length - x.shape[1]) // 2 + 1)), 
        overwrite=False,
        use_bc_learning=False,
        audio_rate=audio_rate)

    n_class = audio_set.nClasses
    audio_loader = DataLoader(audio_set, batch_size=2, # I changed batch_size from 2 to 1 to prevent torch from trying to stack things with different shapes up.
                              shuffle=True)

    # Load a sample network
    net = nn.Sequential(
        nn.Conv1d(1, 32, 9, 3), nn.ReLU(), nn.BatchNorm1d(32),
        nn.Conv1d(32, 32, 9, 3), nn.ReLU(), nn.BatchNorm1d(32),
        nn.Conv1d(32, 32, 9, 3), nn.ReLU(), nn.BatchNorm1d(32),
        nn.Conv1d(32, n_class, 9, 3), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    sounds = audio_loader.dataset.sounds
    labels = audio_loader.dataset.labels 
    print(audio_set.get_label_mapping())
    # Training loop
    n_epochs = 20
    summary = {'loss': [[] for _ in range(n_epochs)], 'acc': [[] for _ in range(n_epochs)]}
    for e in range(n_epochs):
        for i, (sounds, labels) in enumerate(audio_loader):
            # if sounds.shape != 81920:
            #     print(sounds.shape)
            #     print(labels)
            # print(i)
            # # Zero the grads
            optimizer.zero_grad()

            # Run the Net
            x = net(sounds)
            x = x.view(x.size()[:-1])

            # Optimize net
            loss = criterion(x, labels.long())
            loss.backward()
            optimizer.step()
            summary['loss'][e].append(loss.item())

             # Calculat accuracy
            _, pred = x.data.topk(1, dim=1)
            pred = pred.view(pred.shape[:-1])
            acc = torch.sum(pred == labels)/x.shape[0]
            summary['acc'][e].append(acc.item())
            
        print('Loss: {}, Accuracy: {}'.format(np.mean(summary['loss'][e]), np.mean(summary['acc'][e])))

