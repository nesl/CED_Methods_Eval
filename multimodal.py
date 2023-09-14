import os
import glob
import shutil
import random
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
import torch
import configparser
from esc import *
from wisdm import *


package_dir, _ = os.path.split(os.path.abspath(__file__))

config = configparser.ConfigParser()
for fName in ['config_local.ini', 'config.ini']:
    f_path = join(package_dir, fName)
    if os.path.isfile(f_path):
        config.read(f_path)
        continue


class MultimodalDataset(Dataset):
    """ Class for the multimodal dataset using audio and IMU data """
    
    def __init__(self,
                 audio_dataset: ESC,
                 imu_dataset: WISDM,
                 root=config['Paths']['MULTI_DATA'],
                 num_data_per_class=500,
                 time_window=5,
                 overwrite=False,
                #  embed_model=None
                 ):
        """
        Args:
            audio_dataset (ESC): audio dataset of ESC class
            imu_dataset (WISDM): imu dataset of WISDM class
            num_data_per_class (int): the number of data for each multimodal data class
            time_window (int): the time window for one input (unit: second), should be the same as those of audio and imu dataset
        """
        # self.get_embeddings = True
        # if embed_model is None:
        #     self.get_embeddings = False
        self.time_window = time_window
        self.num_data_per_class = num_data_per_class
        
        # TODO: read the class definition from a json file
        self.class_definition = {'walk':{'audio':'footsteps', 'imu':'walking'}, 
                                 'sit':{'audio':'no_sound', 'imu':'sitting'},
                                 'brush_teeth':{'audio':'brushing_teeth', 'imu':'teeth'},
                                 'click_mouse':{'audio':'mouse_click', 'imu':'sitting'},
                                 'drink':{'audio':'drinking_sipping', 'imu':'drinking'},
                                 'eat':{'audio':'eat', 'imu':'pasta'},
                                 'type':{'audio':'keyboard_typing', 'imu':'typing'},
                                 'flush_toilet':{'audio':'toilet_flush', 'imu':'standing'},
                                #  'use_blender':{'audio':'blender', 'imu':'standing'},
                                #  'use_stove_burner':{'audio':'stove-burner', 'imu':'standing'},
                                #  'clean_dishes':{'audio':'clean-dishes', 'imu':'standing'},
                                #  'chop':{'audio':'chopping', 'imu':'standing'},
                                #  'open_drawer':{'audio':'drawer', 'imu':'standing'},
                                 'wash':{'audio':'water-flowing', 'imu':'standing'},
                                #  'peel':{'audio':'peel', 'imu':'standing'}
                                 }
        self.classes = sorted(self.class_definition.keys())
        self.nClasses = len(self.classes)

        self.db_path = join(root, '{}_audio-{}_{}_imu-{}_{}.npz'.format(
            type(self).__name__, 
            audio_dataset.input_size, 
            ''.join([str(i) for i in audio_dataset.folds]), 
            imu_dataset.input_size, 
            ''.join([str(i) for i in imu_dataset.folds])
            )
        )
        if not os.path.isfile(self.db_path) or overwrite:
            self._create_dataset(audio_dataset, imu_dataset)
        self.get_db() # Get self.sounds, self.imus and self.labels

    def __len__(self):
        return len(self.sounds)
    
    def __getitem__(self, idx):
        # if self.get_embeddings:
        #     sound = self.embeddings[idx]
        sound = self.sounds[idx]
        imu = self.imus[idx]
        label = self.labels[idx]
        return sound, imu, label
    
    def _create_dataset(self, audio_dataset, imu_dataset):
        """ Create a multimodal dataset"""
        dataset = {}
        dataset['sounds'] = []
        dataset['imus'] = []
        dataset['labels'] = []
        # Use tensors for easier operations
        sounds = torch.cat([s.unsqueeze(0) for s in audio_dataset.sounds])
        sounds_labels = torch.cat([l.unsqueeze(0) for l in audio_dataset.labels])
        imus = torch.cat([i.unsqueeze(0) for i in imu_dataset.imus])
        imus_labels = torch.cat([l.unsqueeze(0) for l in imu_dataset.labels])
        # print(imus.shape)

        # Get the mapping from label to class name
        audio_label_dict = audio_dataset.get_label_mapping()
        imu_label_dict = imu_dataset.get_label_mapping()
        # print(imu_label_dict)

        for class_name in self.classes:
            sound_class = self.class_definition[class_name]['audio']
            s_indx = np.where(sounds_labels == audio_label_dict[sound_class])
            sound_data = self._get_samples(sounds[s_indx])
            # print(sounds[s_indx].shape)
            
            imu_class = self.class_definition[class_name]['imu']
            # print(imus.shape)
            a_indx = np.where(imus_labels == imu_label_dict[imu_class])
            imu_data = self._get_samples(imus[a_indx])
            # print(imus[a_indx].shape)

            label = self.classes.index(class_name)

            dataset['sounds'].extend(sound_data)
            dataset['imus'].extend(imu_data)
            dataset['labels'].extend([label for _ in range(len(sound_data))]) 
        print('Saving multimodal dataset...')
        np.savez(self.db_path, **dataset)

    def get_db(self):
        print("Loading...")
        full_db = np.load(self.db_path, allow_pickle=True)
        self.sounds = [torch.tensor(s, dtype=torch.float) for s in full_db['sounds']]
        self.imus = [torch.tensor(a, dtype=torch.float) for a in full_db['imus']]
        self.labels = [torch.tensor(l).int() for l in full_db['labels']]
    
    def _get_samples(self, data):
        """ Return k random samples from data """
        idx = random.choices(range(len(data)), k=self.num_data_per_class)
        samples = [data[i].numpy() for i in idx] # convert back to numpy to save space
        # samples = torch.stack(samples)
        return samples
    
    def get_label_mapping(self):
        classes = self.classes
        mapping = {}
        for c in classes:
            id = classes.index(c)
            mapping[c] = id
        return mapping
    

class MultimodalEmbed(Dataset):
    """ Class for the multimodal embedding dataset"""

    def __init__(self, dataset, config):
        """
        Args:
            dataset (MultimodalDataset): multimodal dataset of audio and imu data
            config (dict): config of the dataset
        """
        self.audio_embeddings = dataset['audio_embeddings']
        self.imu_embeddings = dataset['imu_embeddings']
        self.labels = dataset['labels']
        
        self.db_path = config['db_path']
        self.classes = config['classes']
        self.nClasses = config['nClasses']
        self.time_window = config['time_window']
        self.num_data_per_class = config['num_data_per_class']
        self.label_mapping = config['label_mapping']
    
    def __getitem__(self, index):
        audio_embedding = self.audio_embeddings[index]
        imu_embedding = self.imu_embeddings[index]
        label = self.labels[index]
        return audio_embedding, imu_embedding, label
        
    def __len__(self):
        return len(self.labels)


class FusionEmbed(Dataset):
    """ Class for the fusion embedding dataset"""

    def __init__(self, dataset, config):
        """
        Args:
            dataset (MultimodalEmbed): multimodal dataset of audio and imu embeddings
            config (dict): config of the dataset
        """
        self.embeddings = dataset['embeddings']
        self.labels = dataset['labels']
        
        self.db_path = config['db_path']
        self.classes = config['classes']
        self.nClasses = config['nClasses']
        self.time_window = config['time_window']
        self.num_data_per_class = config['num_data_per_class']
        self.label_mapping = config['label_mapping']
    
    def __getitem__(self, index):
        embedding = self.embeddings[index]
        label = self.labels[index]
        return embedding, label
        
    def __len__(self):
        return len(self.labels)
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torch import nn
    from torch import optim

    time_window = 0.5
    audio_rate = 16000
    audio_input_length = int(audio_rate * time_window)

    audio_set = ESC70Select(
        # root='./Audio/kitchen20/',
        root=['./Audio/ESC50', './Audio/kitchen20', './Audio/silent_sound'],
        time_window=time_window,
        folds=[1, 2, 3, 4],
        transforms=lambda x : nn.functional.pad(x, ((audio_input_length - x.shape[1]) // 2, (audio_input_length - x.shape[1]) // 2)) if (x.shape[1] % 2) == 0 \
            else nn.functional.pad(x, ((audio_input_length - x.shape[1]) // 2, (audio_input_length - x.shape[1]) // 2 + 1)),  
        overwrite=False,
        use_bc_learning=False,
        audio_rate=audio_rate)
    
    imu_set = WISDMSelect(
        folds=[1, 2, 3, 4],
        time_window=time_window,
        overwrite=False)

    multimodal_set = MultimodalDataset(audio_set, imu_set, time_window=time_window)

    sounds = torch.cat([s.unsqueeze(0) for s in multimodal_set.sounds])
    imus = torch.cat([i.unsqueeze(0) for i in multimodal_set.imus])
    labels = torch.cat([l.unsqueeze(0) for l in multimodal_set.labels])
    print(sounds.shape)
    print(imus.shape)
    print(labels.shape)
    print(multimodal_set.get_label_mapping())
    # data_loader = DataLoader(multimodal_set, batch_size=2, 
    #                           shuffle=True, num_workers=4)

    # print(multimodal_set.sounds)

    # multimodal_embed_set = MultimodalDataset(audio_set, imu_set, time_window=time_window, embed_model=False)