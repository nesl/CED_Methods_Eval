import numpy as np
import json
from fsm import *


def load_dataset(data_path, config_file):    
    with open(config_file, 'r') as f:
        dataset_config = json.load(f)
    dataset = np.load(data_path, allow_pickle=True)
    
    return dataset, dataset_config


class Activity():
    def __init__(self, name, fsm_list=None, simple_label=True):
        """
        Args:
            name (string): the name of this activity
            prob (float): the probability of this activity
        """
        self.name = name
        self.fsm_list = fsm_list
        self.simple_label = simple_label

        self.data_path = './Multimodal/fusion_5_audio1234_imu1234_embeddings.npz'
        self.config_file = './Multimodal/dataset_config.json'

        dataset, data_config = load_dataset(self.data_path, self.config_file)
        self.nClasses = data_config['nClasses']
        self.label_mapping = data_config['label_mapping']
        self.data, self.class_index_list = self.get_data(dataset)

        if self.fsm_list is not None:
            self.label_sequence = []
        self.action_sequence = []
        self.data_sequence = []
        self.action_label_sequence = []
        self.action_length = {}
        self.time_window_elapsed = 0
        self._define_actions()

    def _define_actions(self):
        """
        Define self.action_length['act_x'] = (min_n_windows, max_n_windows)
            - 'act_x' has a random number of window size, within the range of (min_n_windows, max_n_windows) 
            - actual time = num of windows * window size
        """
        raise NotImplementedError
    
    def generate_activity(self):
        """
        Each activity is composed of atomic actions from the multimodal datset:
        {
            'walk', 
            'brush_teeth',
            'click_mouse',
            'drink',
            'eat',
            'type', 
            'flush_toilet', 
            'use_blender', 
            'use_stove_burner', 
            'clean_dishes', 
            'chop',
            'open_drawer',
            'wash',
            'peel'
            # need to add: 'sit'
        }
        """
        raise NotImplementedError
    
    def generate_label(self):
        assert self.fsm_list is not None
        if self.simple_label is True:
            return [max(self.label_sequence)]
        return self.label_sequence

    def _add_actions(self, action):
        """
        Add actions for a random number of windows and update time window elapsed
        """
        action_t_min, action_t_max = self.action_length[action]
        action_t = np.random.randint(action_t_min, action_t_max + 1)
        action_id = self.label_mapping[action]

        for _ in range(action_t):
            self.action_sequence.append(action)
            # Randomly get the atomic event data
            action_data = self.data[np.random.choice(self.class_index_list[action_id])]
            self.data_sequence.append(action_data)
            self.action_label_sequence.append(action_id)

            # Generate complex event label if FSMs are given
            if self.fsm_list is not None:
                ce_label = 0
                for fsm in self.fsm_list:
                    l = fsm.update_state(input=action)
                    if l > 0: ce_label = l
                self.label_sequence.append(ce_label)

        self.time_window_elapsed += action_t

    def get_data(self, dataset):
        data = dataset['embeddings']
        label = dataset['labels']
        class_index_list = []

        for i in range(self.nClasses):
            indices = np.where(label==i)[0]
            class_index_list.append(indices)

        return data, class_index_list
        


class RestroomActivity(Activity):
    def __init__(self, enforce_window_length=None, fsm_list=None, simple_label=True):
        super().__init__(name='restroom', fsm_list=fsm_list, simple_label=simple_label)
        self.enforce_window_length = enforce_window_length

    def _define_actions(self):
        """
        window size =  5s
        """
        self.action_length['walk'] = (1, 12) # 5s - 1min
        self.action_length['sit'] = (2, 24) # 10s - 2min
        self.action_length['flush_toilet'] = (1, 3) # 5s - 15s
        self.action_length['wash'] = (1, 12) # 5s - 1min

    def generate_activity(self):
        """
        Synthesize the restroom activity:
            - walk1 -> wash1? -> sitting -> flush -> wash2? -> walk2 ('?' means a random action that may not happen)
        """
        wash_prob1 = 0.1
        wash_prob2 = 0.7

        # walk action (walk in)
        self._add_actions('walk')
        
        # wash action (wash hands - may not happen)
        if np.random.rand() < wash_prob1:
            self._add_actions('wash')

        # sit action (sit on toilet)
        self._add_actions('sit')

        # flush action (flush the toilet)
        self._add_actions('flush_toilet')

        # wash action (wash hands - may not happen)
        if np.random.rand() < wash_prob2:
            self._add_actions('wash')

        # walk action (walk away)
        self._add_actions('walk')

        # Generate sequence of fixed length
        if self.enforce_window_length is not None:
            # Truncate the sequence
            if self.time_window_elapsed > self.enforce_window_length:
                self.action_sequence = self.action_sequence[:self.enforce_window_length]
                self.data_sequence = self.data_sequence[:self.enforce_window_length]
                self.action_label_sequence = self.action_label_sequence[:self.enforce_window_length]
                if self.fsm_list is not None:
                    self.label_sequence = self.label_sequence[:self.enforce_window_length]
            # Extend the sequence with the last action
            elif self.time_window_elapsed < self.enforce_window_length:
                add_window_length = self.enforce_window_length - self.time_window_elapsed
                for _ in range(add_window_length):
                    self.action_sequence.append('walk')
                    action_id = self.label_mapping['walk']
                    action_data = self.data[np.random.choice(self.class_index_list[action_id])]
                    self.data_sequence.append(action_data)
                    self.action_label_sequence.append(action_id)
                    # Generate complex event label if FSMs are given
                    if self.fsm_list is not None:
                        ce_label = 0
                        for fsm in self.fsm_list:
                            l = fsm.update_state(input='walk')
                            if l > 0: ce_label = l
                    self.label_sequence.append(ce_label)

            self.time_window_elapsed = len(self.action_sequence)

        return self.action_sequence, self.data_sequence, self.action_label_sequence, self.time_window_elapsed
    

class WalkingActivity(Activity):
    def __init__(self):
        super().__init__(name='walk_only')

    def _define_actions(self):
        """
        window size =  5s
        """
        self.action_length['walk'] = (1, 180) # 5s - 15min

    def generate_activity(self):
        """
        Synthesize the walking only activity:
            - walk
        """

        self._add_actions('walk')

        return self.action_sequence, self.data_sequence, self.action_label_sequence, self.time_window_elapsed
    

class SittingActivity(Activity):
    def __init__(self):
        super().__init__(name='sit_only')

    def _define_actions(self):
        """
        window size =  5s
        """
        self.action_length['sit'] = (60, 360) # 5min - 30min

    def generate_activity(self):
        """
        Synthesize the sitting still activity:
            - sit
        """
        self._add_actions('sit')

        return self.action_sequence, self.data_sequence, self.action_label_sequence, self.time_window_elapsed
    

class WorkingActivity(Activity):
    def __init__(self):
        self.action_probs = {}
        super().__init__(name='work')

    def _define_actions(self):
        """
        window size =  5s
        """
        self.action_length['sit'] = (1, 60) # 5s - 5min
        self.action_length['type'] = (1, 4) # 5s - 20s
        self.action_length['click_mouse'] = (1, 4) # 5s - 20s
        self.action_length['drink'] = (1, 3) # 5si]s - 15s

        self.action_probs['sit'] = 0.32
        self.action_probs['type'] = 0.32
        self.action_probs['click_mouse'] = 0.32
        self.action_probs['drink'] = 0.04
        assert sum(self.action_probs.values()) == 1


    def generate_activity(self):
        """
        Synthesize the working activity:
            - randomly switch between sit, type, and click mouse within a given time interval 'totoal_t'
        """
        sit_prob = self.action_probs['sit']
        type_prob = self.action_probs['type']
        click_prob = self.action_probs['click_mouse']
        drink_prob = self.action_probs['drink']

        total_t = np.random.randint(360, 1440 + 1) # 30min - 2h

        while self.time_window_elapsed < total_t:
            prob = np.random.rand()
            if prob < sit_prob: 
                # sit happens
                self._add_actions('sit')

            elif prob < sit_prob + type_prob: 
                # type happens
                self._add_actions('type')

            elif prob < sit_prob + type_prob + click_prob: 
                # click happens
                self._add_actions('click')

            else: 
                # drink happens
                self._add_actions('drink')

        return self.action_sequence, self.data_sequence, self.action_label_sequence, self.time_window_elapsed
    

class DrinkingActivity(Activity):
    def __init__(self):
        super().__init__(name='drink_only')

    def _define_actions(self):
        """
        window size =  5s
        """
        self.action_length['drink'] = (1, 3) # 5s - 15s

    def generate_activity(self):
        """
        Synthesize the drinking-only activity:
            - sit
        """
        self._add_actions('drink')

        return self.action_sequence, self.data_sequence, self.action_label_sequence, self.time_window_elapsed
    

class OralCleaningActivity(Activity):
    def __init__(self, enforce_window_length=None, action_length={}, fsm_list=None, simple_label=True):
        """
        action_length (tuple, dict): key - activity name, value - (min_time, max_time)
        """
        super().__init__(name='oral_clean', fsm_list=fsm_list, simple_label=simple_label)
        self.enforce_window_length = enforce_window_length
        if action_length: # if action_length is not empty
            self.action_length = action_length

    def _define_actions(self):
        """
        window size =  5s
        """
        self.action_length['walk'] = (1, 12) # 5s - 1min
        self.action_length['wash'] = (1, 12) # 5s - 1min
        self.action_length['brush_teeth'] = (3, 48) # 15s - 4min

    def generate_activity(self):
        """
        Synthesize the oral cleaning activity:
            - walk1 -> wash1? -> brush -> wash2 -> walk2 ('?' means a random action that may not happen)
        """
        wash_prob = 0.6

        # walk action (walk in)
        # self._add_actions('walk')

        # wash action (wash before brushing teeth - may not happen)
        if np.random.rand() < wash_prob:
            self._add_actions('wash')

        # brush_teeth action
        self._add_actions('brush_teeth')

        # wash action after brushing teeth
        self._add_actions('wash')

        # walk action (walk away)
        self._add_actions('walk')

        # Generate sequence of fixed length
        if self.enforce_window_length is not None:
            # Truncate the sequence
            if self.time_window_elapsed >= self.enforce_window_length:
                self.action_sequence = self.action_sequence[:self.enforce_window_length]
                self.data_sequence = self.data_sequence[:self.enforce_window_length]
                self.action_label_sequence = self.action_label_sequence[:self.enforce_window_length]
                if self.fsm_list is not None:
                    self.label_sequence = self.label_sequence[:self.enforce_window_length]
            # Extend the sequence with the last action
            else:
                add_window_length = self.enforce_window_length - self.time_window_elapsed
                for _ in range(add_window_length):
                    self.action_sequence.append('walk')
                    action_id = self.label_mapping['walk']
                    action_data = self.data[np.random.choice(self.class_index_list[action_id])]
                    self.data_sequence.append(action_data)
                    self.action_label_sequence.append(action_id)
                    # Generate complex event label if FSMs are given
                    if self.fsm_list is not None:
                        ce_label = 0
                        for fsm in self.fsm_list:
                            l = fsm.update_state(input='walk')
                            if l > 0: ce_label = l
                    self.label_sequence.append(ce_label)

            self.time_window_elapsed = len(self.action_sequence)

        return self.action_sequence, self.data_sequence, self.action_label_sequence, self.time_window_elapsed


class HavingMealActivity(Activity):
    def __init__(self, enforce_window_length=None, fsm_list=None, simple_label=True):
        super().__init__(name='have_meal', fsm_list=fsm_list, simple_label=simple_label)
        self.enforce_window_length = enforce_window_length

    def _define_actions(self):
        """
        window size =  5s
        """
        self.action_length['walk'] = (1, 12) # 5s - 1min 
        self.action_length['wash'] = (1, 12) # 5s - 1min
        self.action_length['eat'] = (1, 2) # 5s - 10s
        self.action_length['sit'] = (1, 2) # 5s - 10s
        self.action_length['drink'] = (1, 3) # 5s - 15s


    def generate_activity(self):
        """
        Synthesize the having meal activity:
            - walk1 -> wash? -> eat + sit (intermittently) -> drink? -> walk2 ('?' means a random action that may not happen)
        """
        wash_prob = 0.6
        drink_prob = 0.5

        # walk action (walk in)
        self._add_actions('walk')

        # wash action (wash before having meal - may not happen)
        if np.random.rand() < wash_prob:
            self._add_actions('wash')

        # eating action combined with sitting action
        total_eating_t = 0
        if self.enforce_window_length is not None:
            total_eating_t = self.enforce_window_length - self.time_window_elapsed
        else:
            total_eating_t = np.random.randint(180, 360 + 1) # 15min - 30min
        record_time = self.time_window_elapsed
        while (self.time_window_elapsed - record_time) < total_eating_t:
            self._add_actions('eat')
            self._add_actions('sit')

        if self.enforce_window_length is None:
            # drinking action (after having meal - may not happen)
            if np.random.rand() < drink_prob:
                self._add_actions('drink')

            # walk action (walk away)
            self._add_actions('walk')
        else:
            # Truncate the sequence
            self.action_sequence = self.action_sequence[:self.enforce_window_length]
            self.data_sequence = self.data_sequence[:self.enforce_window_length]
            self.action_label_sequence = self.action_label_sequence[:self.enforce_window_length]
            if self.fsm_list is not None:
                self.label_sequence = self.label_sequence[:self.enforce_window_length]
            self.time_window_elapsed = len(self.action_sequence)

        return self.action_sequence, self.data_sequence, self.action_label_sequence, self.time_window_elapsed