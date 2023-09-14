import numpy as np
from activities import *


class Stage():
    def __init__(self, name, activity_setting):
        """
        Args:
            name (string): name of the stage
            activity_setting (list): contains the set of activities with proabability of occurence and max_num of occurence (optional)
        """
        self.name = name
        self.activity_list = []
        self.activity_names = []
        self.activity_probs = []

        self._init_activity(activity_setting)
        assert self._check_probs()
        self.n_activity = len(self.activity_list)

        self.action_sequence = []
        self.time_window_elapsed = 0
    
    def _get_name_list(self):
        return [activity.name for activity in self.activity_list]
        
    def _check_probs(self):
        return sum(self.activity_probs) == 1
    
    # def renormalize(self):
    #     prob_sum = 0
    #     prob_sum += sum([tuple[1] for tuple in self.normal_activities])
    #     prob_sum += sum([tuple[1] for tuple in self.limited_activities])

    #     for tuple in self.normal_activities:
    #         tuple[1] /= prob_sum
    #     for tuple in self.limited_activities:
    #         tuple[1] /= prob_sum

    def _init_activity(self, activity_setting):
        """
        Generate a list of Activity object accordingly
        """
        for act in activity_setting:
            name, prob, n_max = act

            if name == 'restroom':
                self.activity_list.append(RestroomActivity())
            elif name == 'walk_only':
                self.activity_list.append(WalkingActivity())
            elif name == 'sit_only':
                self.activity_list.append(SittingActivity())
            elif name == 'work':
                self.activity_list.append(WorkingActivity())
            elif name == 'drink_only':
                self.activity_list.append(DrinkingActivity())
            elif name == 'oral_clean':
                self.activity_list.append(OralCleaningActivity())
            else:
                raise Exception("No matching activity class.")
            
            self.activity_probs.append(prob)
            self.activity_names.append(name)

    def _add_activity(self, activity_name):
        idx = self.activity_names.index(activity_name)
        activity = self.activity_list[idx]

        actions, time_window = activity.generate_activity()
        self.action_sequence.extend(actions)
        self.time_window_elapsed += time_window


class DaytimeStage(Stage):
    """
    Daytime stage ontains drinking, sitting, walking, working and using restroom activities.
    """
    def __init__(self, activity_setting=[('drink_only', 0.04, None), ('sit_only', 0.27, None), ('walk_only', 0.27, None), ('work', 0.4, None), ('restroom', 0.02, None)]):
        super().__init__(name='daytime', activity_setting=activity_setting)

    def generate_actions(self):
        """
        Randomly select activities based on probabilities
        """
        prob_threshold = 0
        p = np.random.rand()

        for i in range(self.n_activity):
            prob_threshold += self.activity_probs[i]
            act = self.activity_list[i]
            if p < prob_threshold:
                self._add_activity(act.name)
                break

        
class EveningStage(Stage):
    """
    Evening stage ontains drinking, sitting, walking and using restroom activities.
    """
    def __init__(self, activity_setting=[('drink_only', 0.04, None), ('sit_only', 0.57, None), ('walk_only', 0.37, None), ('restroom', 0.02, None)]):
        super().__init__(name='evening', activity_setting=activity_setting)

    def generate_actions(self):
        """
        Randomly select activities based on probabilities
        """
        prob_threshold = 0
        p = np.random.rand()

        for i in range(self.n_activity):
            prob_threshold += self.activity_probs[i]
            act = self.activity_list[i]
            if p < prob_threshold:
                self._add_activity(act.name)
                break

class DailyCareStage(Stage):
    """
    Daily care stage contains the oral cleanining activity.
    """
    def __init__(self, activity_setting=[('oral_clean', 1, None)]):
        super().__init__(name='dailycare', activity_setting=activity_setting)

    def generate_actions(self):
        """
        Performing oral cleaning activity
        """
        act = self.activity_list[0]
        self._add_activity(act.name)

class MealStage(Stage):
    """
    Meal stage contains having meal activity.
    """
    def __init__(self, activity_setting=[('have_meal', 1, None)]):
        super().__init__(name='meal', activity_setting=activity_setting)

    def generate_actions(self):
        """
        Having meals
        """
        act = self.activity_list[0]
        self._add_activity(act.name)


#TODO: print a log for all the hyperparamter
"""
complex event: 
    brushing teeth before 10 pm, brushing twice a day and each time longer than 2 min
    wash hands before having meals
"""
# stage refers to 
    

