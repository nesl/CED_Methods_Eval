class FSM():
    def __init__(self) -> None:
        pass

    def get_current_state(self):
        return self.state
    
    def update_state(self, input):
        raise NotImplementedError
    

class Event1FSM(FSM):
    def __init__(self):
        super().__init__()
        self.event_label = 1
        self.state = 's1_0'
        self.s1_counter = 0

    def update_state(self, input):
        if self.state == 's1_0':
            if input == 'flush_toilet':
                self.state = 's1_1'
            return 0
        elif self.state == 's1_1':
            if input == 'wash':
                self.state = 's1_0'
                self.s1_counter = 0
                return 0
            elif input == 'flush_toilet':
                return 0
            elif input == 'walk':
                self.s1_counter += 1
                if self.s1_counter > 12:
                    self.state = 's1_0'
                    self.s1_counter = 0
                    return self.event_label
                return 0
            else:
                self.state = 's1_0'
                self.s1_counter = 0
                return self.event_label
        
        raise Exception("Failed to return.")
    

class Event2FSM(FSM):
    def __init__(self):
        super().__init__()
        self.event_label = 2
        self.state = 's2_0'
        self.s1_counter = 0
        self.s2_counter = 0

    def update_state(self, input):
        if self.state == 's2_0':
            if input == 'wash':
                self.state = 's2_1'
                return 0
            elif input == 'eat':
                self.state = 's2_2'
                return self.event_label
            else: 
                return 0
        elif self.state == 's2_1':
            if input == 'wash':
                return 0
            elif input == 'eat':
                self.state = 's2_2'
                return 0
            elif input == 'flush_toilet':
                self.state = 's2_0'
                self.s1_counter = 0
                return 0
            else:
                # count other activities
                self.s1_counter += 1
                if self.s1_counter > 12:
                    self.state = 's2_0'
                    self.s1_counter = 0
                    return 0
                else:
                    return 0
        elif self.state == 's2_2':
            if input == 'eat':
                self.s2_counter = 0
                return 0
            else:
                self.s2_counter += 1
                if self.s2_counter > 36:
                    self.state = 's2_0'
                    self.s2_counter = 0
                return 0

        raise Exception("Failed to return.")
    

class Event3FSM(FSM):
    def __init__(self):
        super().__init__()
        self.event_label = 3
        self.state = 's3_0'
        self.s1_counter = 0 # count the total time of brush actions
        self.s2_counter = 0 # count the time of other actions after the last brush action happens

    def update_state(self, input):
        if self.state == 's3_0':
            if input == 'brush_teeth':
                self.state = 's3_1'
                self.s1_counter += 1
            return 0
        elif self.state == 's3_1':
            if input == 'brush_teeth':
                self.s1_counter += 1
            else:
                self.state = 's3_2'
                self.s2_counter += 1
            return 0
        elif self.state == 's3_2':
            if input == 'brush':
                # reinitialize s2_counter
                self.state = 's3_1'
                self.s1_counter += 1
                self.s2_counter = 0
                return 0
            else:
                self.s2_counter += 1
                s1_record = self.s1_counter
                if self.s2_counter > 2:
                    self.state = 's3_0'
                    self.s1_counter = 0
                    self.s2_counter = 0
                    if s1_record < 24:
                        return self.event_label
                    else: 
                        return 0
                else:
                    return 0
                
        raise Exception("Failed to return.")
