from pylon.constraint import constraint


def enforce_pattern(*logits, **kwargs):
    '''
    logits_sequence: a logit tensor (batch_size x T x num_classes) returned by the primitive classifier
    kwargs['event_label']: ground truth complex event label
    kwargs['event_fsm']: a list of finite state machines for corresponding complex event class
    kwargs['n_event_class]: the number of complex event classes
    '''
    # Expects tensors of shape: batch_size x ... x num_classes
    w = kwargs['window_size']
    max_size = kwargs['max_blc_size']

    
    
    return True

enfore_pattern_cons = constraint(enforce_pattern)

def has_pos_block(sequence, max_size):
    '''
    
    '''
    current = None
    count = 0

    for l in sequence:
        if l > 0:
            if l == current:
                count += 1
                if count >= max_size: 
                    return True
            else:
                current = l
                count = 1
        else:
            current = None
            count = 0

    return False



