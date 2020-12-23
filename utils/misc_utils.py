import collections

def read_split_file( filepath):
    '''
    Read data split txt file provided by KITTI dataset authors
    '''
    with open(filepath) as f:
        trajs = f.readlines()
    trajs = [ x.strip() for x in trajs ]

    return trajs

def update_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update_dict(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        else:
            orig_dict[key] = val
    return orig_dict

def m_makedir(dirpath):
    import os
    if not os.path.exists(dirpath):
        os.makedirs( dirpath)

def split_frame_list(frame_list, t_win_r):
    r'''
    split the frame_list into two : ref_frame (an array) and src_frames (a list),
    where ref_frame = frame_list[t_win_r]; src_frames = [0:t_win_r, t_win_r :]
    '''
    nframes = len(frame_list)
    ref_frame = frame_list[t_win_r]
    src_frames = [ frame_list[idx] for idx in range( nframes) if idx != t_win_r ]
    return ref_frame, src_frames

def get_entries_list_dict(list_dict, keyname):
    r'''
    Given the list of dicts, and the keyname
    return the list [list_dict[0][keyname] ,... ]
    '''
    return [_dict[keyname] for _dict in list_dict ]

def get_entries_list_dict_level(list_dict, keyname, lname):
    r'''
    Given the list of dicts, and the keyname
    return the list [list_dict[0][keyname] ,... ]
    '''
    return [_dict[lname][keyname] for _dict in list_dict ]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3, names=None):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)
        self.names = names
        if names is not None:
            assert self.meters == len(self.names)
        else:
            self.names = [''] * self.meters

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = [0] * i

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        if not isinstance(n, list):
            n = [n] * self.meters
        assert (len(val) == self.meters and len(n) == self.meters)
        for i in range(self.meters):
            self.count[i] += n[i]
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n[i]
            self.avg[i] = self.sum[i] / self.count[i]

    def __repr__(self):
        val = ' '.join(['{} {:.{}f}'.format(n, v, self.precision) for n, v in
                        zip(self.names, self.val)])
        avg = ' '.join(['{} {:.{}f}'.format(n, a, self.precision) for n, a in
                        zip(self.names, self.avg)])
        return '{} ({})'.format(val, avg)
