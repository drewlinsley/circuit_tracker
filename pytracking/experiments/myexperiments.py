from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset

def transt():
    # Evaluate GOT on a circuit KYS I trained
    trackers = trackerlist('transt', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_readout():
    # Evaluate GOT on a circuit KYS I trained
    trackers = trackerlist('transt_readout', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_readout_mixed():
    # Evaluate GOT on a circuit KYS I trained
    trackers = trackerlist('transt_readout_mixed', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_readout_test_v1():
    trackers = trackerlist('transt_readout_test_v1', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_encoder():
    trackers = trackerlist('transt_encoder', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_readout_test_encoder_mult():
    trackers = trackerlist('transt_readout_test_encoder_mult', 'default', range(1))
    dataset = get_dataset('got10k_test')  # , 'lasot')  # _val, 'got10k_ltrval')
    # dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_control():
    trackers = trackerlist('transt_control', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def rtranst():
    trackers = trackerlist('rtranst', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_control_uniform():
    trackers = trackerlist('transt_control_uniform', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_control_normal():
    trackers = trackerlist('transt_control_normal', 'default', range(1))
    dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

def transt_readout_test_encoder_mult_from_front():
    trackers = trackerlist('transt_readout_test_encoder_mult_from_front', 'default', range(1))
    dataset = get_dataset('got10k_test')  # , 'lasot')  # _val, 'got10k_ltrval')
    # dataset = get_dataset('got10k_test')  # _val, 'got10k_ltrval')
    return trackers, dataset

