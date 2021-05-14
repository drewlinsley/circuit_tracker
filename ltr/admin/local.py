class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/TransT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/dev/tracking/lasot/LaSOTBenchmark'
        self.got10k_dir = '/home/dlinsley/tracking/got10k/train'  # '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/datasets/got10k/train'
        # self.got10k_dir = '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/datasets/got10k/train'
        self.trackingnet_dir = '/home/dlinsley/tracking/trackingnet/TrackingNet'
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'

