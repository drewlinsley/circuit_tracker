from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/datasets/got10k/'
    settings.got_packed_results_path = '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/TransT/got_results_packed'
    settings.got_reports_path = '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/TransT/got_results'
    settings.lasot_path = '/dev/tracking/lasot/LaSOTBenchmark'
    # settings.lasot_path = '/media/data/tracking/lasot_extended/LaSOT_extension_subset'
    settings.network_path = '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/TransT/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/TransT/pytracking/result_plots/'
    settings.results_path = '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/TransT/pytracking/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/media/data_cifs/projects/prj_tracking/pytorch_hGRU/TransT/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    # settings.trackingnet_path = '/home/dlinsley/tracking/trackingnet/TrackingNet/'
    settings.trackingnet_path = '../datasets/TrackingNetTest'  # '/home/dlinsley/tracking/trackingnet/TrackingNet'
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

