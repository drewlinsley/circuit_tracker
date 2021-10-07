
CUDA_VISIBLE_DEVICES=6 python pytracking/run_nfl_video.py --tracker_name=transt_readout_test_encoder_mult --tracker_param=nfl --videofile=../../nfl/train/58106_002918_Endzone.mp4 --detections=../../nfl/train_baseline_helmets.csv --output_file=../../nfl/tracks/
CUDA_VISIBLE_DEVICES=6 python pytracking/run_all_nfl_videos.py --tracker_name=transt_readout_test_encoder_mult --tracker_param=nfl --videofile=/media/data_cifs/projects/prj_tracking/nfl/train/ --detections=../../nfl/train_baseline_helmets.csv --output_dir=../../nfl/train_tracks/
