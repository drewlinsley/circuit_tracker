## Update kys/default with checkpoint location. Copy checkpoints to pytracking/networks
# 1. Copy checkpoint to pytracking/networks
# 2. Update pytracking/myexperiments.py
# 3. Create a model specific parameter in pytracking/parameters
# 4. Create a model specific tracker in pytracking/tracker

# pytracking/tracker/transt/__init__.py
# That file needs to be commented when running non-transt models

# python pytracking/run_experiment.py myexperiments got_retrained_bl  # BASELINE
# python pytracking/run_experiment.py myexperiments got_circuit_bl
# python pytracking/run_experiment.py myexperiments got_circuit_dual_bl
# python pytracking/run_experiment.py myexperiments got_circuit_dual_trans_bl
# CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments transt_readout --threads 10
# python pack_got.py transt_readout default transt_readout

# CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments transt_readout_test_v1  #  --threads 5
# python pack_got.py transt_readout_test_v1 default transt_readout_test_v1
 
# CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments transt_encoder  #  --threads 5
# python pack_got.py transt_encoder default transt_encoder

# CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=uniform --threads 5
# python pack_got.py transt default transt

# CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt
# python pack_got.py transt default transt

rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult --threads 4
python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_base
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult --noise=uniform  # --threads 2
python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_uniform
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult --noise=normal  # --threads 2
python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_normal
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult --noise=gamma  # --threads 2
python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_gamma
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult

CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control
python pack_got.py transt_control default transt_control_base


# Control uniform noise mag
rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=uniform --noise_mag=1
python pack_got.py transt_control default transt_control_uniform_1
rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=uniform --noise_mag=0.1
python pack_got.py transt_control default transt_control_uniform_01
rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=uniform --noise_mag=0.01
python pack_got.py transt_control default transt_control_uniform_001

# Control invert
rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=invert_color --noise_mag=1
python pack_got.py transt_control default transt_control_invert
rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=rand_invert_color --noise_mag=1
python pack_got.py transt_control default transt_control_rand_invert
rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=occlusion
python pack_got.py transt_control default transt_control_occlusion
rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=rand_gaussian --noise_mag=0.1
python pack_got.py transt_control default transt_control_rand_gaussian_01
rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=rand_uniform --noise_mag=0.1
python pack_got.py transt_control default transt_control_rand_uniform_01

# Our model
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult --noise=invert_color --noise_mag=0.1  # --threads 2
python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_invert_color
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult --noise=rand_invert_color --noise_mag=0.1  # --threads 2
python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_rand_invert_color
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult --noise=occlusion --noise_mag=0.1  # --threads 2
python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_occlusion
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult --noise=rand_uniform --noise_mag=0.1  # --threads 2
python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_rand_uniform
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult --noise=rand_gaussian --noise_mag=0.1  # --threads 2
python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_rand_gaussian
rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult





rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=normal
python pack_got.py transt_control default transt_control_normal
rm -rf pytracking/tracking_results/transt_control
CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control --noise=gamma
python pack_got.py transt_control default transt_control_gamma
rm -rf pytracking/tracking_results/transt_control


# CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt_control  #  --threads 10
# python pack_got.py transt_control default transt_control


# CUDA_VISIBLE_DEVICES=2 python pytracking/run_experiment.py myexperiments rtranst  # --threads 2
# python pack_got.py rtranst default rtranst

