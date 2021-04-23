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

# CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments transt_readout_test_v1  #  --threads 10
# python pack_got.py transt_readout_test_v1 default transt_readout_test_v1

CUDA_VISIBLE_DEVICES=0 python pytracking/run_experiment.py myexperiments transt_encoder  #  --threads 10
python pack_got.py transt_encoder default transt_encoder

# CUDA_VISIBLE_DEVICES=0,1 python pytracking/run_experiment.py myexperiments transt --threads 10
# python pack_got.py transt default transt


