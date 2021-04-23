## Update kys/default with checkpoint location. Copy checkpoints to pytracking/networks
# 1. Copy checkpoint to pytracking/networks
# 2. Update pytracking/myexperiments.py
# 3. Create a model specific parameter in pytracking/parameters
# 4. Create a model specific tracker in pytracking/tracker

# python pytracking/run_experiment.py myexperiments got_retrained_bl  # BASELINE
# python pytracking/run_experiment.py myexperiments got_circuit_bl
# python pytracking/run_experiment.py myexperiments got_circuit_dual_bl
# python pytracking/run_experiment.py myexperiments got_circuit_dual_trans_bl
python pytracking/run_experiment.py myexperiments transt_readout

