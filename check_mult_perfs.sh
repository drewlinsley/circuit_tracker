
i=113
GPU=0
for ckpt in `ls -t checkpoints/ltr/transt_circuit_encoder/transt_circuit_encoder/*.tar`
do
  echo "$ckpt"
  cp $ckpt pytracking/networks/transt_mult.pth
  rm -rf pytracking/tracking_results/transt_readout_test_encoder_mult
  CUDA_VISIBLE_DEVICES=$GPU python pytracking/run_experiment.py myexperiments transt_readout_test_encoder_mult
  python pack_got.py transt_readout_test_encoder_mult default train_model_encoder_mult_base_$i
  ((i--))
done

