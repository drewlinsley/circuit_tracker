
python -u pysot_toolkit/test.py --dataset GOT-10k --name 'transt_circuit_encoder' --ckpt checkpoints/ltr/transt_circuit_encoder/transt_circuit_encoder/TransT_ep0030.pth.tar  # BEST YET
# python -u pysot_toolkit/test.py --dataset GOT-10k --name 'transt_circuit_encoder' --ckpt checkpoints/ltr/transt_circuit_encoder/transt_circuit_encoder/TransT_ep0157.pth.tar
python pack_got_pysot.py transt_readout_test_encoder_mult default results/GOT-10k/transt_circuit_encoder/ pysot_circuit

