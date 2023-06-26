NUM_GPUS=1
CHECKPOINT_DIR=CLIPSCorrCNN/fixAll_normAll_baseline_b8_cutmix_CorrnewCNNFeat
CONFIG_NAME=CLIPSCorrCNN.yaml

mkdir -p exps/${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --use_env \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name ${CHECKPOINT_DIR} \
--config configs/${CONFIG_NAME} \
2>&1 | tee -a exps/${CHECKtPOINT_DIR}/train.log
