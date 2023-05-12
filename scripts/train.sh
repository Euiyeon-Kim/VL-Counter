NUM_GPUS=1
CHECKPOINT_DIR=CLIPCorrCNN/openCLIP_allNorm_CNNFeat_noReLU
CONFIG_NAME=debug.yaml

mkdir -p exps/${CHECKPOINT_DIR} && \
python -m torch.distributed.launch --use_env \
--nproc_per_node=${NUM_GPUS} \
train.py \
--exp_name ${CHECKPOINT_DIR} \
--config configs/${CONFIG_NAME} \
2>&1 | tee -a exps/${CHECKPOINT_DIR}/train.log
