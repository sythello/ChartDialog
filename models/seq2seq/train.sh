# This is seq2seq-text

DATA_DIR=$1
MODEL_DIR=$2
VERSION=$3
echo ``DATA_DIR=${DATA_DIR}``
echo ``MODEL_DIR=${MODEL_DIR}``
echo ``VERSION=${VERSION}``

# e.g.
# DATA_DIR=[YOUR PATH]/ChartDialog-save/s2s_single
# MODEL_DIR=[YOUR PATH]/ChartDialog-save/save_models
# VERSION=s2s_single.1

mkdir logs

python train.py \
-data $DATA_DIR/data_text \
-save_model $MODEL_DIR/${VERSION}/model.${VERSION} \
-save_checkpoint_steps 5000 \
-encoder_type brnn -layers 2 -rnn_size 128 -batch_size 16 -train_steps 100000 -valid_steps 5000 \
-log_file logs/log-${VERSION}.txt \
-world_size 1 -gpu_ranks 0

# Remove '-world_size 1 -gpu_ranks 0' if training on CPU

