# This is seq2seq-text

DATA_DIR=$1
MODEL_DIR=$2
VERSION=$3
STEP=$4
echo ``DATA_DIR=${DATA_DIR}``
echo ``MODEL_DIR=${MODEL_DIR}``
echo ``VERSION=${VERSION}    STEP=${STEP}``

# e.g.
# DATA_DIR=[YOUR PATH]/ChartDialog-save/s2s_single
# MODEL_DIR=[YOUR PATH]/ChartDialog-save/save_models
# VERSION=s2s_single.1
# STEP=100000

mkdir -p outputs

python translate.py -model $MODEL_DIR/$VERSION/model.${VERSION}_step_${STEP}.pt \
-src $DATA_DIR/src.test.txt \
-tgt $DATA_DIR/tgt.test.txt \
-output outputs/pred_test_${VERSION}_${STEP}.txt \
-gpu 0

# Remove the '-gpu 0' if running on CPU
