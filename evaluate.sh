# This is seq2seq-text

DATA_DIR=$1
OUTPUT_FILE=$2
GRAN=$3
echo ``DATA_DIR: ${DATA_DIR}``
echo ``OUTPUT_FILE: ${OUTPUT_FILE}``
echo ``Granularity: ${GRAN}``

# e.g. 
# DATA_DIR=[YOUR PATH]/ChartDialog-save/s2s_single
# OUTPUT_FILE=./models/seq2seq/outputs/pred_test_s2s_single.1_100000.txt
# GRAN=single

python evaluate.py --src ${DATA_DIR}/src.test.txt \
--tgt ${DATA_DIR}/tgt.test.txt \
--out ${OUTPUT_FILE} \
--gran ${GRAN}
