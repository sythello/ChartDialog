# This is seq2seq-text

DATA_DIR=$1
echo ``DATA_DIR: ${DATA_DIR}``	

# e.g. 
# DATA_DIR=[YOUR PATH]/ChartDialog-save/s2s_single

python preprocess.py \
-train_src ${DATA_DIR}/src.train.txt \
-train_tgt ${DATA_DIR}/tgt.train.txt \
-valid_src ${DATA_DIR}/src.dev.txt \
-valid_tgt ${DATA_DIR}/tgt.dev.txt \
-src_seq_length 384 \
-tgt_seq_length 192 \
-save_data ${DATA_DIR}/data_text
