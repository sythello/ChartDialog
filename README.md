# ChartDialog

### Dataset download
https://drive.google.com/file/d/1B8tKdmI3LlAMsPQQW6KsCbTKfsm9vnFA/view?usp=sharing

### How to run the baseline (seq2seq)
#### Dialog preprocessing
This step transforms dialogs into seq2seq-style files (input, gold output, etc.)
```
bash dialog_preprocess.sh [YOUR PATH]/ChartDialog-data [YOUR PATH]/ChartDialog-save
```
\[YOUR PATH\] is the directory under which you put the ChartDialog-data folder.

#### Model-specific preprocessing
```
cd models/seq2seq
bash preprocess.sh [YOUR PATH]/ChartDialog-save/s2s_single
```
The granularity ``single`` is used in this example; it can also be ``pair`` or ``split``. For more information about the granularity please refer to our paper.

#### Training
```
bash train.sh [YOUR PATH]/ChartDialog-save/s2s_single [YOUR PATH]/ChartDialog-save/save_models s2s_single.1
```
The parameter setting in ``train.sh`` was used in our experiments in the paper.

#### Translating (Generating output)
```
bash translate.sh [YOUR PATH]/ChartDialog-save/s2s_single [YOUR PATH]/ChartDialog-save/save_models s2s_single.1 100000
```

#### Evaluating
```
cd ../..
bash evaluate.sh [YOUR PATH]/ChartDialog-save/s2s_single ./models/seq2seq/outputs/pred_test_s2s_single.1_100000.txt single
```

### Pretrained model download
- seq2seq: https://drive.google.com/open?id=15Pv1IeAas0L1PAqCaDPOqiBvKhFonGft

