input_dir=$1
preprocess_data_dir=$2

# e.g.
# input_dir=[YOUR PATH]/ChartDialog-data
# preprocess_data_dir=[YOUR PATH]/ChartDialog-save

echo ``python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type s2s --gran pair``
python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type s2s --gran pair

echo ``python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type s2s --gran single --no_img``
python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type s2s --gran single --no_img

echo ``python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type s2s --gran split --no_img``
python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type s2s --gran split --no_img

echo ``python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type clf --gran pair --no_img``
python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type clf --gran pair --no_img

echo ``python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type clf --gran single --no_img``
python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type clf --gran single --no_img

echo ``python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type clf --gran split --no_img``
python dialog_preprocess.py --input_dir ${input_dir} --preprocess_data_dir ${preprocess_data_dir} --model_type clf --gran split --no_img

