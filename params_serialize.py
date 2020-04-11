import json
import numpy as np
import os
from tqdm import tqdm
from collections import OrderedDict

import plotter_3_0 as plotter

def serialize_loose_1(params, pad_none=False):
	# params: dict
	param_strs_list = []
	for k in plotter.SLOTS_NAT_ORDER:
		if k in params:
			v = params[k]
		elif pad_none:
			v = None
		else:
			continue
		p_str = ' '.join(k.capitalize().split('_') + [':'] + str(v).split(' '))
		param_strs_list.append(p_str)
	# param_strs_list.sort()

	return ' | '.join(param_strs_list)

def deserialize_loose_1(params_str):
	# params_str: string
	if params_str == '[unchanged]':
		return OrderedDict()

	params_str = params_str.split(' || ')[0]	# Get rid of utterance, if any

	param_strs_list = params_str.split(' | ')
	params = OrderedDict()
	malform = False
	for _str in param_strs_list:
		try:
			k_str, v_str = _str.split(' : ')
			k = '_'.join(k_str.lower().split(' '))
			v = v_str
			params[k] = v
		except:
			malform = True

	if malform:
		print('deserialize_loose_1 MALFORM:', params_str)
	return params

def serialize_half_1(params, pad_none=False):
	# params: dict
	param_strs_list = []
	for k in plotter.SLOTS_NAT_ORDER:
		if k in params:
			v = params[k]
		elif pad_none:
			v = None
		else:
			continue
		p_str = k + ' ' + '_'.join(str(v).split(' '))
		param_strs_list.append(p_str)
	# param_strs_list.sort()

	return ' '.join(param_strs_list)

def deserialize_half_1(params_str):
	# params_str: string
	if params_str == '[unchanged]':
		return OrderedDict()

	params_str = params_str.split(' || ')[0]	# Get rid of utterance, if any

	param_strs_list = params_str.split(' ')
	params = OrderedDict()
	malform = False
	temp_k = None
	for i in range(len(param_strs_list)):
		if temp_k is None:
			k = param_strs_list[i]
			if k not in plotter.SLOTS_NAT_ORDER:
				malform = True
				continue
			temp_k = k
		else:
			v_str = param_strs_list[i]
			if v_str in plotter.SLOTS_NAT_ORDER:
				malform = True
				temp_k = v_str
				continue
			v = ' '.join(v_str.split('_'))
			params[temp_k] = v
			temp_k = None
	if temp_k is not None:
		malform = True

	if malform:
		print('deserialize_half_1 MALFORM:', params_str)
	return params

def serialize_dense_1(params, pad_none=False):
	# params: dict
	param_strs_list = []
	for k in plotter.SLOTS_NAT_ORDER:
		if k in params:
			v = params[k]
		elif pad_none:
			v = None
		else:
			continue
		p_str = k + ':' + '_'.join(str(v).split(' '))
		param_strs_list.append(p_str)
	# param_strs_list.sort()

	return ' '.join(param_strs_list)

def deserialize_dense_1(params_str):
	# params_str: string
	if params_str == '[unchanged]':
		return OrderedDict()

	params_str = params_str.split(' || ')[0]	# Get rid of utterance, if any

	param_strs_list = params_str.split(' ')
	params = OrderedDict()
	malform = False
	for _str in param_strs_list:
		try:
			k, v_str = _str.split(':')
			v = ' '.join(v_str.split('_'))
			params[k] = v
		except:
			malform = True

	if malform:
		print('deserialize_dense_1 MALFORM:', params_str)
	return params


def main_old():
	plots_folder = 'sample-plots/plots'
	params_folder = 'sample-plots/params'
	params_txt_folder = 'sample-plots/params-txt'
	root_folder = 'sample-plots/dataset'

	os.makedirs(params_txt_folder, exist_ok=True)
	os.makedirs(root_folder, exist_ok=True)

	n_files = len(list(filter(lambda x : x.endswith('params.json'), os.listdir(params_folder))))

	output_file_name_a = os.path.join(root_folder, 'params-txt-all.txt')
	with open(output_file_name_a, 'w') as out_f_a:
		for i in tqdm(range(n_files)):
			input_file_name = os.path.join(params_folder, '{}.params.json'.format(i + 1))
			params = json.load(open(input_file_name, 'r'))
			params_str = serialize(params)
			assert deserialize(params_str) == params, str(params) + '\n' + str(params_str) + '\n' + str(deserialize(params_str))

			out_f_a.write(params_str + '\n')
			output_file_name_s = os.path.join(params_txt_folder, '{}.params.txt'.format(i + 1))
			with open(output_file_name_s, 'w') as out_f:
				out_f.write(params_str + '\n')

if __name__ == '__main__':
	import dialog_preprocess as prep

	serializers = [serialize_loose_1, serialize_half_1, serialize_dense_1]
	deserializers = [deserialize_loose_1, deserialize_half_1, deserialize_dense_1]
	cmp_serializers = [prep.serialize_loose_1, prep.serialize_half_1, prep.serialize_dense_1]
	cmp_deserializers = [prep.deserialize_loose_1, prep.deserialize_half_1, prep.deserialize_dense_1]

	params_data_dir = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/ParlAI-data/plots-data/params'
	params_fname_list = list(filter(lambda x : x.endswith('params.json'), os.listdir(params_data_dir)))

	for serializer, deserializer, cmp_serializer, cmp_deserializer in zip(serializers, deserializers, cmp_serializers, cmp_deserializers):
		for fname in tqdm(params_fname_list):
			params_dict = json.load(open(os.path.join(params_data_dir, fname), 'r'))
			s = serializer(params_dict)
			s_cmp = cmp_serializer(params_dict)
			assert s == s_cmp, s + '\n' + s_cmp

			d = deserializer(s)
			d_cmp = cmp_deserializer(s)
			assert d == d_cmp, json.dumps(d, indent=4) + '\n' + json.dumps(d_cmp, indent=4)





