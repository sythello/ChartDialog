import json
import numpy as np
import os
from tqdm import tqdm
from collections import OrderedDict

import plotter

def serialize_split_1(params, pad_none=False):
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

def deserialize_split_1(params_str):
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
		print('deserialize_split_1 MALFORM:', params_str)
	return params

def serialize_single_1(params, pad_none=False):
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

def deserialize_single_1(params_str):
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
		print('deserialize_single_1 MALFORM:', params_str)
	return params

def serialize_pair_1(params, pad_none=False):
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

def deserialize_pair_1(params_str):
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
		print('deserialize_pair_1 MALFORM:', params_str)
	return params


if __name__ == '__main__':
	# Sanity check
	serializers = [serialize_split_1, serialize_single_1, serialize_pair_1]
	deserializers = [deserialize_split_1, deserialize_single_1, deserialize_pair_1]

	for serializer, deserializer in zip(serializers, deserializers):
		for i in range(100):
			data, params_dict = plotter.sample_one_hard()
			_s = serializer(params_dict)
			_d = deserializer(_s)
			_s2 = serializer(_d)

			# assert params_dict == _d, json.dumps(params_dict, indent=4) + '\n\n' + json.dumps(_d, indent=4)
			assert _s == _s2, _s + '\n' + _s2

	print('Sanity check passed.')


