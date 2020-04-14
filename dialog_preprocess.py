import pickle, os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import json
import numpy as np
import argparse
from collections import OrderedDict, defaultdict

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import plotter
from params_serialize import serialize_split_1, deserialize_split_1, \
	serialize_single_1, deserialize_single_1, serialize_pair_1, deserialize_pair_1

'''
This script preprocesses a dialog into (user_utterance, current_plot, gold_output).

user_utterance -> 'data-prep/{model_type}-{gran}/src.{subset}.txt', e.g. ChartDialog-data-prep/data-prep/s2s-single/src.train.txt

gold_output -> 'data-prep/{model_type}-{gran}/src.{subset}.txt', e.g. ChartDialog-data-prep/data-prep/s2s-single/tgt.train.txt

current_plot -> 'data-prep/img'

'''

def get_params_diff(params1, params2, slots=plotter.SLOTS_NAT_ORDER):
	# Based on SLOTS_NAT_ORDER
	# params1: base/current, params2: new
	# Return: dict(key, new_value)

	params_diff = {}
	for k in slots:
		if k not in params2:
			continue
		v = params2[k]
		if k not in params1 or params1[k] != v:
			params_diff[k] = v
	return params_diff

def get_params_diff_tuple(params1, params2, slots=plotter.SLOTS_NAT_ORDER):
	# Based on SLOTS_NAT_ORDER
	# params1: base/current, params2: new
	# Return: [(key, old_value if exists, new_value)]

	params_diff = []
	for k in slots:
		if k not in params2:
			continue
			
		v = params2[k]
		if k not in params1:
			params_diff.append((k, v))
		elif params1[k] != v:
			params_diff.append((k, params1[k], v))

	return params_diff

def preprocess_clf_1(list_fname, in_data_dir, out_data_dir, set_type, serializer, \
	save_img=False, plots_data_dir=None, img_out_data_dir=None, img_list_fname=None, \
	label_pools=None, \
	use_unchanged=True, \
	no_state=False, \
	no_utterance=False, \
	save_history=False):
	# Preprocessing for clf-1.x; img processing included.

	fname_list = open(list_fname, 'r').read().strip().split('\n')

	img_list = []
	img_id = 0
	data_pairs = []

	for fname in tqdm(fname_list):
		data = json.load(open(os.path.join(in_data_dir, fname), 'r'))
		if save_img:
			example_id = data['example_id']
			# print(example_id)
			example_data = pickle.load(open(os.path.join(plots_data_dir, 'data-series', '{}.dataseries.pkl'.format(example_id)), 'rb'))
		curr_params = dict(plotter.DEFAULT_DICT)

		desc_words = []
		history = []
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]
			
			curr_text = serializer(curr_params).lower()
			
			desc_words += word_tokenize(desc_act['text'].lower())
			history.append(desc_act['text'])

			if 'plotting' in op_act and op_act['plotting']:
				if save_img:
					# Generate the current plot img; not using legend since most types do not support
					img_id += 1
					img_fname = 'src.{}.{}.png'.format(img_id, set_type)

					call_params = plotter.plotter_kwargs_unnaturalize(**curr_params)

					fig = plotter.plotter(**example_data, **call_params)

					fig_buffer = BytesIO()
					fig.savefig(fig_buffer, dpi=50)
					fig_img = Image.open(fig_buffer)
					fig_img = fig_img.convert(mode='RGB')
					fig_img.save(os.path.join(img_out_data_dir, img_fname), format='PNG')
					fig_buffer.close()
					plt.close(fig)

					# legend_buffer = BytesIO()
					# fig_legend.savefig(legend_buffer, dpi=100)
					# legend_img = Image.open(legend_buffer)
					# plt.close(fig_legend)

					# merge_img = Image.new(mode=fig_img.mode, size=(fig_img.width, fig_img.height + legend_img.height))
					# merge_img.paste(fig_img)
					# merge_img.paste(legend_img, box=(0, fig_img.height))
					# merge_img = merge_img.convert(mode='RGB')
					# merge_img.save(os.path.join(img_out_data_dir, img_fname), format='PNG')

					# fig_img = fig_img.convert(mode='RGB')
					# fig_img.save(os.path.join(img_out_data_dir, img_fname), format='PNG')
					# fig_buffer.close()
					# legend_buffer.close()

					# fig.savefig(os.path.join(img_out_data_dir, img_fname), dpi=50)
					# plt.close(fig)
					img_list.append(img_fname)

				new_params = op_act['plotting_params']
				params_diff = get_params_diff(curr_params, new_params)
				curr_params = dict(plotter.DEFAULT_DICT)
				curr_params.update(new_params)

				desc_text = ' '.join(desc_words)

				if no_state:
					assert not no_utterance, 'Must have state or utterance'
					src_text = desc_text
				elif no_utterance:
					src_text = curr_text
				else:
					src_text = curr_text + ' || ' + desc_text
				data_pairs.append((src_text, new_params, params_diff, list(history)))
				desc_words = []
			else:
				# Concatenate all desc/op utterances until a plotting action is made
				desc_words += word_tokenize(op_act['text'].lower())
			history.append(op_act['text'])


	# Collect labels
	if label_pools is None:
		assert set_type == 'all'	# Only collect labels for all
		label_pools = [set() for _ in range(len(plotter.SLOTS_NAT_ORDER))]
		for _, new_params, params_diff, _ in data_pairs:
			for slot_id in range(len(plotter.SLOTS_NAT_ORDER)):
				slot_name = plotter.SLOTS_NAT_ORDER[slot_id]
				if slot_name in new_params:
					label_pools[slot_id].add(str(new_params[slot_name]).lower().strip())

		if use_unchanged:
			label_pools = [['[unchanged]'] + list(s) for s in label_pools]
		else:
			label_pools = [list(s) for s in label_pools]
		out_label_fname = os.path.join(out_data_dir, 'label.txt')
		json.dump(label_pools, open(out_label_fname, 'w'), indent=4)

	# Build dataset
	clf_data_pairs = []		# elem = (text, (label1, label2, ..., label_L), history)
	for text, new_params, params_diff, history in data_pairs:
		labels = []
		for slot_id in range(len(plotter.SLOTS_NAT_ORDER)):
			slot_name = plotter.SLOTS_NAT_ORDER[slot_id]
			if use_unchanged:
				l = params_diff[slot_name] if slot_name in params_diff else '[unchanged]'
			else:
				l = new_params[slot_name] if slot_name in new_params else None
			# labels.append(label_pools[slot_id].index(l))
			labels.append(l)

		labels_str = '\t'.join([str(l).lower().strip() for l in labels])
		history_str = ' | '.join(history)
		clf_data_pairs.append((text, labels_str, history_str))

	out_src_fname = os.path.join(out_data_dir, 'src.{}'.format(set_type + '.txt'))
	out_tgt_fname = os.path.join(out_data_dir, 'tgt.{}'.format(set_type + '.txt'))
	out_history_fname = os.path.join(out_data_dir, 'history.{}'.format(set_type + '.txt'))

	out_src_f = open(out_src_fname, 'w')
	out_tgt_f = open(out_tgt_fname, 'w')
	for p in clf_data_pairs:
		out_src_f.write(p[0] + '\n')
		out_tgt_f.write(p[1] + '\n')
	out_src_f.close()
	out_tgt_f.close()

	if save_history:
		out_history_f = open(out_history_fname, 'w')
		for p in clf_data_pairs:
			out_history_f.write(p[2] + '\n')
		out_history_f.close()

	if save_img:
		with open(img_list_fname, 'w') as f:
			f.write('\n'.join(img_list))

	return label_pools


def preprocess_s2s_1(list_fname, in_data_dir, out_data_dir, set_type, serializer, \
	save_img=False, plots_data_dir=None, img_out_data_dir=None, img_list_fname=None, \
	use_unchanged=True, \
	no_state=False, \
	no_utterance=False, \
	save_history=False):
	# Preprocessing for s2s-1.x; img processing included.

	fname_list = open(list_fname, 'r').read().strip().split('\n')

	img_list = []
	img_id = 0
	data_pairs = []

	for fname in tqdm(fname_list):
		data = json.load(open(os.path.join(in_data_dir, fname), 'r'))
		if save_img:
			example_id = data['example_id']
			# print(example_id)
			example_data = pickle.load(open(os.path.join(plots_data_dir, 'data-series', '{}.dataseries.pkl'.format(example_id)), 'rb'))
		curr_params = dict(plotter.DEFAULT_DICT)

		desc_words = []
		history = []
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]
			
			curr_text = serializer(curr_params).lower()
			
			desc_words += word_tokenize(desc_act['text'].lower())
			history.append(desc_act['text'])

			if 'plotting' in op_act and op_act['plotting']:
				if save_img:
					# Generate the current plot img; not using legend since most types do not support
					img_id += 1
					img_fname = 'src.{}.{}.png'.format(img_id, set_type)

					call_params = plotter.plotter_kwargs_unnaturalize(**curr_params)

					fig = plotter.plotter(**example_data, **call_params)
					fig_buffer = BytesIO()
					fig.savefig(fig_buffer, dpi=50)
					fig_img = Image.open(fig_buffer)
					fig_img = fig_img.convert(mode='RGB')
					fig_img.save(os.path.join(img_out_data_dir, img_fname), format='PNG')
					fig_buffer.close()
					plt.close(fig)

					img_list.append(img_fname)

				new_params = op_act['plotting_params']
				params_diff = get_params_diff(curr_params, new_params)
				curr_params = dict(plotter.DEFAULT_DICT)
				curr_params.update(new_params)

				desc_text = ' '.join(desc_words)

				if no_state:
					assert not no_utterance, 'Must have state or utterance'
					src_text = desc_text
				elif no_utterance:
					src_text = curr_text
				else:
					src_text = curr_text + ' || ' + desc_text
				data_pairs.append((src_text, new_params, params_diff, list(history)))
				desc_words = []
			else:
				# Concatenate all desc/op utterances until a plotting action is made
				desc_words += word_tokenize(op_act['text'].lower())
			history.append(op_act['text'])


	# # Collect labels
	# if label_pools is None:
	# 	assert set_type == 'all'	# Only collect labels for all
	# 	label_pools = [set() for _ in range(len(plotter.SLOTS_NAT_ORDER))]
	# 	for _, new_params, params_diff, _ in data_pairs:
	# 		for slot_id in range(len(plotter.SLOTS_NAT_ORDER)):
	# 			slot_name = plotter.SLOTS_NAT_ORDER[slot_id]
	# 			if slot_name in new_params:
	# 				label_pools[slot_id].add(str(new_params[slot_name]).lower().strip())

	# 	if use_unchanged:
	# 		label_pools = [['[unchanged]'] + list(s) for s in label_pools]
	# 	else:
	# 		label_pools = [list(s) for s in label_pools]
	# 	out_label_fname = os.path.join(out_data_dir, 'label.txt')
	# 	json.dump(label_pools, open(out_label_fname, 'w'), indent=4)

	# Build dataset
	s2s_data_pairs = []
	for src_str, new_params, params_diff, history in data_pairs:
		tgt_str = serializer(params_diff, pad_none=False)
		history_str = ' | '.join(history)
		if len(tgt_str) == 0 and use_unchanged:
			tgt_str = '[unchanged]'
		s2s_data_pairs.append((src_str, tgt_str, history_str))

	out_src_fname = os.path.join(out_data_dir, 'src.{}'.format(set_type + '.txt'))
	out_tgt_fname = os.path.join(out_data_dir, 'tgt.{}'.format(set_type + '.txt'))
	out_history_fname = os.path.join(out_data_dir, 'history.{}'.format(set_type + '.txt'))

	out_src_f = open(out_src_fname, 'w')
	out_tgt_f = open(out_tgt_fname, 'w')
	for p in s2s_data_pairs:
		out_src_f.write(p[0] + '\n')
		out_tgt_f.write(p[1] + '\n')
	out_src_f.close()
	out_tgt_f.close()

	if save_history:
		out_history_f = open(out_history_fname, 'w')
		for p in s2s_data_pairs:
			out_history_f.write(p[2] + '\n')
		out_history_f.close()

	if save_img:
		with open(img_list_fname, 'w') as f:
			f.write('\n'.join(img_list))


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('--input_dir', dest='input_dir', type=str,
		help='the directory "ChartDialog-data"')
	parser.add_argument('--preprocess_data_dir', dest='preprocess_data_dir', type=str,
		help='the directory to save preprocessed data')
	parser.add_argument('--no_img', dest='no_img', action='store_true',
		help='skips the image preprocessing. Set if you already have the images preprocessed in any setting.')
	parser.add_argument('--no_state', dest='no_state', action='store_true',
		help='remove current state from model input (only user utterances)')
	parser.add_argument('--no_utterance', dest='no_utterance', action='store_true',
		help='remove user utterances from model input (only current state)')
	parser.add_argument('--model_type', dest='model_type', type=str, choices=['clf', 's2s'], default='s2s',
		help='clf for classification models (MaxEnt, BERT, RNN-MLP); s2s for seq2seq models')
	parser.add_argument('--gran', dest='gran', type=str, choices=['pair', 'single', 'split'], default='single',
		help='granularity of text-plot-specifications')
	args = parser.parse_args()

	in_data_dir = os.path.join(args.input_dir, 'dialogs')				# Successful dialogs
	index_dir = os.path.join(args.input_dir, 'dialogs-split-index')		# Index files
	plots_data_dir = os.path.join(args.input_dir, 'plots-data')

	dataset_names = ['train', 'dev', 'test']

	img_out_data_dir = os.path.join(args.preprocess_data_dir, 'imgs')
	os.makedirs(img_out_data_dir, exist_ok=True)

	out_data_dir = os.path.join(args.preprocess_data_dir, '{}_{}'.format(args.model_type, args.gran))
	os.makedirs(out_data_dir, exist_ok=True)

	SERIALIZER_DICT = {
		'pair': serialize_pair_1,
		'single': serialize_single_1,
		'split': serialize_split_1
	}

	serializer = SERIALIZER_DICT[args.gran]
	save_img = (not args.no_img)
	no_state = args.no_state
	no_utterance = args.no_utterance
	use_unchanged = True 	# Always use unchanged
	save_history = False 	# Only used for agreement HITs

	if args.model_type == 'clf':
		# Collect label pools for each slot
		label_pools = preprocess_clf_1(list_fname=os.path.join(index_dir, 'dialog_file_list.txt'),
			in_data_dir=in_data_dir, out_data_dir=out_data_dir, set_type='all', serializer=serializer,
			save_img=False,		# Never save img in this step
			use_unchanged=use_unchanged,
			no_state=no_state,
			no_utterance=no_utterance,
			save_history=save_history)
		label_pools = json.load(open(os.path.join(out_data_dir, 'label.txt'), 'r'))

		for dn in dataset_names:
			preprocess_clf_1(list_fname=os.path.join(index_dir, 'dialog_file_list.{}.txt'.format(dn)),
				in_data_dir=in_data_dir, out_data_dir=out_data_dir, set_type=dn, serializer=serializer,
				save_img=save_img,
				plots_data_dir=plots_data_dir, img_out_data_dir=img_out_data_dir, img_list_fname=os.path.join(img_out_data_dir, 'src.{}.img_list.txt'.format(dn)),
				label_pools=label_pools,
				use_unchanged=use_unchanged,
				no_state=no_state,
				no_utterance=no_utterance,
				save_history=save_history)
	elif args.model_type == 's2s':
		for dn in dataset_names:
			preprocess_s2s_1(list_fname=os.path.join(index_dir, 'dialog_file_list.{}.txt'.format(dn)),
				in_data_dir=in_data_dir, out_data_dir=out_data_dir, set_type=dn, serializer=serializer,
				save_img=save_img,
				plots_data_dir=plots_data_dir, img_out_data_dir=img_out_data_dir, img_list_fname=os.path.join(img_out_data_dir, 'src.{}.img_list.txt'.format(dn)),
				use_unchanged=True,		# Always use unchanged for s2s to get rid of empty lines
				no_state=no_state,
				no_utterance=no_utterance,
				save_history=save_history)

if __name__ == '__main__':
	main()




