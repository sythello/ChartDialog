import pickle, os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import json
import numpy as np

from collections import OrderedDict, defaultdict

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

import plotter
from params_serialize import serialize_loose_1, deserialize_loose_1, \
	serialize_half_1, deserialize_half_1, serialize_dense_1, deserialize_dense_1


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


def preprocess_file_list_op_all(list_fname, in_data_dir, out_data_dir, fname_suffix):
	fname_list = open(list_fname, 'r').read().strip().split('\n')

	data_pairs = []

	for fname in tqdm(fname_list):
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))

		curr_params = dict(INIT_PARAMS)
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]
			
			curr_text = serialize(curr_params)
			desc_words = word_tokenize(desc_act['text'].lower())
			desc_text = ' '.join(desc_words)
			if op_act['plotting']:
				new_params = op_act['plotting_params']
				if new_params['label'] == '':
					new_params['label'] = None
				if new_params['markeredgewidth'] == 0:
					new_params['markeredgewidth'] = None
				params_diff = {k : new_params[k] for k in new_params if (k not in curr_params) or (curr_params[k] != new_params[k])}
				op_text = serialize(params_diff)
				if op_text == '':
					op_text = '[Unchanged]'
				curr_params = new_params
			else:
				op_words = word_tokenize(op_act['text'].lower())
				op_text = ' '.join(op_words)

			data_pairs.append((curr_text + ' |||| ' + desc_text, op_text))

	out_src_fname = os.path.join(out_data_dir, 'src.{}'.format(fname_suffix))
	out_tgt_fname = os.path.join(out_data_dir, 'tgt.{}'.format(fname_suffix))

	out_src_f = open(out_src_fname, 'w')
	out_tgt_f = open(out_tgt_fname, 'w')
	for p in data_pairs:
		out_src_f.write(p[0] + '\n')
		out_tgt_f.write(p[1] + '\n')
	out_src_f.close()
	out_tgt_f.close()

def preprocess_file_list_op_plot(list_fname, in_data_dir, out_data_dir, fname_suffix):
	fname_list = open(list_fname, 'r').read().strip().split('\n')

	data_pairs = []

	for fname in tqdm(fname_list):
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))

		curr_params = dict(INIT_PARAMS)
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]
			
			curr_text = serialize(curr_params)
			if op_act['plotting']:
				new_params = op_act['plotting_params']
				if new_params['label'] == '':
					new_params['label'] = None
				if new_params['markeredgewidth'] == 0:
					new_params['markeredgewidth'] = None
				params_diff = {k : new_params[k] for k in new_params if (k not in curr_params) or (curr_params[k] != new_params[k])}
				op_text = serialize(params_diff)
				if op_text == '':
					op_text = '[Unchanged]'
				curr_params = new_params
			else:
				# op_words = word_tokenize(op_act['text'].lower())
				# op_text = ' '.join(op_words)
				continue
			desc_words = word_tokenize(desc_act['text'].lower())
			desc_text = ' '.join(desc_words)

			data_pairs.append((curr_text + ' |||| ' + desc_text, op_text))

	out_src_fname = os.path.join(out_data_dir, 'src.{}'.format(fname_suffix))
	out_tgt_fname = os.path.join(out_data_dir, 'tgt.{}'.format(fname_suffix))

	out_src_f = open(out_src_fname, 'w')
	out_tgt_f = open(out_tgt_fname, 'w')
	for p in data_pairs:
		out_src_f.write(p[0] + '\n')
		out_tgt_f.write(p[1] + '\n')
	out_src_f.close()
	out_tgt_f.close()


def preprocess_file_list_op_plot_dense(list_fname, in_data_dir, out_data_dir, fname_suffix):
	fname_list = open(list_fname, 'r').read().strip().split('\n')

	data_pairs = []

	for fname in tqdm(fname_list):
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))

		curr_params = dict(INIT_PARAMS)
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]
			
			curr_text = serialize_dense(curr_params)
			if op_act['plotting']:
				new_params = op_act['plotting_params']
				if new_params['label'] == '':
					new_params['label'] = None
				if new_params['markeredgewidth'] == 0:
					new_params['markeredgewidth'] = None
				params_diff = {k : new_params[k] for k in new_params if (k not in curr_params) or (curr_params[k] != new_params[k])}
				op_text = serialize_dense(params_diff)
				if op_text == '':
					op_text = '[Unchanged]'
				curr_params = new_params
			else:
				# op_words = word_tokenize(op_act['text'].lower())
				# op_text = ' '.join(op_words)
				continue
			desc_words = word_tokenize(desc_act['text'].lower())
			desc_text = ' '.join(desc_words)

			data_pairs.append((curr_text + ' ' + desc_text, op_text))

	out_src_fname = os.path.join(out_data_dir, 'src.{}'.format(fname_suffix))
	out_tgt_fname = os.path.join(out_data_dir, 'tgt.{}'.format(fname_suffix))

	out_src_f = open(out_src_fname, 'w')
	out_tgt_f = open(out_tgt_fname, 'w')
	for p in data_pairs:
		out_src_f.write(p[0] + '\n')
		out_tgt_f.write(p[1] + '\n')
	out_src_f.close()
	out_tgt_f.close()


def preprocess_file_list_op_plot_dense_nostate(list_fname, in_data_dir, out_data_dir, fname_suffix):
	fname_list = open(list_fname, 'r').read().strip().split('\n')

	data_pairs = []

	for fname in tqdm(fname_list):
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))

		curr_params = dict(INIT_PARAMS)
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]
			
			# curr_text = serialize_dense(curr_params)
			if op_act['plotting']:
				new_params = op_act['plotting_params']
				if new_params['label'] == '':
					new_params['label'] = None
				if new_params['markeredgewidth'] == 0:
					new_params['markeredgewidth'] = None
				params_diff = {k : new_params[k] for k in new_params if (k not in curr_params) or (curr_params[k] != new_params[k])}
				op_text = serialize_dense(params_diff)
				if op_text == '':
					op_text = '[Unchanged]'
				curr_params = new_params
			else:
				# op_words = word_tokenize(op_act['text'].lower())
				# op_text = ' '.join(op_words)
				continue
			desc_words = word_tokenize(desc_act['text'].lower())
			desc_text = ' '.join(desc_words)

			data_pairs.append((desc_text, op_text))

	out_src_fname = os.path.join(out_data_dir, 'src.{}'.format(fname_suffix))
	out_tgt_fname = os.path.join(out_data_dir, 'tgt.{}'.format(fname_suffix))

	out_src_f = open(out_src_fname, 'w')
	out_tgt_f = open(out_tgt_fname, 'w')
	for p in data_pairs:
		out_src_f.write(p[0] + '\n')
		out_tgt_f.write(p[1] + '\n')
	out_src_f.close()
	out_tgt_f.close()


def preprocess_file_list_op_plot_dense_clf(list_fname, in_data_dir, out_data_dir, fname_suffix, label_pools=None, use_unchanged=True):
	fname_list = open(list_fname, 'r').read().strip().split('\n')

	data_pairs = []

	for fname in tqdm(fname_list):
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))

		curr_params = dict(INIT_PARAMS)
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]
			
			curr_text = serialize_dense(curr_params)
			if op_act['plotting']:
				new_params = op_act['plotting_params']
				if new_params['label'] == '':
					new_params['label'] = None
				if new_params['markeredgewidth'] == 0:
					new_params['markeredgewidth'] = None
				params_diff = {k : new_params[k] for k in new_params if (k not in curr_params) or (curr_params[k] != new_params[k])}
				# op_text = serialize_dense(params_diff)
				# if op_text == '':
				# 	op_text = '[Unchanged]'
				curr_params = new_params
			else:
				# op_words = word_tokenize(op_act['text'].lower())
				# op_text = ' '.join(op_words)
				continue
			desc_words = word_tokenize(desc_act['text'].lower())
			desc_text = ' '.join(desc_words)

			data_pairs.append((curr_text + ' ' + desc_text, new_params, params_diff))

	if label_pools is None:
		label_pools = [set() for _ in range(len(SLOTS))]
		for _, params, params_diff in data_pairs:
			for slot_id in range(len(SLOTS)):
				label_pools[slot_id].add(params[SLOTS[slot_id]])

		if use_unchanged:
			label_pools = [['[unchanged]'] + list(s) for s in label_pools]
		else:
			label_pools = [list(s) for s in label_pools]
		out_label_fname = os.path.join(out_data_dir, 'label.{}'.format(fname_suffix))
		json.dump(label_pools, open(out_label_fname, 'w'), indent=4)

	clf_data_pairs = []		# elem = (text, (label1, label2, ..., label10))
	for text, params, params_diff in data_pairs:
		labels = []
		for i in range(len(SLOTS)):
			slot_name = SLOTS[i]
			if use_unchanged:
				l = params_diff[slot_name] if slot_name in params_diff else '[unchanged]'
			else:
				l = params[slot_name]
			# labels.append(label_pools[i].index(l))
			labels.append(l)

		labels_str = '\t'.join([str(l) for l in labels])
		clf_data_pairs.append((text, labels_str))

	out_src_fname = os.path.join(out_data_dir, 'src.{}'.format(fname_suffix))
	out_tgt_fname = os.path.join(out_data_dir, 'tgt.{}'.format(fname_suffix))

	out_src_f = open(out_src_fname, 'w')
	out_tgt_f = open(out_tgt_fname, 'w')
	for p in clf_data_pairs:
		out_src_f.write(p[0] + '\n')
		out_tgt_f.write(p[1] + '\n')
	out_src_f.close()
	out_tgt_f.close()

	return label_pools


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
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))
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
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))
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



def preprocess_agreement_1(list_fname, in_data_dir, out_data_dir, set_type, serializer, n_per_type=50):
	# Preprocessing for agreement test (keep everything untokenized and cased)

	fname_list = open(list_fname, 'r').read().strip().split('\n')

	img_list = []
	img_id = 0
	data_pairs = []
	type_counts = defaultdict(int)

	for fname in tqdm(fname_list):
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))
		example_id = data['example_id']
		describer_id = data['Describer']
		operator_id = data['Operator']

		curr_params = dict(plotter.DEFAULT_DICT)

		# desc_texts = []
		history = []
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]
			
			curr_text = serializer(curr_params)
			
			# desc_texts.append(desc_act['text'])
			history.append(desc_act['text'])

			if 'plotting' in op_act and op_act['plotting']:
				new_params = op_act['plotting_params']
				params_diff = get_params_diff_tuple(curr_params, new_params)
				str_diff = '\n'
				for tp in params_diff:
					if len(tp) == 3:
						if tp[1] is not None:
							str_diff += '{}: {} &#8594; {}\n'.format(*tp)
						else:
							str_diff += '{}: {}\n'.format(tp[0], tp[2])
					else:
						str_diff += '{}: {}\n'.format(*tp)

				new_params_full_slots = dict(plotter.DEFAULT_DICT)
				new_params_full_slots.update(new_params)

				data_pairs.append((curr_params, new_params_full_slots, list(history), example_id, describer_id, operator_id, op_act))
				
				curr_params = new_params_full_slots

				history.append(op_act['text'] + str_diff)
			else:
				# Concatenate all desc/op utterances until a plotting action is made
				# desc_texts.append(op_act['text'])
				history.append(op_act['text'])

	# Build dataset
	# s2s_data_pairs = []
	for curr_params, new_params, history, example_id, describer_id, operator_id, op_act in data_pairs:
		tgt_plot_type = new_params['plot_type']
		src_plot_type = curr_params['plot_type']
		src_plot_type_v = plotter.PLOT_TYPE_S2V[src_plot_type] if src_plot_type in plotter.PLOT_TYPE_S2V else 'None'
		tgt_plot_type_v = plotter.PLOT_TYPE_S2V[tgt_plot_type] if tgt_plot_type in plotter.PLOT_TYPE_S2V else 'None'
		if tgt_plot_type_v == 'None':
			continue
		if (src_plot_type_v != 'None') and (src_plot_type_v != tgt_plot_type_v):
			continue
		plot_type = tgt_plot_type_v

		type_id = type_counts[plot_type] + 1
		if type_id > n_per_type:
			continue

		save_fname = os.path.join(out_data_dir, '{}.{}.json'.format(plot_type, type_id))
		save_dict = dict(curr_params)
		save_dict['example_id'] = example_id
		save_dict['utterances'] = history
		save_dict['original_describer_id'] = describer_id
		save_dict['original_operator_id'] = operator_id
		save_dict['original_op_act'] = op_act
		json.dump(save_dict, open(save_fname, 'w'), indent=4)
		type_counts[plot_type] += 1

	# out_src_fname = os.path.join(out_data_dir, 'src.{}'.format(set_type + '.txt'))
	# out_tgt_fname = os.path.join(out_data_dir, 'tgt.{}'.format(set_type + '.txt'))
	# out_history_fname = os.path.join(out_data_dir, 'history.{}'.format(set_type + '.txt'))

	# out_src_f = open(out_src_fname, 'w')
	# out_tgt_f = open(out_tgt_fname, 'w')
	# for p in s2s_data_pairs:
	# 	out_src_f.write(p[0] + '\n')
	# 	out_tgt_f.write(p[1] + '\n')
	# out_src_f.close()
	# out_tgt_f.close()

	# out_history_f = open(out_history_fname, 'w')
	# for p in s2s_data_pairs:
	# 	out_history_f.write(p[2] + '\n')
	# out_history_f.close()



def preprocess_aux_1(list_fname, in_data_dir, out_data_dir, set_type, serializer):
	# Auxiliary data: for each datapoint (each line in dataset), the dialog filename + example id

	fname_list = open(list_fname, 'r').read().strip().split('\n')

	img_list = []
	img_id = 0
	data_pairs = []

	for fname in tqdm(fname_list):
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))
		example_id = data['example_id']
		# print(example_id)
		# example_data = pickle.load(open(os.path.join(plots_data_dir, 'data-series', '{}.dataseries.pkl'.format(example_id)), 'rb'))
		# curr_params = dict(plotter.DEFAULT_DICT)

		# desc_words = []
		# history = []
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]

			if 'plotting' in op_act and op_act['plotting']:
				data_pairs.append((fname, example_id))

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
	# s2s_data_pairs = []
	# for src_str, new_params, params_diff, history in data_pairs:
	# 	tgt_str = serializer(params_diff, pad_none=False)
	# 	history_str = ' | '.join(history)
	# 	if len(tgt_str) == 0 and use_unchanged:
	# 		tgt_str = '[unchanged]'
	# 	s2s_data_pairs.append((src_str, tgt_str, history_str))

	out_fname = os.path.join(out_data_dir, 'aux.{}'.format(set_type + '.txt'))

	out_f = open(out_fname, 'w')
	for p in data_pairs:
		out_f.write('{}\t{}\n'.format(*p))
	out_f.close()


def preprocess_file_list_op_plot_images(list_fname, in_data_dir, plots_data_dir, img_out_data_dir, img_fname_suffix, img_list_fname):
	fname_list = open(list_fname, 'r').read().strip().split('\n')

	img_list = []
	img_id = 0

	for fname in tqdm(fname_list):
		data = pickle.load(open(os.path.join(in_data_dir, fname), 'rb'))
		example_id = data['example_id']
		x, y = np.loadtxt(os.path.join(plots_data_dir, 'data-series', '{}.dataseries.txt'.format(example_id)))

		curr_params = dict(INIT_PARAMS)
		for turn in data['data']:
			desc_act = turn['utterances'][0]
			op_act = turn['utterances'][1]
			
			if op_act['plotting']:
				img_id += 1
				img_fname = 'src.{}.{}'.format(img_id, img_fname_suffix)
				fig, fig_legend = plotter.plotter(x, y, save_legend=True, **curr_params)

				fig_buffer = BytesIO()
				fig.savefig(fig_buffer, dpi=50)
				fig_img = Image.open(fig_buffer)
				plt.close(fig)

				legend_buffer = BytesIO()
				fig_legend.savefig(legend_buffer, dpi=100)
				legend_img = Image.open(legend_buffer)
				plt.close(fig_legend)

				merge_img = Image.new(mode=fig_img.mode, size=(fig_img.width, fig_img.height + legend_img.height))
				merge_img.paste(fig_img)
				merge_img.paste(legend_img, box=(0, fig_img.height))
				merge_img = merge_img.convert(mode='RGB')
				merge_img.save(os.path.join(img_out_data_dir, img_fname), format='PNG')
				img_list.append(img_fname)
				fig_buffer.close()
				legend_buffer.close()

				new_params = op_act['plotting_params']
				if new_params['label'] == '':
					new_params['label'] = None
				if new_params['markeredgewidth'] == None:
					new_params['markeredgewidth'] = 0
				# params_diff = {k : new_params[k] for k in new_params if (k not in curr_params) or (curr_params[k] != new_params[k])}
				# op_text = serializer(params_diff)
				# if op_text == '':
				# 	op_text = '[Unchanged]'
				curr_params = new_params
			else:
				# op_words = word_tokenize(op_act['text'].lower())
				# op_text = ' '.join(op_words)
				continue
			# desc_words = word_tokenize(desc_act['text'].lower())
			# desc_text = ' '.join(desc_words)

			# data_pairs.append((curr_text + ' ' + desc_text, op_text))

	with open(img_list_fname, 'w') as f:
		f.write('\n'.join(img_list))

def main():
	in_data_dir = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/ParlAI-data/plot_desc_collected/3_0_all'			# Dialogs
	index_dir = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/Plotting-agent/data/dialog_data_3_0'				# Filename list files
	dataset_names = ['train', 'dev', 'test']

	plots_data_dir = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/ParlAI-data/plots-data'
	img_out_data_dir = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/Plotting-agent/data/dialog_data_3_0/imgs'
	os.makedirs(img_out_data_dir, exist_ok=True)

	out_data_dir = '/Users/mac/Desktop/syt/Deep-Learning/Projects-M/Plotting-agent/data/dialog_data_3_0/aux'
	os.makedirs(out_data_dir, exist_ok=True)

	serializer = serialize_dense_1
	save_img = False
	use_unchanged = True
	no_state = False
	no_utterance = False
	# src_cased = False
	save_history = False

	## Clf
	# label_pools = preprocess_clf_1(list_fname=os.path.join(index_dir, 'dialog_file_list.txt'),
	# 	in_data_dir=in_data_dir, out_data_dir=out_data_dir, set_type='all', serializer=serializer,
	# 	save_img=False,		# Never save img in this step
	# 	use_unchanged=use_unchanged,
	# 	no_state=no_state,
	# 	no_utterance=no_utterance,
	# 	save_history=save_history)
	# label_pools = json.load(open(os.path.join(out_data_dir, 'label.txt'), 'r'))
	# for dn in dataset_names:
	# 	preprocess_clf_1(list_fname=os.path.join(index_dir, 'dialog_file_list.{}.txt'.format(dn)),
	# 		in_data_dir=in_data_dir, out_data_dir=out_data_dir, set_type=dn, serializer=serializer,
	# 		save_img=save_img,
	# 		plots_data_dir=plots_data_dir, img_out_data_dir=img_out_data_dir, img_list_fname=os.path.join(img_out_data_dir, 'src.{}.img_list.txt'.format(dn)),
	# 		label_pools=label_pools,
	# 		use_unchanged=use_unchanged,
	# 		no_state=no_state,
	# 		no_utterance=no_utterance,
	# 		save_history=save_history)

	## S2S
	# for dn in dataset_names:
	# 	preprocess_s2s_1(list_fname=os.path.join(index_dir, 'dialog_file_list.{}.txt'.format(dn)),
	# 		in_data_dir=in_data_dir, out_data_dir=out_data_dir, set_type=dn, serializer=serializer,
	# 		save_img=save_img,
	# 		plots_data_dir=plots_data_dir, img_out_data_dir=img_out_data_dir, img_list_fname=os.path.join(img_out_data_dir, 'src.{}.img_list.txt'.format(dn)),
	# 		use_unchanged=True,		# Always use unchanged for s2s to get rid of empty lines
	# 		no_state=no_state,
	# 		no_utterance=no_utterance,
	# 		save_history=save_history)

	## Agreement
	# for dn in dataset_names + ['all']:
	# 	list_fname = os.path.join(index_dir, 'dialog_file_list.txt' if dn == 'all' else 'dialog_file_list.{}.txt'.format(dn))
	# 	preprocess_agreement_1(list_fname=list_fname, in_data_dir=in_data_dir, out_data_dir=out_data_dir, set_type=dn, serializer=serializer)
	# list_fname = os.path.join(index_dir, 'dialog_file_list.txt')
	# preprocess_agreement_1(list_fname=list_fname, in_data_dir=in_data_dir, out_data_dir=out_data_dir, set_type='all', serializer=serializer, n_per_type=50)

	## Auxiliary (example id)
	for dn in dataset_names + ['all']:
		list_fname = os.path.join(index_dir, 'dialog_file_list.txt' if dn == 'all' else 'dialog_file_list.{}.txt'.format(dn))
		preprocess_aux_1(list_fname=list_fname, in_data_dir=in_data_dir, out_data_dir=out_data_dir, set_type=dn, serializer=serializer)


if __name__ == '__main__':
	main()




