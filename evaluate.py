import plotter
import params_serialize as serial
import argparse

SLOTS = list(plotter.SLOTS_NAT_ORDER)
DESERIAL_DICT = {
	'pair': serial.deserialize_pair_1,
	'single': serial.deserialize_single_1,
	'split': serial.deserialize_split_1
}

def S2S_evaluate(src_fname, tgt_fname, out_fname, gran):
	deserializer = DESERIAL_DICT[gran]

	inputs = open(src_fname, 'r').read().strip().split('\n')
	model_outputs = open(out_fname, 'r').read().strip().split('\n')
	gold_outputs = open(tgt_fname, 'r').read().strip().split('\n')
	# aux_lines = open('/Users/mac/Desktop/syt/Deep-Learning/Projects-M/Plotting-agent/data/dialog_data_3_0/aux/aux.test.txt', 'r').read().strip().split('\n')
	# len(inputs), len(model_outputs), len(gold_outputs), len(aux_lines)

	n_samples = len(gold_outputs)
	n_exact_match = sum([(model_outputs[i] == gold_outputs[i]) for i in range(n_samples)])
	exact_acc = float(n_exact_match) / n_samples

	# Result match 
	# Need to uniformize to string to compare (?)
	n_result_match = 0
	n_slot_match = 0
	n_change_match = 0
	n_gold_changes = 0
	n_out_changes = 0
	result_match_bools = []

	# inputs_dense_wstate = open('/Users/mac/Desktop/syt/Deep-Learning/Projects-M/Plotting-agent/data/dialog_data_3_0/s2s_1_2/src.test.txt', 'r').read().strip().split('\n')
	out_dicts = []
	tgt_dicts = []
	curr_dicts = []

	for i in range(n_samples):
	    src_line = inputs[i]
	    tgt_line = gold_outputs[i]
	    out_line = model_outputs[i]

	    if gran == 'split':
	    	curr_params_str = src_line.split(' || ')[0]
	    elif gran == 'single':
	    	curr_params_str = ' '.join(src_line.split(' ')[:2 * len(SLOTS)])
	    elif gran == 'pair':
	    	curr_params_str = ' '.join(src_line.split(' ')[:len(SLOTS)])
	    else:
	    	raise Exception('Unknown granularity')

	    curr_params_dict = deserializer(curr_params_str)
	    curr_dicts.append(curr_params_dict)
	#     for k, v in curr_params_dict.items():
	#         curr_params_dict[k] = str(v) 
	    
	    out_dict = dict(curr_params_dict)
	    out_change_dict = deserializer(out_line)
	    for k, v in out_change_dict.items():
	        out_dict[k] = v
	    out_dicts.append(out_dict)
	    
	    tgt_dict = dict(curr_params_dict)
	    gold_change_dict = deserializer(tgt_line)
	    for k, v in gold_change_dict.items():
	        tgt_dict[k] = v
	    tgt_dicts.append(tgt_dict)
	        
	    if out_dict == tgt_dict:
	        n_result_match += 1
	        result_match_bools.append(True)
	    else:
	        result_match_bools.append(False)
	    
	    for k in SLOTS:
	        if out_dict[k] == tgt_dict[k]:
	            n_slot_match += 1
	            if out_dict[k] != curr_params_dict[k]:
	                n_change_match += 1
	        if tgt_dict[k] != curr_params_dict[k]:
	            n_gold_changes += 1
	        if out_dict[k] != curr_params_dict[k]:
	            n_out_changes += 1
	        
	#     print(curr_params_dict)
	#     print(out_change_dict)
	#     print(out_dict)
	#     print(gold_change_dict)
	#     print(tgt_dict)
	#     print('-' * 100)

	result_acc = float(n_result_match) / n_samples
	slot_acc = float(n_slot_match) / (n_samples * len(SLOTS))
	change_prec = float(n_change_match) / n_out_changes
	change_recall = float(n_change_match) / n_gold_changes
	change_F1 = 2 * change_prec * change_recall / (change_prec + change_recall)

	print('Output line exact match: {}/{} = {:.4f}'.format(n_exact_match, n_samples, exact_acc))
	print()
	
	print('Output state exact match: {}/{} = {:.4f}'.format(n_result_match, n_samples, result_acc))
	print('Slot match: {}/{} = {:.4f}'.format(n_slot_match, n_samples * len(SLOTS), slot_acc))
	print('Changes precision: {}/{} = {:.4f}'.format(n_change_match, n_out_changes, change_prec))
	print('Changes recall: {}/{} = {:.4f}'.format(n_change_match, n_gold_changes, change_recall))
	print('Changes F1 = {:.4f}'.format(change_F1))




if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--src', dest='src', type=str,
		help='the source file')
	parser.add_argument('--tgt', dest='tgt', type=str,
		help='the gold output file')
	parser.add_argument('--out', dest='out', type=str,
		help='the output file from model')
	parser.add_argument('--gran', dest='gran', type=str, choices=['pair', 'single', 'split'],
		help='granularity')

	args = parser.parse_args()

	S2S_evaluate(args.src, args.tgt, args.out, args.gran)





