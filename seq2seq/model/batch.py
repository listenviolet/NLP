import tensorflow as tf 

MAX_LEN = 50 # max length of a sentence
SOS_ID = 1   # <sos> id

# Get sentence length to feed the dynamic_rnn input
def MakeDataset(file_path):
	dataset = tf.data.TextLineDataset(file_path)

	# split by space
	dataset = dataset.map(lambda string: tf.string_split([string]).values)
	# convert to tf.int32 format
	dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
	# (sentence, words_num_in_sentence) into Dataset
	dataset = dataset.map(lambda x: (x, tf.size(x)))
	return dataset

def MakeSrcTrgDataset(src_path, trg_path, batch_size):
	''' get the source and target data
		combine to one dataset: (src_data, trg_data)
		dataset[0][0]: source sentence
		dataset[0][1]: source sentence length
		dataset[1][0]: target sentence
		dataset[1][1]: target sentence length
	'''
	src_data = MakeDataset(src_path)
	trg_data = MakeDataset(trg_path)
	dataset = tf.data.Dataset.zip((src_data, trg_data))

	# delete the sentences which only contrain <eos>: length > 1
	# delete the sentences exceeds the length limit: length <= MAX_LEN
	def FilterLength(src_tuple, trg_tuple):
		((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
		src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
		trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
		return tf.logical_and(src_len_ok, trg_len_ok)

	dataset = dataset.filter(FilterLength)

	''' Decoder needs two format trg sentence:
	    trg_input: <sos> X Y Z
	    trg_label:      X Y Z <eos>
	    the format of the sentences in files is: X Y Z <eos>
	    we need to generate: <sos> X Y Z
	    and add to the dataset
	'''

	def MakeTrgInput(src_tuple, trg_tuple):
		((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
		trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis = 0)
		return ((src_input, src_len), (trg_input, trg_label, trg_len))

	dataset = dataset.map(MakeTrgInput)

	# shuffle the dataset
	dataset = dataset.shuffle(10000)

	# def the dims of the output after padding
	padded_shapes = (
		(tf.TensorShape([None]),
		tf.TensorShape([])),
		(tf.TensorShape([None]),
		tf.TensorShape([None]),
		tf.TensorShape([])))
	batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
	return batched_dataset

