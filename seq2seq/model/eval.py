import tensorflow as tf 

CHECKPOINT_PATH = "../log/-8800"

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
SHARE_EMB_SOFTMAX = True

SOS_ID = 1
EOS_ID = 2

class NMTModel(object):
	"""docstring for NMTModel"""
	# Same as in train.py
	def __init__(self):

		''' Define the encoder and decoder 
		'''
		self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
			[tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
		self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
			[tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

		''' Define word embeddings of the source and target language
		'''
		self.src_embedding = tf.get_variable('src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
		self.trg_embedding = tf.get_variable('trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE])

		''' Define the softmax layer vars
		'''
		if SHARE_EMB_SOFTMAX:
			self.softmax_weight = tf.transpose(self.trg_embedding)
		else:
			self.softmax_weight = tf.get_variable("weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
		self.softmax_bias = tf.get_variable("softmax_bias", [TRG_VOCAB_SIZE])

	def inference(self, src_input):
		# batch_size = 1
		src_size = tf.convert_to_tensor([len(src_input)], dtype = tf.int32)
		src_input = tf.convert_to_tensor([src_input], dtype = tf.int32)
		src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

		with tf.variable_scope("encoder"):
			enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb, src_size, dtype = tf.float32)

		# Setting the max steps
		MAX_DEC_LEN = 100

		with tf.variable_scope("decoder/rnn/multi_rnn_cell"):
			# use a TensorArray to store the generated sentence - variable length
			init_array = tf.TensorArray(dtype = tf.int32, size = 0, dynamic_size = True, clear_after_read = False)

			# make <sos> the first input of the decoder
			init_array = init_array.write(0, SOS_ID)

			# construct the initial state of while_loop:
			# 0: the step
			init_loop_var = (enc_state, init_array, 0)

			''' loop condition
				output <eos>
				reach MAX_DEC_LEN limit
			''' 
			def continue_loop_condition(state, trg_ids, step):
				# tf.reduce_all():
				# compute the "logical_and" of elements across dimentions of a tensor
				return tf.reduce_all(tf.logical_and(tf.not_equal(trg_ids.read(step), EOS_ID), tf.less(step, MAX_DEC_LEN - 1)))

			''' Loop body
			''' 
			def loop_body(state, trg_ids, step):
				# read the last step word and embedding
				trg_input = [trg_ids.read(step)]
				trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

				dec_outputs, next_state = self.dec_cell.call(inputs = trg_emb, state = state)

				output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
				logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)
				next_id = tf.argmax(logits, axis = 1, output_type = tf.int32)
				# write into the next trg_ids
				trg_ids = trg_ids.write(step + 1, next_id[0])
				return next_state, trg_ids, step + 1

			''' Execute while_loop
			'''
			state, trg_ids, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)

			return trg_ids.stack()

def main():
	with tf.variable_scope("nmt_model", reuse = None):
		model = NMTModel()
	test_sentence = [90, 13, 9, 689, 4, 2]
	output_op = model.inference(test_sentence)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, CHECKPOINT_PATH)

	# Read the translation result
	output = sess.run(output_op)
	print(output)
	sess.close()

if __name__ == "__main__":
	main()

# Input:
# [90, 13, 9, 689, 4, 2]
# This is  a  patients . <eos>
# Output:
# [  1  10   7   9  12 239  16   6   2]
# <sos> 这   是  一  个  病   人   。 <eos>