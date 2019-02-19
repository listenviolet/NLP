# coding: utf-8
import tensorflow as tf 
from batch import MakeDataset, MakeSrcTrgDataset

SRC_TRAIN_DATA = '../data/train.en'
TRG_TRAIN_DATA = '../data/train.zh'

CHECKPOINT_PATH = '../attention_log/atten'

HIDDEN_SIZE = 1024
NUM_LAYERS = 2
SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 4000
BATCH_SIZE = 100
NUM_EPOCH = 5
KEEP_PROB = 0.8
MAX_GRAD_NORM = 5
SHARE_EMB_SOFTMAX = True

class NMTModel(object):
	"""docstring for NMTModel"""
	def __init__(self):

		''' Define the encoder and decoder 
			bidirectional_dynamic_rnn
		'''
		self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
		self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
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

	''' Construct the forward compute graph
	'''
	def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
		batch_size = tf.shape(src_input)[0]

		''' Get source and target input word embeddings
		'''
		# convert the src_input and trg_input to embeddings
		src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
		trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

		# apply dropout on word embeddings
		src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
		trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

		''' Encoder - use dynamic_rnn
			enc_outputs: [batch_size, max_time, HIDDEN_SIZE]
			enc_state: a tuple contains #NUM_LAYERS LSTMStateTuple classes. dims: [batch_size, state_size]
		'''
		with tf.variable_scope("encoder"):
			enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size, dtype = tf.float32)
			enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)
		''' Decoder - use dynamic_rnn
			dec_outputs: [batch_size, max_time, HIDDEN_SIZE]
		'''	
		with tf.variable_scope("decoder"):
			''' Attention mechanism
				memory_sequence_length: Tensor: [batch_size]
				representing every sentence length in batch
				according to this, attention mechanism set 0 in padding postions
			''' 
			attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_SIZE, enc_outputs, memory_sequence_length = src_size)

			# Wrap with decoder
			attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, attention_mechanism, attention_layer_size = HIDDEN_SIZE)

			# Use attention_cell and dynamic_rnn construct decoder
			# not need to set init_state => get from attention
			# former without attention: 
			# dec_outputs, _ = tf.nn.dynamic_rnn(self.dec_cell, trg_emb, trg_size, initial_state = enc_state)
			dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell, trg_emb, trg_size, dtype = tf.float32)

		''' Compute log perplexity each time step
		''' 
		output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
		logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(trg_label, [-1]), logits = logits)

		# When compute the avg cost
		# set the padding position weight be 0
		label_weights = tf.sequence_mask(trg_size, maxlen = tf.shape(trg_label)[1], dtype = tf.float32)
		label_weights = tf.reshape(label_weights, [-1])
		cost = tf.reduce_sum(loss * label_weights)
		cost_per_token = cost / tf.reduce_sum(label_weights)

		''' Define BP
		'''
		trainable_variables = tf.trainable_variables()
		# control grads; define opt; define train_op
		grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
		grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1.0)
		train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
		return cost_per_token, train_op

''' Train an epoch and return global steps
	save a checkpoint every 200 steps
''' 
def run_epoch(session, cost_op, train_op, saver, step):
	while True:
		try:
			cost, _ = session.run([cost_op, train_op])
			if step % 10 == 0:
				print("After %d steps, per token cost is %.3f" % (step, cost))
			if step % 200 == 0:
				saver.save(session, CHECKPOINT_PATH, global_step = step)
			step += 1
		except tf.errors.OutOfRangeError:
			break
	return step

def main():
	initializer = tf.random_uniform_initializer(-0.05, 0.05)

	''' Define the training model
	''' 
	with tf.variable_scope("nmt_model", reuse = None, initializer = initializer):
		train_model = NMTModel()
	''' Define the input data
	''' 
	data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
	iterator = data.make_initializable_iterator()
	(src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

	# Define forward compute graph
	cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)

	# Train model
	saver = tf.train.Saver()
	step = 0
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		for i in range(NUM_EPOCH):
			print("In iteration: %d" % (i + 1))
			sess.run(iterator.initializer)
			step = run_epoch(sess, cost_op, train_op, saver, step)

if __name__ == "__main__":
	main()
