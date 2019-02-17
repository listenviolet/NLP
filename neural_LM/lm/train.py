import numpy as np 
import tensorflow as tf 
from convert_to_cnt import convert 
from batching import read_data, make_batches

RAW_TRAIN_DATA = '../data/ptb.train.txt'
RAW_EVAL_DATA = '../data/ptb.valid.txt'
RAW_TEST_DATA = '../data/ptb.test.txt'

TRIAN_DATA = '../data/ptb.train'
EVAL_DATA = '../data/ptb.valid'
TEST_DATA = '../data/ptb.test'


HIDDEN_SIZE = 300
NUM_LAYERS = 2
VOCAB_SIZE = 10000
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_SETP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1

NUM_EPOCH = 5
LSTM_KEEP_PROB = 0.9
EMBEDDING_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True

class PTBModel(object):
	def __init__(self, is_training, batch_size, num_steps):
		self.batch_size = batch_size
		self.num_steps = num_steps

		# def the input and targets. Both are [batch_size, num_steps]
		self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
		self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

		# def LSTM + dropout
		dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
		lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(
						tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
						output_keep_prob = dropout_keep_prob)
					for _ in range(NUM_LAYERS)]
		cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

		# def init state: 0
		self.initial_state = cell.zero_state(batch_size, tf.float32)


		# def embedding matrix
		embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
		# inputs to word_embeddings
		# inputs shape: [batch_size, num_steps, hidden_size]
		inputs = tf.nn.embedding_lookup(embedding, self.input_data)

		# if in training: apply dropout
		if is_training:
			inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

		# def outputs list.
		outputs = []
		state = self.initial_state
		with tf.variable_scope('RNN'):
			for time_step in range(num_steps):
				if time_step > 0: tf.get_variable_scope().reuse_variables()
				#cell_output shape: [batch_size, hidden_size]
				cell_output, state = cell(inputs[:, time_step, :], state)
				#output size: [num_steps, batch_size, hidden_size]
				outputs.append(cell_output)

		# outputs: [num_steps, batch_size, hidden_size]
		# tf.concat(outputs, 1): [batch_size, num_steps * hidden_size]
		# tf.reshape(..., [-1, HIDDEN_SIZE]): [batch_size * num_steps, hidden_size]
		output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])


		# Softmax layer, get logits
		if SHARE_EMB_AND_SOFTMAX:
			weight = tf.transpose(embedding)
		else:
			weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
		bias = tf.get_variable('bias', [VOCAB_SIZE])
		logits = tf.matmul(output, weight) + bias

		# def cross entropy loss and avg loss
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels = tf.reshape(self.targets, [-1]),
			logits = logits)
		self.cost = tf.reduce_sum(loss) / batch_size

		self.final_state = state 

		
		# Back propagation => only in training
		if not is_training: return 

		trainable_varibales = tf.trainable_variables()

		# contral grads
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_varibales), MAX_GRAD_NORM)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1.0)
		self.train_op = optimizer.apply_gradients(zip(grads, trainable_varibales))

def run_epoch(session, model, batches, train_op, output_log, step):
	# aux variables for computing perplexity
	total_costs = 0.0
	iters = 0
	state = session.run(model.initial_state)
	# train an epoch
	# every time train a batch
	for x, y in batches:
		cost, state, _ = session.run([model.cost, model.final_state, train_op],
						{model.input_data:x, model.targets: y, model.initial_state: state})
		total_costs += cost
		iters += model.num_steps

		# Only output log in training:
		if output_log and step % 100 == 0:
			print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))

		step += 1

	return step, np.exp(total_costs / iters)

def main():
	convert(RAW_TRAIN_DATA, TRIAN_DATA)
	convert(RAW_EVAL_DATA, EVAL_DATA)
	convert(RAW_TEST_DATA, TEST_DATA)
	initializer = tf.random_uniform_initializer(-0.05, 0.05)

	# def the nn model for training
	with tf.variable_scope('language_model', reuse = None, initializer = initializer):
		train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_SETP)

	with tf.variable_scope('language_model', reuse = True, initializer = initializer):
		eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

	with tf.Session() as session:
		tf.global_variables_initializer().run()
		train_batches = make_batches(read_data(TRIAN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_SETP)
		eval_batches = make_batches(read_data(EVAL_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)
		test_batches = make_batches(read_data(TEST_DATA), EVAL_BATCH_SIZE, EVAL_NUM_STEP)

		step = 0

		for i in range(NUM_EPOCH):
			print("In iteration: %d" % (i + 1))

			step, train_pplx = run_epoch(session, train_model, train_batches, train_model.train_op, True, step)
			print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_pplx))

			_, eval_pplx = run_epoch(session, eval_model, eval_batches, tf.no_op(), False, 0)
			print("Epoch: %d Eval Perplexity: %.3f" % (i + 1, eval_pplx))

		_, test_pplx = run_epoch(session, eval_model, test_batches, tf.no_op(), False, 0)
		print("Test Perplexity: %.3f" % test_pplx)

if __name__ == "__main__":
	main()