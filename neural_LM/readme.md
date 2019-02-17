## Dataset:
	[PTB dataset](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-example.tgz)

	with 9998 different words and one <unk>

	add <eos> manually => total 10000 words in vocab file.

## Batching method:
	combine all the sentences into a string and then split it.
	each batch contains #batch_size subsequences
	each subsequence length: num_steps
	so that num_batches: total_words // (batch_size * num_steps)

## Reduce parameter numbers:
	share weight between softmax and embedding

## Results:
	(the results are different from the book)
	In iteration: 1
	After 0 steps, perplexity is 10048.026
	After 100 steps, perplexity is 633.517
	After 200 steps, perplexity is 137.637
	After 300 steps, perplexity is 55.781
	After 400 steps, perplexity is 30.295
	After 500 steps, perplexity is 19.247
	After 600 steps, perplexity is 13.663
	After 700 steps, perplexity is 10.366
	After 800 steps, perplexity is 8.282
	After 900 steps, perplexity is 6.883
	After 1000 steps, perplexity is 5.893
	After 1100 steps, perplexity is 5.158
	After 1200 steps, perplexity is 4.593
	After 1300 steps, perplexity is 4.152
	Epoch: 1 Train Perplexity: 4.054
	Epoch: 1 Eval Perplexity: 1.265
	In iteration: 2
	After 1400 steps, perplexity is 1.176
	After 1500 steps, perplexity is 1.157
	After 1600 steps, perplexity is 1.145
	After 1700 steps, perplexity is 1.130
	After 1800 steps, perplexity is 1.118
	After 1900 steps, perplexity is 1.107
	After 2000 steps, perplexity is 1.098
	After 2100 steps, perplexity is 1.090
	After 2200 steps, perplexity is 1.083
	After 2300 steps, perplexity is 1.077
	After 2400 steps, perplexity is 1.072
	After 2500 steps, perplexity is 1.067
	After 2600 steps, perplexity is 1.063
	Epoch: 2 Train Perplexity: 1.061
	Epoch: 2 Eval Perplexity: 1.060
	In iteration: 3
	After 2700 steps, perplexity is 1.020
	After 2800 steps, perplexity is 1.013
	After 2900 steps, perplexity is 1.012
	After 3000 steps, perplexity is 1.011
	After 3100 steps, perplexity is 1.010
	After 3200 steps, perplexity is 1.010
	After 3300 steps, perplexity is 1.009
	After 3400 steps, perplexity is 1.009
	After 3500 steps, perplexity is 1.008
	After 3600 steps, perplexity is 1.008
	After 3700 steps, perplexity is 1.008
	After 3800 steps, perplexity is 1.007
	After 3900 steps, perplexity is 1.007
	Epoch: 3 Train Perplexity: 1.007
	Epoch: 3 Eval Perplexity: 1.032
	In iteration: 4
	After 4000 steps, perplexity is 1.017
	After 4100 steps, perplexity is 1.006
	After 4200 steps, perplexity is 1.005
	After 4300 steps, perplexity is 1.004
	After 4400 steps, perplexity is 1.004
	After 4500 steps, perplexity is 1.004
	After 4600 steps, perplexity is 1.004
	After 4700 steps, perplexity is 1.004
	After 4800 steps, perplexity is 1.004
	After 4900 steps, perplexity is 1.004
	After 5000 steps, perplexity is 1.003
	After 5100 steps, perplexity is 1.003
	After 5200 steps, perplexity is 1.003
	After 5300 steps, perplexity is 1.003
	Epoch: 4 Train Perplexity: 1.003
	Epoch: 4 Eval Perplexity: 1.017
	In iteration: 5
	After 5400 steps, perplexity is 1.004
	After 5500 steps, perplexity is 1.003
	After 5600 steps, perplexity is 1.003
	After 5700 steps, perplexity is 1.003
	After 5800 steps, perplexity is 1.002
	After 5900 steps, perplexity is 1.002
	After 6000 steps, perplexity is 1.002
	After 6100 steps, perplexity is 1.002
	After 6200 steps, perplexity is 1.002
	After 6300 steps, perplexity is 1.002
	After 6400 steps, perplexity is 1.002
	After 6500 steps, perplexity is 1.002
	After 6600 steps, perplexity is 1.002
	Epoch: 5 Train Perplexity: 1.002
	Epoch: 5 Eval Perplexity: 1.006
	Test Perplexity: 1.001

