import codecs 
import sys 

def convert(vocab_file, raw_data, output_data):
	with codecs.open(vocab_file, 'r', 'utf-8') as f_vocab:
		vocab = [w.strip() for w in f_vocab.readlines()]
	word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
	# print(word_to_id["卫"])

	def get_id(word):
		return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

	fin = codecs.open(raw_data, 'r', 'utf-8')
	fout = codecs.open(output_data, 'w', 'utf-8')

	for line in fin:
		words = line.strip().split() + ["<eos>"]
		out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
		fout.write(out_line)

	fin.close()
	fout.close()

VOCAB_EN = '../data/vocab.en'
VOCAB_ZH = '../data/vocab.zh'

RAW_EN = '../data/train.txt.en'
RAW_ZH = '../data/train.txt.zh'

OUTPUT_EN = '../data/train.en'
OUTPUT_ZH = '../data/train.zh'

convert(VOCAB_EN, RAW_EN, OUTPUT_EN)
convert(VOCAB_ZH, RAW_ZH, OUTPUT_ZH)