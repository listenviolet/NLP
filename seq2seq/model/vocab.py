import codecs 
import collections
from operator import itemgetter 

def generateVocabFile(raw_data_file, vocab_file, length_limit):
	counter = collections.Counter()

	with codecs.open(raw_data_file, 'r', 'utf-8') as f:
		for line in f:
			for word in line.strip().split():
				counter[word] += 1

	sorted_word_to_cnt = sorted(counter.items(), key = itemgetter(1), reverse = True)
	# convert into list
	sorted_words = [x[0] for x in sorted_word_to_cnt]

	sorted_words = ['<unk>', '<sos>', '<eos>'] + sorted_words
	if len(sorted_words) > length_limit:
		sorted_words = sorted_words[: length_limit]

	with codecs.open(vocab_file, 'w', 'utf-8') as file_output:
		for word in sorted_words:
			file_output.write(word + '\n')

RAW_EN = '../data/train.txt.en'
VOCAB_EN = '../data/vocab.en'
LIMIT_EN = 10000

RAW_ZH = '../data/train.txt.zh'
VOCAB_ZH = '../data/vocab.zh'
LIMIT_ZH = 4000

generateVocabFile(RAW_EN, VOCAB_EN, LIMIT_EN)
generateVocabFile(RAW_ZH, VOCAB_ZH, LIMIT_ZH)