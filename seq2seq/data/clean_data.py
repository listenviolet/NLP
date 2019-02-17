import codecs
import sys

ORGIN_EN = "./train.tags.en-zh.en"
CLEANED_TRAIN_EN = "./train.raw.en"

ORGIN_ZH = "./train.tags.en-zh.zh"
CLEANED_TRAIN_ZH = "./train.raw.zh"

def cleanTags(orgin_file, cleaned_file):
	fin = codecs.open(orgin_file, 'r', 'utf-8')
	fout = codecs.open(cleaned_file, 'w', 'utf-8')

	tags = [("<description>", "</description>"), ("<url>", "</url>"), 
			("<title>", "</title>"), ("<keywords>", "</keywords>"),
			("<speaker>", "</speaker>"), ("<talkid>", "</talkid>"),
			("<reviewer", "</reviewer>"),("<translator", "</translator>")]

	for line in fin.readlines():
		flag = 1
		for item in tags:
			if item[0] in line:
				flag = 0
				break
		if flag: fout.write(line)


	fin.close()
	fout.close()

cleanTags(ORGIN_EN, CLEANED_TRAIN_EN)
cleanTags(ORGIN_ZH, CLEANED_TRAIN_ZH)