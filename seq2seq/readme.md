## Download dataset:
[IWLST TED](https://wit3.fbk.eu/mt.php?release=2015-01)

## Delete the html lines
**data/clean_data.py** cleaned all the html tags and its contents line by line
**train.raw.en** and **train.raw.zh** are the cleaned files

## Tokenize:
### For English: 
use tokenizer: [moses](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl)
**perl ./moses_tokenizer.perl -no-escape -1 en <./train.raw.en> train.txt.en**
source file: ./train.raw.en
target file: train.txt.en
-no-escape: not replace the signals to html tags(ex. ':' => '&quot;')
-1 en: source file language is English

**Notice:** 
You have to create folder **../share/nonbreaking_prefixes** and download file **nonbreaking_prefix.en** from https://github.com/moses-smt/mosesdecoder/tree/master/scripts/share/nonbreaking_prefixes into this direction. 
Otherwise an error will occur: WARNING: No known abbreviations for language 'en', attempting fall-back to English version...
ERROR: No abbreviations files found in /path/to/perl/file/../share/nonbreaking_prefixes

### For Chinese:
segment by character
**sed 's/ //g; s/\(.\)/\1 /g' ./train.raw.zh > train.txt.zh**
ex. 大 卫 . 盖 罗 ： 这 位 是 比 尔 . 兰 格 ， 我 是 大 卫 . 盖 罗 。 

**Notice**
in the book, it gives the following commend:
**sed 's/ //g; s/\B/ /g' ./train.raw.zh > train.txt.zh**
's/ //g' : delete the space
's/\B/ /g' : replace the boundary of characters to space
however, it doesn't work properly.
eg. 大 卫.盖 罗：这...
大
卫.盖
罗：这
...
it doesn't split the signals and the characters.


## Generate vocal_en and vocab_zh files
1. Sorted by frequency
2. Limit the vocab size
3. Add <sos>, <eos>, <unk> to vocab files
4. Replace the rare words/characters to <unk>

## Convert to cnt format
Sample result: line 1: 
大 卫 . 盖 罗 ： 这 位 是 比 尔 . 兰 格 ， 我 是 大 卫 . 盖 罗 。 
30 787 148 931 630 104 10 235 7 144 300 148 680 502 4 5 7 30 787 148 931 630 6 2



