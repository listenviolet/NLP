import nltk 
import urllib.request # used to download the content of the web link
from bs4 import BeautifulSoup # used to clean html

# Get the wen link content
response = urllib.request.urlopen('http://python.org/')
html = response.read()

# Clean html
soup = BeautifulSoup(html, 'lxml')
clean = soup.get_text()
tokens = [tok for tok in clean.split()]
# print(tokens[:100])

# Get the freq distribution
Freq_dist_nltk = nltk.FreqDist(tokens)
# print(Freq_dist_nltk)

# for k, v in Freq_dist_nltk.items():
# 	print(str(k),": ", str(v))

# Plot
Freq_dist_nltk.plot(50, cumulative = False)

# Clean stop words
stopwords = [word.strip().lower() for word in open('./english.stop.txt')]
clean_tokens = [tok for tok in tokens if len(tok.lower()) > 1 and (tok.lower() not in stopwords)]
clean_Freq_dist_nltk = nltk.FreqDist(clean_tokens)
clean_Freq_dist_nltk.plot(50, cumulative = False)

with open('./freq_dist_clean.txt', 'w') as f:
	for k, v in clean_Freq_dist_nltk.items():
		f.write(str(k)+':'+str(v)+'\n')
