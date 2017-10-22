## Author: Cody Dennhardt
## Date: 16/10/2017
## Natural Language Processing : Project 1

# Imports
import nltk
import os
import timeit
import re
import pickle
import math
import random
from urllib.request import urlopen

# Definitions
def download_page(url):
	print('Fetching: ' + url + '\n')
	with urlopen(url) as response:
		html_content = response.read()
		encoding = response.headers.get_content_charset('utf-8')
		html_text = html_content.decode(encoding)
		return html_text

def dict_to_list(dictionary):
   return([(k, v) for k, v in dictionary.items()])

def trivialTokenizer(text):
   # remove \d+| if you want to get rid of all digit sequences
   pattern = re.compile(r"\d+|Mr\.|Mrs\.|Dr\.|\b[A-Z]\.|[a-zA-Z_]+-[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+-[a-zA-Z_]+|[a-zA-Z_]+|--|'s|'t|'d|'ll|'m|'re|'ve|[.,:!?;\"'()\[\]&@#-]")
   return(re.findall(pattern, text))

# found at : https://stackoverflow.com/questions/14992521/python-weighted-random
def weighted_choice(choices):
	choices = dict_to_list(choices)
	total = sum(w for c, w in choices)
	r = random.uniform(0, total)
	upto = 0
	for c, w in choices:
		if upto + w >= r:
			return c
		upto += w

# Create a sentence and return it and print it
def getSentence(model):
	sentence = []
	avgProb = 0
	sentenceEnders = ['.', '?', '!']
	randomStart = '.'
	while(randomStart in sentenceEnders):
		randomStart = random.choice(list(model.keys()))
	sentence.append(randomStart.capitalize())
	while(randomStart not in sentenceEnders):
		nextWordList = model[randomStart]
		nextWord = weighted_choice(nextWordList)
		avgProb += model[randomStart][nextWord]
		randomStart=nextWord
		sentence.append('I' if randomStart=='i' else randomStart)
	
	avgProb = avgProb / len(sentence)
	print(*sentence, avgProb)
	return (sentence, avgProb)

# Incase you run in an IDE which changes the CWD, this changes the save location of files to the same directory as the python file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Globals
site = 'https://sites.google.com/site/comp498004'
get = '?attredirects=0&d=1'
books = ('A Dark Nights Work.txt',
		 'An Accursed Race.txt',
		 'Cousin Phyllis.txt',
		 'Cranford.txt',
		 'Doom of the Griffiths.txt',
		 'Lizzie Leigh.txt',
		 'Mary Barton.txt',
		 'My Lady Ludlow.txt',
		 'North and South.txt',
		 'Ruth.txt',
		 'Sylvias Lovers.txt',
		 'The Moorland Cottage.txt',
		 'The Poor Clare.txt',
		 'Wives and Daughters.txt')
# Start timing of getting all text: Testing for optimization
start = timeit.default_timer()
tokens = []
V = 4000 # 'V' is the size of our model dictionary which we will populate with 'V-1' choices of follow up word in any given bigram
model = {} # The dictionary in which I create my model
modelFileName = 'model{}.pickle'.format(V)
if os.path.exists(modelFileName) and os.path.getsize(modelFileName) > 0:
	with open(modelFileName, 'rb') as handle:
		model = pickle.load(handle)
else:
	# Get all the tokens of all books listed in books
	for book in books:
		tokens = tokens + trivialTokenizer(download_page(site + '/data/' + str(book.replace(' ', '%20')) + get))
	
	### Create bigrams of all the tokens
	### (may result in 1 weird bigram when transitioning over books,
	### I figured this isn't a huge issue but I didn't know how to append to a generator type object in python)
	bigrams = list(nltk.bigrams(token.lower() for token in tokens))
	
	# Create Frequency Dist for tokens and bigrams
	fdist = nltk.FreqDist(token.lower() for token in tokens)
	fdistBigrams = nltk.FreqDist(bigrams)
	
	# Get V amount of most popular tokens and make a V*(V-1) dictionary with frequency counts
	words = fdist.most_common(V)
	words = [w[0] for w in words]
	
	# Clean bigrams
	cleaned_bigrams = [a for a in bigrams if a[0] in words and a[1] in words]
	fdistCleaned = nltk.FreqDist(cleaned_bigrams)
	
	# TODO get bigrams with occurances less than 10, for now substituted 90% of V
	L = math.floor(len(cleaned_bigrams) * 0.9)
	lapace = 1/L
	
	print('Bigrams is len : ', len(bigrams))
	print('Cleaned is len : ', len(cleaned_bigrams))
	print('Lapace is : ', lapace)
	
	for word in words:
		model[word] = {}
		rowCount = 0
		sanityCheck = 0
		for word2 in words:
			if(word != word2):
				pair = (word, word2)
				amount = fdistCleaned[pair] + lapace
				model[word][word2]=amount
				rowCount += amount
		for word2 in words:
			if(word != word2):
				probability = model[word][word2] / rowCount
				model[word][word2] = probability
				sanityCheck += probability
		print(sanityCheck)
	
	with open(modelFileName, 'wb') as handle:
		pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Timing for optimization purposes
end = timeit.default_timer()
print(end-start)

### Prints out the model
# for item in model:
# 	print('{:15s} : {:10s}'.format('START OF BIGRAM', item))
# 	for key in model[item]:
# 		print('{:15} : {:10s} - probability {}'.format('Followed by', key, model[item][key]))
# 	print('------------------------------------------------------------------')

sentence1 = getSentence(model)
sentence2 = getSentence(model)

if(sentence1[1] > sentence2[1]):
	print("The first sentence is more probable")
elif(sentence2[1] > sentence1[1]):
	print("The second sentence is more probable")
else:
	print("Both sentences are equally probable, wow!")