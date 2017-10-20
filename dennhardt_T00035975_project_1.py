## Author: Cody Dennhardt
## Date: 16/10/2017
## Natural Language Processing : Project 1

# Imports
import nltk
import os
import timeit
import re
import pickle
from urllib.request import urlopen
#from operator import itemgetter

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
V = 3000 # 'V' is the size of our model dictionary which we will populate with 'V-1' choices of follow up word in any given bigram
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
	bigrams = nltk.bigrams(token.lower() for token in tokens)
	
	# Create Frequency Dist for tokens and bigrams
	fdist = nltk.FreqDist(token.lower() for token in tokens)
	fdistBigrams = nltk.FreqDist(bigrams)
	
	# Get V amount of most popular tokens and make a V*(V-1) dictionary with frequency counts
	words = fdist.most_common(V)
	for word in words:
		model[word[0]] = {}
		for word2 in words:
			if(word[0] != word2[0]):
				pair = (word[0], word2[0])
				amount = fdistBigrams[pair]
				model[word[0]][word2[0]]=amount
	
	with open(modelFileName, 'wb') as handle:
		pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Timing for optimization purposes
end = timeit.default_timer()
print(end-start)
