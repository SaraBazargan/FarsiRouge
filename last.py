import sys
import glob
import errno
import codecs
import nltk
from nltk.util import ngrams
import collections
import statistics


models = []
path = 'C://Users/sara/Desktop/rouge implementation/kholaseha/model/*.txt'   
files = glob.glob(path)   
for name in files: # 'file' is a builtin type, 'name' is a less-ambiguous variable name.
    model = []
    try:
        with open(name, encoding= 'utf-8') as f: # No need to specify 'r': this is the default.
            ff = f.read()
            fff = ff.split()
            #print(fff)
            models.append(fff)
            model.append(fff)
            '''
            for i in range(len(fff)):
                if fff[i] == '\n':
                    fff.pop(i)
            #bi = nltk.bigrams(fff)
            for word in fff:
                if word not in model and word != '\n':
                    model.append(word)
            '''
    except IOError as exc:
        if exc.errno != errno.EISDIR: # Do not fail if a directory is found, just ignore it.
            raise # Propagate other kinds of IOError.
    print('number of words: ', len(model[0]))

system = []  
fp = open('C://Users/sara/Desktop/rouge implementation/kholaseha/system/random_summary02.txt', 'r', encoding= 'utf-8')
txt = fp.read().split()
print('num words in system summary: ', len(txt))
'''
bi2 = nltk.bigrams(txt)
for line in txt:
    if word not in system and word != '\n':
        system.append(word)
'''
fp.close()


'''
def jaccard_similarity(x,y):
    """ 
        this measure computes the similarity based on the interdection
        of 2 sets divided by union of them.
    """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return 1-(intersection_cardinality/float(union_cardinality))
sim = jaccard_similarity(model, system)
print('similarity = ', sim)

#sim2 = jaccard_similarity(bi, bi2)
#print('similarity of bigrams = ', sim2)
'''

def counter_overlap(counter1, counter2):
    result = 0
    for k, v in counter1.items():
        result += min(v, counter2[k])
        #print(k, min(v, counter2[k]))
    return result

    

def rouge_n(system, model, n):
    """
    Compute the ROUGE-N score of a peer with respect to model, for
    a given value of `n`.
    """
    recall = 0
    matches = 0
    total_number_of_Ngram_system = 0
    total_number_of_Ngram_model = 0
    #system_words = system.split()
    system_ngram = ngrams(system, n)
    system_fd = nltk.FreqDist(system_ngram)
    #model_words = model.split()
    model_ngram = ngrams(model, n)
    model_fd = nltk.FreqDist(model_ngram) 
    
    matches = counter_overlap(system_fd, model_fd)
    
    for v in system_fd.values():
        total_number_of_Ngram_system += v
    for v in model_fd.values():
        total_number_of_Ngram_model += v
        
    recall = matches / total_number_of_Ngram_model
    #print('recall = ', recall)
    precision = matches / total_number_of_Ngram_system
    #print('matches = ', matches, len(system))
    #print('precision = ', precision)
    if (recall + precision) > 0.0:
        return ((2 * recall * precision) / (recall + precision), recall, precision)
    else:
        return (0.0, recall, precision)
    

recall_list = []
precision_list = []
temp = []
for i, model in enumerate(models):
    #print('ROUGE-1 measures with regard to reference ', i+1, ':')    
    (F_measure, r, p) = rouge_n(txt, model, 1)
    #print('F_measure = ', F_measure)
    #print()
    #if F_measure > temp:
        #temp = F_measure
    temp.append(F_measure)
    recall_list.append(r)
    precision_list.append(p)
print('ROUGE-1_F-measure mean = ', statistics.mean(temp))
print('recall_mean = ', statistics.mean(recall_list))
print('precision_mean = ', statistics.mean(precision_list))
print()
print()


recall_list = []
precision_list = []
temp = []
for i, model in enumerate(models):
    #print('ROUGE-2 measures with regard to reference ', i+1, ':')
    (F_measure, r, p) = rouge_n(txt, model, 2)
    #print('F_measure = ', F_measure)
    #print()
    #if F_measure > temp:
        #temp = F_measure
    temp.append(F_measure)
    recall_list.append(r)
    precision_list.append(p)
print('ROUGE-2_F-measure mean = ', statistics.mean(temp))
print('recall_mean = ', statistics.mean(recall_list))
print('precision_mean = ', statistics.mean(precision_list))
print()


recall_list = []
precision_list = []
temp = []
for i, model in enumerate(models):
    #print('ROUGE-3 measures with regard to reference ', i+1, ':')
    (F_measure, r, p) = rouge_n(txt, model, 3)
    #print('F_measure = ', F_measure)
    #print()
    #if F_measure > temp:
        #temp = F_measure
    temp.append(F_measure)
    recall_list.append(r)
    precision_list.append(p)
print('ROUGE-3_F-measure mean = ', statistics.mean(temp))
print('recall_mean = ', statistics.mean(recall_list))
print('precision_mean = ', statistics.mean(precision_list))
print()


recall_list = []
precision_list = []
temp = []
for i, model in enumerate(models):
    #print('ROUGE-4 measures with regard to reference ', i+1, ':')
    (F_measure, r, p) = rouge_n(txt, model, 4)
    #print('F_measure = ', F_measure)
    #print()
    #if F_measure > temp:
        #temp = F_measure
    temp.append(F_measure)
    recall_list.append(r)
    precision_list.append(p)
print('ROUGE-4_F-measure mean = ', statistics.mean(temp))
print('recall_mean = ', statistics.mean(recall_list))
print('precision_mean = ', statistics.mean(precision_list))
print()


def _get_index_of_lcs(x, y):
	return len(x), len(y)


def _len_lcs(x, y):
	'''
	Returns the length of the Longest Common Subsequence between sequences x
	and y.
	Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
	
	:param x: sequence of words
	:param y: sequence of words
	:returns integer: Length of LCS between x and y
	'''
	table = _lcs(x, y)
	n, m = _get_index_of_lcs(x, y)
	return table[n, m] ### right-upper cell of the table (DP table)


def _lcs (x, y):
	'''
	Computes the length of the longest common subsequence (lcs) between two
	strings. The implementation below uses a DP programming algorithm and runs
	in O(nm) time where n = len(x) and m = len(y).
	Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
	:param x: collection of words
	:param y: collection of words
	:returns table: dictionary of coord and len lcs
	'''
	n, m = _get_index_of_lcs(x, y)
	table = dict()
	for i in range(n + 1):
		for j in range (m + 1):
			if i == 0 or j == 0:
				table[i, j] = 0
			elif x[i-1] == y[j-1]:
				table[i, j] = table[i-1, j-1] + 1
			else:
				table[i, j] = max(table[i-1, j], table[i, j-1])
	return table


def _f_lcs(llcs, m, n, r_list, p_list, tmp):
	'''
	Computes the LCS-based F-measure score
	Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
	rouge-working-note-v1.3.1.pdf
	
	:param llcs: Length of LCS
	:param m: number of words in reference summary 
	:param n: number of words in candidate summary
	:returns float: LCS-based F-measure score
	'''
	r_lcs = llcs / m
	#print('recall = ', r_lcs)
	r_list.append(r_lcs)
	p_lcs = llcs / n
	#print('precision = ', p_lcs)
	p_list.append(p_lcs)
	beta = p_lcs / r_lcs
	num = (1 + (beta ** 2)) * r_lcs * p_lcs 
	denom = r_lcs + ((beta ** 2) * p_lcs)
	f_measure = num / denom
	tmp.append(f_measure)
	return f_measure


def rouge_l_sentence_level(evaluated_text, reference_text, recall_list, precision_list, temp):
	"""
	Computes ROUGE-L (sentence level) of two text collections of sentences.
	http://research.microsoft.com/en-us/um/people/cyl/download/papers/
	rouge-working-note-v1.3.1.pdf
	
	Calculated according to:
	R_lcs = LCS(X,Y)/m
	P_lcs = LCS(X,Y)/n
	F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)
	where:
	X = reference summary
	Y = Candidate summary
	m = length of reference summary
	n = length of candidate summary
	:param evaluated_sentences: 
		The sentences that have been picked by the summarizer
	:param reference_sentences:
		The sentences from the referene set
	:returns float: F_lcs
	:raises ValueError: raises exception if a param has len <= 0
	"""
	if len(evaluated_text) <= 0 or len(reference_text) <= 0: 
		raise (ValueError("Collections must contain at least 1 sentence."))
	    
	m = len(reference_text)
	n = len(evaluated_text)
	lcs = _len_lcs(evaluated_text, reference_text)
	return _f_lcs(lcs, m, n, recall_list, precision_list, temp)


recall_list = []
precision_list = []
temp = []
for i, mod in enumerate(models):
    #print('LCS_F-measure with regsrd to reference ', (i+1), ':')
    #print(i , mod)
    
    print('F-measure = ', rouge_l_sentence_level(txt, mod, recall_list, precision_list, temp))
    print()
    
print('LCS_F-measure mean = ', statistics.mean(temp))
print('recall_mean = ', statistics.mean(recall_list))
print('precision_mean = ', statistics.mean(precision_list))

