MODULE = 'C:\Python27\Lib\site-packages\pattern-2.6'
import sys
import re
from itertools import islice, tee
if MODULE not in sys.path: sys.path.append(MODULE)
from pattern.web import Wikipedia, plaintext
import math
from unittest import TestCase

class WikiParser:
    def __init__(self):
        pass
    
    def get_articles(self, start):
        list_of_strings = []
        article = Wikipedia().article(start).plaintext()
        list_of_strings.append(self.plain_text(article))

        #for link in Wikipedia().article(start).links:
        #    if Wikipedia().article(link).language is 'en':
        #        list_of_strings.append(self.plain_text(link))
                
        return list_of_strings
    
    def plain_text(self, article):
        corrected = article.lower()
        corrected = re.sub('\n', ' ', corrected)
        corrected = re.sub('[,:?!;*\\\'_"\(\)\]\[]', '', corrected)
        corrected = re.sub(' - ', ' ', corrected)
        while re.search('  ', corrected) != None:
            corrected = re.sub('  ', ' ', corrected)
        return corrected


class TextStatistics:
    def __init__(self, articles):
        self._articles = articles
        pass
    
    def get_top_words(self, n, use_idf = False):
        list_of_words_in_descending_order_by_freq = []
        list_of_their_corresponding_freq = []
        for text in self._articles:
            big_dict = self.count_words(text)
        if use_idf:
            for word in big_dict:
                number_of_texts = 0
                for text in self._articles:
                    if word in text:
                        number_of_texts += 1
                idf = self.idfs(len(self._articles), number_of_texts)
                big_dict[word] = big_dict[word]*idf
        for word in sorted(big_dict, key=lambda n: big_dict[n], reverse=True):
            list_of_words_in_descending_order_by_freq.append(word)
            list_of_their_corresponding_freq.append(big_dict[word])
        return (list_of_words_in_descending_order_by_freq[:n], list_of_their_corresponding_freq[:n])

    def get_top_ngrams(self, n, use_idf = False):
        list_of_3grams_in_descending_order_by_freq = []
        list_of_their_corresponding_freq = []
        sentences = []
        for text in self._articles:
            one_text_sent = text.split('. ')
            for sent in one_text_sent:
                if sent[-2:] != '. ':
                    sent = sent + '. '
                sentences.append(sent)
            big_dict = self.make_ngrams(text)
        if use_idf:
            for ngram in big_dict:
                number_of_sentences = 0
                for sentence in sentences:
                    if ngram in sentence:
                        number_of_sentences += 1
                if number_of_sentences ==0:
                    number_of_sentences = 1
                idf = self.idfs(len(sentences), number_of_sentences)
                big_dict[ngram] = big_dict[ngram]*idf
        for ngram in sorted(big_dict, key=lambda n: big_dict[n], reverse=True):
            list_of_3grams_in_descending_order_by_freq.append(ngram)
            list_of_their_corresponding_freq.append(big_dict[ngram])
        return (list_of_3grams_in_descending_order_by_freq[:n], list_of_their_corresponding_freq[:n])
    
    def make_ngrams(self, text):
        big_dict = {}
        ngrams = zip(*(islice(seq, index, None) for index, seq in enumerate(tee(text, 3))))
        ngrams = [''.join(x) for x in ngrams]
        for gram in ngrams:
            if gram in big_dict:
                big_dict[gram] += 1
            else:
                big_dict[gram] = 0
        return big_dict

    def count_words(self, text):
        big_dict = {}
        words = text.split(' ')
        for word in words:
            if word in big_dict:
                big_dict[word] += 1
            else:
                big_dict[word] = 0
        return big_dict

    def idfs(self, num_all, num_corresponding):
        return math.log(num_all/num_corresponding)

    def test_idfs(self):
        idf = self.idfs(32, 2)
        self.assertAlmostEqual(idf, 2.77, 2)

    def test_idf_words(self):
        tuple_lists = self.get_top_words(5, use_idf = True)
        freqs = tuple_lists[1]
        for freq in freqs:
            idf = self.idfs(100, 5)
            self.assertAlmostEqual(idf, -1, 2)
    
class Experiment:
    def __init___(self):
        pass

    def show_results(self, start):
        articles = WikiParser().get_articles(start)
        top_20_ngr = TextStatistics(articles).get_top_ngrams(20, use_idf = True)
        top_20_words = TextStatistics(articles).get_top_words(20, use_idf = True)
        print 'Top-20 3grams in article', start, 'and all the articles it refers to:', top_20_ngr
        print 'Top-20 words in article', start, 'and all the articles it refers to:', top_20_words

    #По одной только статье Natural language processing, не дождалась результатов по всем ссылкам :(
    #Top-20 3grams in article Natural language processing and all the articles it refers to: ([u' th', u'the', u' of', u'ion', u'ent', u'he ', u'of ', u'. t', u'. i', u' re', u'ng ', u' in', u'. s', u'ch ', u'tio', u'ngu', u'age', u'tic', u' la', u'is '], [155.9581156259877, 138.62943611198907, 112.28984325071113, 106.74466580623158, 105.46677971213853, 104.66522426455174, 104.66522426455174, 104.40711650156649, 104.40711650156649, 104.36816742347042, 102.5857827228719, 100.50634118119207, 99.18676067648816, 97.77649369146177, 94.9611637367125, 94.48065682545744, 92.28343224812122, 91.49542783391277, 91.18481995945311, 90.10913347279289])
    #Top-20 words in article Natural language processing and all the articles it refers to: ([u'essay', u'limited', u'all', u'considered.', u'semantic', u'chinese', u'17relationship', u'particular', u'results', u'existing', u'adjective', u'efficient', u'1971.', u'morphology', u'manipulate.', u'primer', u'decisions', u'worked', u'directed.', u'relationships'], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

start = 'Natural language processing'
c = Experiment().show_results(start)
