import nltk
import matplotlib
nltk.download()
from __future__ import division
import random
from nltk.book import *
import nltk.tokenize
import pandas as pd
import string
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

sentence1 = "Thomas Jefferson began building Monticello at the age of 26."
sentence2 = "Καλησπέρα ονομάζομαι Κων/νος Μαλωνάς και είμαι 24 χρονών."
sentence3 = "It's better to walk thousands of miles than to read thousands of books."
sentence4 = "Water flows in only to flow out."

# α) Tokenization με την λύση που προτείνει το βιβλίο.

sentence1.split()
sentence2.split()
sentence3.split()
sentence4.split()

# β) Tokenization με nltk.

sent1 = nltk.word_tokenize(sentence1)
sent2 = nltk.word_tokenize(sentence2)
sent3 = nltk.word_tokenize(sentence3)
sent4 = nltk.word_tokenize(sentence4)

sent = [sent1, sent2, sent3, sent4]
sent
"""
Συπέρασμα:
Παρατηρούμε πως η μέθοδος με το nltk αποδίδει
καλύτερα γιατί πιάνει και την τελεία '.'.
Παρόλαυτα παρατηρούμε πως στο sentence3 
δεν έπιασε την απόστροφο.
"""

# ΕΡΩΤΗΣΗ 1 =====================================================================

# Πίνακες συμπτώσεων με split()
df_1 = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence1.split()])), columns=['sentence1']).T
df_1

df_2 = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence2.split()])), columns=['sentence2']).T
df_2

df_3 = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence3.split()])), columns=['sentence3']).T
df_3

df_4 = pd.DataFrame(pd.Series(dict([(token, 1) for token in sentence4.split()])), columns=['sentence4']).T
df_4



# Πίνακες συμπτώσεων με nltk

df_1 = pd.DataFrame(pd.Series(dict([(token, 1) for token in sent1])), columns=['sentence1']).T
df_1

df_2 = pd.DataFrame(pd.Series(dict([(token, 1) for token in sent2])), columns=['sentence2']).T
df_2

df_3 = pd.DataFrame(pd.Series(dict([(token, 1) for token in sent3])), columns=['sentence3']).T
df_3

df_4 = pd.DataFrame(pd.Series(dict([(token, 1) for token in sent4])), columns=['sentence4']).T
df_4

"""
Συμπέρασμα:
Παρατηρούμε όπως είναι λογικό 
πως στους πίνακες συμπτώσεων με 
την μέθοδο του nltk συμπεριλαμβάνεται και η τελεία '.'.
 
"""

#ΕΡΩΤΗΣΗ 2 ======================================================================


"""
Η κάθε γραμμή σε ένα πίνακα συμπτ΄ώσεων αναπαριστά το
document. Ενώ η στήλη τις λέξεις στο vocabulary.

Το Pandas library είναι χρήσιμο για την διαχείριση 
arrays. Επίσης μπορούμε να αναπαραστήσουμε τα tokens 
του λεξικού μας σαν labels που το κάθε ξεχωριστά 
χαρακτηρίζει ένα column, όπως επίσης το ίδιο μπορεί
να κάνει και για τα rows. Ακόμα τα Pandas Dataframes
είναι ιδιαίτερα εύχρηστα αφού στην περίπτωση μας 
μπορούμε να προσθέσουμε επιπλέον sentences για
να δημιουργήσουμε έναν πίνακα συμπτώσεων που περιλαμβάνει 
περισσότερα documents και tokens.
"""

# Αλλάζουμε τις προτάσεις για να υπάρχουν περισσότερες κοινές λέξεις.
sentences = "Thomas Jefferson began building Monticello at the age of 26. \n" \
            "Καλησπέρα ονομάζομαι Κων/νος Μαλωνάς και είμαι 24 χρονών. \n" \
            "Καλησπέρα ονομάζομαι Ιωάννα, είμαι 25 χρονών. \n" \
            "Το όνομα μου είναι Γιώργος, είμαι πολιτικός μηχανικός. \n" \
            "Hello, I am Nick the Greek and I am currently staying at Flamingo Hotel."


# α)
    
corpus = {}
for i, sent in enumerate(sentences.split('\n')):
    corpus['sent{}'.format(i + 1)] = dict((tok, 1) for tok in sent.split())
    
df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
df.iloc[:,0:10] #Τυπώνουμε όλες τις γραμμές και τις πρώτες 10 στήλες.

"""
Παρατηρούμε πως οι λέξεις,
'at' και 'the', υπάρχουν και στην πρόταση 
sent1 και στην πρόταση sent5
"""
# Εμφανίζουμε τον αριθμό των επικαλυπτόμενων tokens με το .dot()

df = df.T #transpose
df

df.sent1.dot(df.sent5) # Έχουμε όπως περιμέναμε 2 κοινές λέξεις.

# Ελέγχουμε και για άλλες προτάσεις

df.sent2.dot(df.sent3) # Εμφανίζει 4 
df.sent2.dot(df.sent4) # Εμφανίζει 1
df.sent2.dot(df.sent5) # Εμφανίζει 0



# β)


# Αρχικά κάνουμε κανονικοποίηση και αφαίρεση των προθεμάτων στα text1 και text2

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

normalized_text1=[x.lower() for x in text1]
normalized_text2=[x.lower() for x in text2]

normalized_text1
normalized_text2


# Αφαιρούμε τα προθέματα και τα σημεία στίξης
clean_text1 = []
for token in text1:
 if token not in string.punctuation:
     clean_text1.append(token)


clean_text2 = []
for token in text2:
 if token not in string.punctuation:
     clean_text2.append(token)



clean_text1_2 = []
for token in clean_text1:
 if token not in stopwords:
     clean_text1_2.append(token)
     
clean_text2_2 = []
for token in clean_text2:
 if token not in stopwords:
     clean_text2_2.append(token)     

text1 = clean_text1_2
text2 = clean_text2_2


text1 = text1[0:50] # Παίρνουμε τις πρώτες 50 λέξεις του text1
text2 = text2[0:50] # Παίρνουμε τις πρώτες 50 λέξεις του text2

# Μετατρέπουμε την λίστα σε string
text1ToStr = ' '.join([str(elem) for elem in text1])
text2ToStr = ' '.join([str(elem) for elem in text2])

# Προσθέτουμε το \n στο τέλος κάθε string 
text1ToStr = text1ToStr + '\n'
text2ToStr = text2ToStr + '\n'

texts = text1ToStr + text2ToStr #concatenate

# Φτιάχνουμε των πίνακα σύμπτωσης
corpus = {}
for i, sent in enumerate(texts.split('\n')):
    corpus['text{}'.format(i + 1)] = dict((tok, 1) for tok in sent.split())

# Μετατροπή σε dataframe
df_texts = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
df_texts.drop(['text3'], axis = 0, inplace = True) #Drop την τελευταία γραμμή.

df_texts_T = df_texts.T # Κάνουμε τις γραμμές στήλες
df_texts_T
df_texts_T.text1.dot(df_texts_T.text2) # Τα δύο κείμενα έχουν 1 όμοια λέξη.


#Εμφάνιση κοινών λέξεων-συμβόλων
[(i, j) for (i, j) in (df_texts_T.text1 & df_texts_T.text2).items() if j] # [('The', 1)]




#ΕΡΩΤΗΣΗ 3 ======================================================================

"""
Δύο έγγραφα που έχουν μεγάλο ποσοστό ίδιων λέξεων
είναι περισσότερο πιθανό να πραγματεύονται παρόμοιο θέμα.
Μια καλή μέθοδος για να διαπιστώσουμε την ομοιότητα περιεχομένου,
μεταξύ εγγράφων είναι το posting list με το οποίο μπορούμε να βρούμε για
φράσεις που εμγανίζονται αυτούσιες σε παραπάνω από ένα έγγραφα, σε ποια έγγραφα
εμφανίζονται, την συχνότητα εμφάνισης της φράσης-λέξης σε κάθε έγγραφο ξεχωριστά
και την θέση μέσα στο έγγραφο.

"""




#ΕΡΩΤΗΣΗ 4 ======================================================================

# Αρχικά κάνουμε normalize



# Βρίσκουμε την συχνότητα των λέξεων σε κάθε κείμενο
count_words_text1 = {} # Λεξικό που θα έχουμε σαν values τις συχνότητες εμφάνισης κάθε λέξης.
for i in text1:
    count_words_text1['{}'.format(i)] = text1.count(i)
    
    
count_words_text2 = {} 
for i in text2:
    count_words_text2['{}'.format(i)] = text2.count(i)
    



# Κάνουμε sort το λεξικό σε ascending order
text1_sort = {}
text2_sort = {}

text1_sort = {k: v for k, v in sorted(count_words_text1.items(), key=lambda item: item[1])}
text1_sort

text2_sort = {k: v for k, v in sorted(count_words_text2.items(), key=lambda item: item[1])}
text2_sort

"""
Παρατηρούμε πως οι περισσότερες λέξεις έχουνε συχνότητα εμφάνισης 1
εκτός από τις λέξεις grammars, old, He, Usher με συχνότητα εμφάνισης 2.
Επιλέγουμε λοιπόν τις λέξεις grammars,old,He.

Από το text2 επιλέγουμε τις λέξεις estate,many,lived
"""

# text1
text1_keys = [] # Λίστα με τις λέξεις που στο λεξικό χρησιμοποιούνται σαν keys

for key in text1_sort.keys():
    text1_keys.append(key)

len(text1_keys) #46

text1_words = [] # Λίστα που θα βάλουμε τις λέξεις που θα χρησιμοποιήσουμε για το posting list
i = 0
while i < 3:
    text1_words.append(text1_keys[45 - i])
    i += 1

text1_words #['grammars', 'old', 'He']


# text2
text2_keys = [] 

for key in text2_sort.keys():
    text2_keys.append(key)

len(text2_keys)

text2_words = [] 
i = 0
while i < 3:
    text2_words.append(text2_keys[45 - i])
    i += 1


text2_words #['lived', 'many', 'estate']


# Αναπαριστούμε τις συχνότερα εμφανιζόμενες λέξεις των text1 και text2 σαν postinvg lists
def posting_list(word, num, doc):
    temp = {}
    temp[word] = [num,[doc]]
    return temp

list_txt_1_1 = posting_list('grammars', 2, 'text1')
list_txt_1_2 = posting_list('old', 2, 'text1')
list_txt_1_3 = posting_list('He', 2, 'text1')

list_txt_1_1
list_txt_1_2 
list_txt_1_3

list_txt_2_1 = posting_list('lived', 2, 'text2')
list_txt_2_2 = posting_list('many', 2, 'text2') 
list_txt_2_3 = posting_list('estate', 2, 'text2')

list_txt_2_1
list_txt_2_2 
list_txt_2_3


# Αναπαράσταση της πρότασης μας και των πρώτων 50 λέξεων των text1 και text2 σε bag of words.

norm_sent2 = [x.lower() for x in sent2]
bag_of_words_sent2 = Counter(norm_sent2)
bag_of_words_sent2 

bag_of_words_text1 = Counter(text1)
bag_of_words_text1
bag_of_words_text1.most_common(5)

bag_of_words_text2 = Counter(text2)
bag_of_words_text2
bag_of_words_text2.most_common(5)




#ΕΡΩΤΗΣΗ 5 ======================================================================
nltk.download()
from nltk.book import *


tokenizer = TreebankWordTokenizer()


# Πρόταση του βιβλίου text1

listToStr_1 = ' '.join([str(elem) for elem in sent1]) 
listToStr_1

# Πρόταση του βιβλίου text2
listToStr_2 = ' '.join([str(elem) for elem in sent2]) 
listToStr_2

docs = [listToStr_1] + [listToStr_2]
docs

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
len(doc_tokens[0])
doc_tokens

all_doc_tokens = sum(doc_tokens, [])
len(all_doc_tokens)
all_doc_tokens

lexicon = sorted(set(all_doc_tokens))
len(lexicon)
lexicon # Αφαιρούνται τα ίδια tokens στην περιπτωσή μας μόνο η '.'

# Το vector κάθε πρότασης θα έχει 14 θέσεις. Όποια λέξη δεν υπάρχει στο document θα έχει τιμή 0 
from collections import OrderedDict
zero_vector = OrderedDict((token, 0) for token in lexicon)
zero_vector


import copy
doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value / len(lexicon)
    doc_vectors.append(vec)

doc_vectors

doc_vectors

import math
def cosine_sim(vec1, vec2):
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))

    return dot_prod / (mag_1 * mag_2)


res = cosine_sim(doc_vectors[0], doc_vectors[1])
res # 0.15075567228888184

"""
Συμπέρασμα:
Οι δύο προτάσεις είναι ελάχιστα παρ΄όμοιες,
όπως βλέπουμε και α΄πο το cosine similarity το 
οποίο είναι ποιο κοντά στο 0 παρά στο 1,
κάτι που θα υποδίκνυε πως οι vectors των προτάσεων, δείχνουν
στην ίδια κατε΄ύθυνση προς όλες τις διαστάσεις. Στην πραγματικότητα
το μόνο token που έχουν κοινό οι δύο προτάσεις είναι η '.'. 
"""


#ΕΡΩΤΗΣΗ 6 ======================================================================

# Ξαναέκανε nlt.dowload και from nltk.book import * για να πάρω τα text1 και text2

len(text1) #260819
len(text2) #141576

clean_text1 = []
for token in text1:
 if token not in string.punctuation:
     clean_text1.append(token)



text1 = []
for token in clean_text1:
 if token not in stopwords:
     text1.append(token)

clean_text2 = []
for token in text2:
 if token not in string.punctuation:
     clean_text2.append(token)



text2 = []
for token in clean_text2:
 if token not in stopwords:
     text2.append(token)     


len(text1) #122226
len(text2) #62033

#Μετατροπή σε string
str_txt_1 = ' '.join([str(elem) for elem in text1]) 
str_txt_1

str_txt_2 = ' '.join([str(elem) for elem in text2]) 
str_txt_2



docs = [str_txt_1] + [str_txt_2]
docs

doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
len(doc_tokens[0])
doc_tokens

all_doc_tokens = sum(doc_tokens, [])
len(all_doc_tokens)
all_doc_tokens

lexicon = sorted(set(all_doc_tokens))
len(lexicon)
lexicon # Αφαιρούνται τα ίδια tokens στην περιπτωσή μας μόνο η '.'

# Το vector κάθε πρότασης θα έχει 14 θέσεις. Όποια λέξη δεν υπάρχει στο document θα έχει τιμή 0 
from collections import OrderedDict
zero_vector = OrderedDict((token, 0) for token in lexicon)
zero_vector


import copy
doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value / len(lexicon)
    doc_vectors.append(vec)

doc_vectors

import math
def cosine_sim(vec1, vec2):
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]
    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]
    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))

    return dot_prod / (mag_1 * mag_2)




res = cosine_sim(doc_vectors[0], doc_vectors[1])
res #0.9249671301142237

"""
Το cosine similarity των κειμένων text1 και text2,
πλησιάζει το 1. Άρα τα δύο κείμενα χρησιμοποιούν πολλές 
παρόμοιες λέξεις. Άμα πρώτα αφαιρέσουμε από τα κείμενα,
τα σημεία στίξης και τα προθέματα τότε το cosine similarity
μειώνεται στο 0.7379495237201621 που πάλι υποδηλώνει υψηλή 
ομοιότητα μεταξύ των κειμένων.
"""



# TF-IDF ======================================================================



!pip install sklearn

from sklearn.feature_extraction.text import TfidfVectorizer


sent1=[x.lower() for x in sent1]
sent2=[x.lower() for x in sent2]

sent1 = ' '.join([str(elem) for elem in sent1]) 
sent1

sent2 = ' '.join([str(elem) for elem in sent2]) 
sent2

docs = [sent1] + [sent2]
docs

corpus = docs
vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(corpus)
print(model.todense().round(2))

# ΕΡΩΤΗΣΗ 7 ======================================================================


"""
Το TF-IDF είναι περισσότερο περιγραφικό 
από το πλήθος εμφάνισης της λέξης (word count) 
στο document. Γι΄  αυτό τον λόγο αντικαθοστούμε το word count
στο vector κάθε document με το TF-IDF της λέξης.
"""


sent1
sent2 
docs = [sent1] + [sent2]


document_tfidf_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    
    for key, value in token_counts.items():
        docs_containing_key = 0
        for _doc in docs:
            if key in _doc:
                docs_containing_key += 1
        tf = value / len(lexicon)
        if docs_containing_key:
            idf = len(docs) / docs_containing_key
        else:
            idf = 0
        vec[key] = tf * idf
    document_tfidf_vectors.append(vec)

document_tfidf_vectors[0]
















query_txt1_1 = "What should I call him?" 
query_txt1_2 = "Call Ishmael?" 

query_txt2_1 = "Which family settled in Sussex.?"
query_txt2_2 = "Family of dashwood settled where.?"

query_vec = copy.copy(zero_vector)




tokens = tokenizer.tokenize(query_txt1_2.lower())
token_counts = Counter(tokens)
for key, value in token_counts.items():
    docs_containing_key = 0
    for _doc in docs:
        if key in _doc.lower():
            docs_containing_key += 1
    if docs_containing_key == 0:
        continue
    tf = value / len(tokens)
    idf = len(docs) / docs_containing_key
    query_vec[key] = tf * idf

print(cosine_sim(query_vec, document_tfidf_vectors[0]))
print(cosine_sim(query_vec, document_tfidf_vectors[1]))



 
