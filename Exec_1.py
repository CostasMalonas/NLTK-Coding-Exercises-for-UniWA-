# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 18:03:48 2020

@author: kosta
"""

import nltk
import matplotlib
nltk.download()
from __future__ import division
import random
from nltk.book import *

#Ερώτημα 1α =========================================================

#Θα υπολογίσουμε πόσο πλούσιο είναι το λεξιλόγιο στο text6
def Richness(var):
  print("Richness:",len(var) / len(set(var))) #set(text6) υπολογίζει τα tokens
 
Richness(text6) #Συγκριτικά με τα περισσότερα κείμενα παρατηρούμε πως
                # το text6 είναι περισσότερο φτωχό ως προς το λεξιλόγιο.




#Η λέξη lol εμφανίζεται πολύ περισσότερες φορές από τις άλλες 2.
def countWordsAndPercent(w, text):
    for i in w:
        print("Word " + i + " appears: ",  text.count(i))
        print("Percent of " + i + " is: ", 100 * text.count(i) / len(text))
        print("\n")
    
words = ['omg', 'OMG', 'lol']               

countWordsAndPercent(words, text5)

word = ['LAUNCELOT']

countWordsAndPercent(word, text6)

#======================================================================



#Ερώτημα 1β ==========================================================

i = 0
MontyPython = []
ChatCorpus = []
while i < 3:
    MontyPython.append(text6[random.randint(0, len(text6) - 1)]) #random.randint(0, len(text6)
    ChatCorpus.append(text5[random.randint(0, len(text5) - 1)])  #παίρνουμε τυχαίους ακέραιους
    i+=1

"""
 Παρατηρούμε όπως είναι λογικό πως
 τα σημεία στίξης έχουν πολύ μεγαλύτερη
 συχνότητα εμφάνισης. Επίσης τα ρήματα που είναι σε πρώτο πρόσωπο 
 π.χ I want έχουν μεγαλύτερο ποσοστό εμφάνισης.

"""
countWordsAndPercent(MontyPython, text6) 

countWordsAndPercent(ChatCorpus, text5) 


"""
Τελικά συμπαιρένουμε πως οι λέξεις γραμμένες με μικρά εμγανίζονται 
με μεγαλύτερη συχνότητα από ότι η λέξεις με κεφαλαία
(Στο text5 διότι στο text6 όλα είναι κεφαλαία). 
Επίσης παρατηρούμε πως τα σημεία στίξης κατέχουν πολύ μεγαλύτερο ποσοστό
εμφάνισης, όπως επίσης και οι λέξεις the, to, a κ.λ.π
"""
fdist1 = FreqDist(text1)
fdist1
fdist1.most_common(50) 
fdist1.plot(50)


#Ερώτηση 2
"""
Δεν μπορούμε να βγάλουμε κάποιο συμπέρασμα για το κείμενο από το γράφημα που 
εμφαν΄ίζεται διότι οι λέξεις που εμφανίζονται με μεγαλύτερη συχνότητα είναι 
άρθρα(the, of, that) κ.τ.λ.
"""

#Ερώτηση 3

"""
Η λέξη ARTHUR εμφανίζεται με μεγαλύτερη συχνότητα από το LAUNCELOT.
Όπως βλέπουμε η λέξη KNIGHT εμφανίζεται και αυτή σε μεγάλη συχνότητα.
Οπότε συμπεραίνουμε πως το βιβλίο είναι κάποιο μυθιστόρημα που έχει να 
κάνει με ιππότες.
"""
fdist6 = FreqDist(text6)
fdist6
fdist6.most_common(50) 
fdist6.plot(50)

#Βήμα 2 ================================================================

sent1
tokens1=sent1 
normalized_sent1=[x.lower() for x in tokens1]
normalized_sent1



#Κανονικοποίηση text6
tokens6=text6 
normalized_text6=[x.lower() for x in tokens6]
normalized_text6[0:10]

fdist6 = FreqDist(normalized_text6)
fdist6
fdist6.most_common(50) 
fdist6.plot(50)

#Ερώτηση 4 Παρατηρούμε πως μικρές αλλαγές πραγματοποιούνται όσον αφορά κάποιες προθέσεις.

"""
Το stemming είναι αποτελεσματικό όταν στο ερώτημα δεν έχουμε AND
"""
tokens1 = text2[0:200] #Βάζουμε τις πρώτες 200 λέξεις στο tokens1
porter = nltk.PorterStemmer()
for t in tokens1:
   print(porter.stem(t)) #stemming


#Κανονικοποίηση
"""
Με την κανονικοποιήση επιτυγχάνουμε να φέρουμε τους όρους του ευρετηρίου
και του ερωτήματος στην ίδια μορφή.
"""
nltk.download('wordnet')
wnl = nltk.WordNetLemmatizer()
for t in tokens1:
    print(wnl.lemmatize(t))

textGreek = "Όταν η αγάπη σε καλεί , ακολούθησέ την ,"
token = textGreek.split()
token

#Stemming ελληνικού κειμένου. Δεν συμβαίνει τίποτα.
porter = nltk.PorterStemmer()
for t in token:
   print(porter.stem(t))

"""
Κανονικοποίηση ελληνικού κειμένου. Δεν συμβαίνει τίποτα. 
Επίσης άμα αλλάξουμε το κείμενο σε αγγλικά πάλι δεν παίρνουμε
κάποιο αποτέλεσμα όσον αφορά την κανονικοποίηση. Όμως άμα κάνουμε
κανονικοποίηση με την χρήση της lower συνάρτησης τότε παίρνουμε αποτέλεσμα.
"""
textEnglish = "HELLO I AM KOSTAS"
token = textEnglish.split()

nltk.download('wordnet')
wnl = nltk.WordNetLemmatizer()
for t in token:
    print(wnl.lemmatize(t))
 
    
normalized_text=[x.lower() for x in token]
normalized_text  


textGreek_2 = "ΓΕΙΑ ΕΙΜΑΙ Ο ΚΩΣΤΑΣ"
token = textGreek_2.split()  

normalized_text=[x.lower() for x in token]
normalized_text     
    

"""
Ερώτηση 5
Τελικό συμπέρασμα είναι πως ο παραπάνω κώδικας για stemming και 
κανονικοποίηση δεν δουλεύει με τα ελληνικά. Επίσης για ελληνικά 
πρέπει να χρησιμοποιούμε την συνάρτηση lower για να κάνουμε 
κανονικοποίηση.
"""


#ΕΡΩΤΗΣΗ 6==================================================================


import nltk.tokenize

sentence = "Monticello wasn't designated as UNESCO World Heritage Siteuntil 1987."
nltk.word_tokenize(sentence)

"""
To text2 είναι ήδη tokenized. Χρησιμοποιήσαμε το textGreek_2 και δούλεψε κανονικά.
Για να δουλέψει η συνάρτηση πρέπει να της περάσουμε σαν είσοδο string. Επίσης
χρησιμοποιήσαμε και ένα ποιήμα και πάλι πήραμε σωστό αποτέλεσμα. Η μέθοδος 
με την συνάρτηση tokenize είναι καλύτερη γιατί πιάνει και τα σημεία στίξης.
"""
nltk.word_tokenize(textGreek_2)

poem = "ΥΠΕΡΑΣΠΙΣΗ ΤΗΣ ΧΑΡΑΣ Να υπερασπιστούμε τη χαρά σαν να ‘ναι οχυρό να την υπερασπιστούμε από τα σκάνδαλα και τη ρουτίνα από τη μιζέρια και τους μίζερους από τις προσωρινές και οριστικές απουσίες να υπερασπιστούμε τη χαρά σαν αρχή να την υπερασπιστούμε από την έκπληξη και τους εφιάλτες από τους ουδέτερους και τα νετρόνια από τις γλυκές ατιμώσεις και τις άσχημες διαγνώσεις"
nltk.word_tokenize(poem)

poem[0:20]

#Βήμα 4 Αφαίρεση σημείων στίξης και προθημάτων (stop words)

import string
print(string.punctuation)

cleaned_tokens = []
for token in text6:
 if token not in string.punctuation:
     cleaned_tokens.append(token )


cleaned_tokens[0:20]

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
print(stopwords)

cleaned_tokens_2 = []
for token in cleaned_tokens:
 if token not in stopwords:
     cleaned_tokens_2.append(token)
     
cleaned_tokens_2[0:50]     
     
type(text2)
# ΕΡΩΤΗΣΗ 7 ===============================================================

#stopwords αγγλικής γλώσσας
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
stopwords
len(stopwords) #179

#stopwords ελληνικής γλώσσας
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('greek')
stopwords
len(stopwords) #265

def clean_marks_stopwords(text, language):
    
    normalized_text=[x.lower() for x in text] #Πρώτα κανονικοποιούμε το κείμενο
    cleaned_text = []
    for token in normalized_text:
        if token not in string.punctuation:
            cleaned_text.append(token)
    
    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words(language)  
        
    
    cleaned_text_final = []
    for word in cleaned_text:
        if word not in stopwords:
            cleaned_text_final.append(word)
            
    print(cleaned_text_final)
    return cleaned_text_final

# ΕΡΩΤΗΣΗ 8 =========================================================

cleaned_text = clean_marks_stopwords(text2[0:200], 'english')
cleaned_text[0:20]



sentence = "Καλησπέρα, ονομάζομαι Κων/νος Μαλωνάς και είμαι φοιτητής, στο Π.Α.Δ.Α  "
tokens = nltk.word_tokenize(sentence)
normalized_greek_sentence=[x.lower() for x in tokens]
normalized_greek_sentence[0:5]
clean_marks_stopwords(normalized_greek_poem, 'greek') #Παίρνουμε σωστά αποτελέσματα και με ελληνικό κείμενο


     
     
# ΕΡΩΤΗΣΗ 9 =========================================================

fdist2 = FreqDist(text2[0:200])
fdist2


clean_text_2 = clean_marks_stopwords(text2[0:200], 'english')
clean_text_2[0:20]
fdist2 = FreqDist(clean_text_2)
fdist2
#Παραρηρούμε πως μετά τον καθαρισμό του κειμένου από τα stopwords και 
#την κανονικοποίηση του η λέξη με την μεγαλύτερη συχνότητα είναι το estate.
#Aπό τις λέξεις που ακολουθούν συμπαιρένουμε πως πρόκειται για κάποιο λογοτεχνικό
#μυθιστόρημα που εκτυλίζεται στην Αγγία το 1811
