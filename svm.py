from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
from stemming.porter2 import stem
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.wordnet import WordNetLemmatizer
import re

# iterated the files and train

cachedStopWords = stopwords.words("english")
operators = set(('nor', 'against', 'not', 'off', 'again', 'no', 'don/'''))
cachedStopWords = set(cachedStopWords) - operators
print cachedStopWords

def process(text):
    text = ' '.join([stem(word) for word in text.split() if word not in cachedStopWords])
    return text.lower()


def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    count = 1
    for line in f:
        line = line.replace("&#39;", "'")
        line = line.replace("<br>", " ")
        line = line.replace("#34;", " ")
        # fixing crawl issue should be ...
        line = re.sub('&hellip;', ' ... ', line)
        # fixing crawl issue should be &
        line = re.sub('&amp;|&', 'and', line)

        try:
            review, rating=line.strip().split('\t')
            reviews.append(process(review))
            labels.append(rating)
            count += 1
        except Exception:
            print 'error in line ', count, ":", line
            continue

    f.close()
    return reviews,labels

# rev_train, labels_train=loadData('addreviews2.txt')
rev_train, labels_train = loadData('training.txt')
rev_tmp, labels_tmp = loadData('1245stars.txt')
rev_train += rev_tmp
labels_train += labels_tmp
rev_test, labels_test = loadData('training_2.txt')

#Build a counter based on the training dataset
counter = CountVectorizer()
counter.fit(rev_train)
# dump counter
file = open("counter.pkl", 'wb')
pickle.dump(counter, file, 0)
file.close()


#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
# joblib.dump(counts_train, 'counts_train.pkl')
counts_test = counter.transform(rev_test)#transform the testing data

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(counts_train, labels_train)
file = open("fit.pkl", 'wb')
pickle.dump(clf, file, 0)
file.close()


# joblib.dump(clf, 'fit.pkl')
predicted = clf.predict(counts_test)

#print the accuracy
print accuracy_score(predicted, labels_test)
