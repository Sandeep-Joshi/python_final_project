# -*- coding: utf-8 -*-
# -*- coding: iso-8859-15 -*-
"""
A simple script that demonstrates how we classify textual data with sklearn.

"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sets import Set
import urllib
import json
import os
from stemming.porter2 import stem
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.externals import joblib
import pickle

def replace_all(text, dic):
    for i, j in dic.iteritems():
        text = text.replace(i, j)
    return text

#read the reviews and their polarities from a given file
#pos = Set(line.strip() for line in open('positive-words.txt'))
#neg = Set(line.strip() for line in open('negative-words.txt'))
toskip = Set([24,35,40,41,46,47,49,87,92,93,101,102,107,120,123,146,159,180,181,194,209,211,238,256,258,262,287,293,296,308,323,325,335,348,379,394,404,409,411,416,435,444,446,448,470,475,480,486,502,510,513,517,531,572])

# iterated the files and train

cachedStopWords = stopwords.words("english")
operators = set(('nor', 'against', 'not', 'off', 'again', 'no', 'don/'''))
negations = '(.*?nor[^a-z])|(.*?neither)|(.*?against)|(.*?never)|(.*?not[^a-z])|(.*?n\s*\'\s*t)|(.*?n\s*\"\s*t)'
cachedStopWords = set(cachedStopWords) - operators


def createFitDumps():
    file = open(os.path.basename(__file__), 'rb')
    flag = False
    dfile = None
    fitFileList = Set()
    for i, line in enumerate(file):
        # print line
        if not flag and line.startswith("##<<--.-->>##"):
            flag = True
            continue
        elif flag:
            # Create file dumps
            if line.startswith("namedump"):
                if dfile:
                    dfile.close()
                filename = line.split(" = ")[1].replace('"', '').replace("\n", "").replace("\r", "")
                # print 'writing to', filename
                fitFileList.add(filename)
                dfile = open(filename, 'wb')
            elif line.startswith("valuedump"):
                continue
            elif line.startswith('"""'):
                continue
            else:
                # Write to the file which is opened
                dfile.write(line)
        else:
            continue
    return fitFileList


def process(text):
    text = ' '.join([stem(word) for word in text.split() if word not in cachedStopWords])
    return text.lower()


def sentAna(review):
    data = urllib.urlencode({"text": review})
    u = urllib.urlopen("http://text-processing.com/api/sentiment/", data).read()
    test = json.loads(u)
    return test["probability"]["neg"]

def check(text):
    # check if the line contains recommend
    d = re.findall("\.*([^.,;]*?)recommend(.*?)", text)  # not checking the following recommend
    # see if the text before contains negation words
    for item in d:
        if len(item[0]) >= 70:
            continue
        l = re.findall(negations, item[0])
        if len(l) < 1:
            return 1, 1  # no negation but there's a recommend
        else:
            return 0, 1  # recommend following a negation

    # Check if user has mentioned star rating
    d = re.findall("(\d+\.\d+|\d+|one|two|three|four|five)\s*star(?:s|\s*)(?:[^a-z])", text)
    for item in d:
        try:
            i = int(item[0])
            if i < 3:
                return 0, 3
            elif i > 3:
                return 1, 3
        except Exception:
            if item[0] == "one" or item[0] == "two":
                return 0, 3
            elif item[0] == "four" or item[0] == "five":
                return 1, 3

    # Check if the line contains go, visit etc
    # d = re.findall("\.*([^.,;]*?)(?: back| return| come)(.*?)$", text)  # not checking the following recommend
    d = re.findall("\.*([^.,;]*?)(?: back| return| come)(?:\s|\.|,|;)", text)  # not checking the following recommend
    # see if the text before contains negation words
    for item in d:
        try:
            if len(item[0]) >= 70:
                continue
            l = re.findall(negations, item[0])
            if len(l) < 1:
                return 1, 2  # no negation but there's a recommend
            else:
                return 0, 2  # recommend following a negation
        except IndexError:
            print text
            continue

    return None, 0  # no recommend found

def dump(filename, obj):
    # dump counter
    print "dumping file"
    file = open(filename, 'wb')
    pickle.dump(obj, file, 0)
    file.close()
    print "dumping file. Completed."


#Try tf-if, then work on inversing and training properly, replace neg or positive with ++ and --
def loadData(fname,isTrain):
    reviews = []
    labels = []
    f = open(fname)
    c = 1
    i = 0
    for line in f:
        i += 1
        line = line.replace("&#39;", "'")
        line = line.replace("<br>", " ")
        line = line.replace("#34;", " ")
        # fixing crawl issue should be ...
        line = re.sub('&hellip;', ' ... ', line)
        # fixing crawl issue should be &
        line = re.sub('&amp;|&', 'and', line)

        try:
            if i < 10:
                print line
            review, rating = line.split('\t')
            review = review.lower()
        except ValueError:
            print line
            continue
        reviews.append(review)
        labels.append(int(rating))

    f.close()
    return reviews, labels

rev_train, labels_train = loadData('training.txt', False)
rev_tmp, labels_tmp = loadData('1245stars.txt', False)
rev_train += rev_tmp
labels_train += labels_tmp
rev_test, labels_test = loadData('training_2.txt', False)


#Build a counter based on the training dataset

#counter = CountVectorizer(vocabulary=(pos|neg))
# counter = CountVectorizer()
# counter.fit(rev_train)
# dump("counter.pkl", counter)
counter = joblib.load('counter.pkl')

#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)  # transform the training data
counts_test = counter.transform(rev_test)  # transform the testing data
# print len(rev_test)
# pick 4 classifiers
# clf1 = LogisticRegression()
# clf2 = KNeighborsClassifier(weights='distance')
# clf3 = MultinomialNB(alpha=0.2, fit_prior=True)
# clf4 = svm.SVC(gamma=0.001, C=100., probability=True)
# clf4 = joblib.load("fit.pkl")

#build a voting classifer
# eclf = VotingClassifier(estimators=[('lr', clf1), ('knn', clf2), ('mnb', clf3), ('svm', clf4)], voting='soft',
#                         weights=[1, 1, 2, 2])

#train all classifier on the same datasets
# eclf.fit(counts_train, labels_train)
# clf1.fit(counts_train, labels_train)
# clf2.fit(counts_train, labels_train)
# clf3.fit(counts_train, labels_train)
# clf4.fit(counts_train, labels_train)

# dump fits
# dump("voting.pkl", eclf)
eclf = joblib.load("voting.pkl")
# dump("svm.pkl", clf4)



for i, clf in enumerate([eclf,]):
# for i, clf in enumerate([eclf, clf1, clf2, clf3, clf4]):
    #use hard voting to predict (majority voting)
    pred=clf.predict(counts_test)

    prob = clf.predict_proba(counts_test)

    # score = eclf.score(pred, labels_test)

    # print accuracy
    print accuracy_score(pred,labels_test)
    break
    f2=open(str(i)+'pred_new.xls','w')
    f = open(str(i)+'predictions_new.txt','w')
    f3 = open(str(i)+'test4new.xls', 'w')

    for i, p in enumerate(pred):

        f.write(str(p) + '\n')
        # if (str(labels_test[i])!=str(p)):
        #     print(rev_test[i])
        f2.write(rev_test[i] + '\t' + str(labels_test[i]) +'\n')

        # check if its in range of 5-6
        flag = ''
        value = 0
        if (prob[i][1]>=0.5 and prob[i][1]<0.6):
            value = sentAna(rev_test[i])
            if (value > .5):
                flag = 'x'
        val, id = check(rev_test[i])

        f3.write(('x' if (str(labels_test[i]) != str(p)) else '') + '\t'+ str(p) + '\t' + str(labels_test[i])
                 + '\t' + str(prob[i][1]) + '\t' + str(prob[i][1]*100//10) + '\t' + flag + '\t' + str(value)
                 + '\t' + str(id) + '\t' + str(val) + '\t'
                 + ('1' if (str(labels_test[i]) == str(val)) else '0') + '\t'
                 + ('1' if (str(labels_test[i]) == str(p)) else '0') + '\t'
                 + rev_test[i] + '\t' + flag + '\n')
    f3.close()
    f.close()
    f2.close()