# -*- coding: utf-8 -*-
# -*- coding: iso-8859-15 -*-
from time import sleep
import os
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import sys
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sets import Set
from stemming.porter2 import stem
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import re
import pickle
import urllib
import json

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
    file = open(filename, 'wb')
    pickle.dump(obj, file, 0)
    file.close()


def process(text):
    text = ' '.join([stem(word) for word in text.split() if word not in cachedStopWords])
    return text.lower()


def loadData(fname, ifTest):
    reviews = []
    labels = []
    f = open(fname)
    for line in f:
        if ifTest:
            review = line.split('\t')[0].strip().lower()
        else:
            review, rating = line.strip().split('\t')
            labels.append(rating)
        reviews.append(review.lower())
    f.close()
    return reviews, labels


def main(argv):
    filelist = createFitDumps()

    rev_train, labels_train = loadData('training.txt', False)
    rev_tmp, labels_tmp = loadData('1245stars.txt', False)
    rev_train += rev_tmp
    labels_train += labels_tmp
    rev_test, labels_test = loadData('training_2.txt', False)

    counter = joblib.load('counter.pkl')
    counts_train = counter.transform(rev_train)
    counts_test = counter.transform(rev_test)
    # Build a counter based on the training dataset
    estimator = joblib.load("fit.pkl")


    # clf1 = LogisticRegression()
    # clf2 = KNeighborsClassifier(weights='distance')
    clf3 = MultinomialNB(alpha=0.2, fit_prior=True)
    # clf4 = svm.SVC(gamma=0.001, C=100., probability=True)

    # Fit them
    eclf = joblib.load("voting.pkl")
    #build a voting classifer
    # eclf = VotingClassifier(estimators=[('lr', clf1), ('knn', clf2), ('mnb', clf3), ('svm', clf4)], voting='soft',
    #                         weights=[1, 1, 2, 2])

    #train all classifier on the same datasets
    # eclf.fit(counts_train, labels_train)

    # clf1.fit(counts_train, labels_train)
    # clf2.fit(counts_train, labels_train)
    clf3.fit(counts_train, labels_train)
    clf4 = joblib.fit("svm.pkl")

    predicted = estimator.predict(counts_test)
    # print the accuracy
    # print accuracy_score(predicted, labels_test)

    # Write output
    f = open('predictions.txt', 'w')
    for p in predicted:
        f.write(str(p) + '\n')
    f.close()

    # remove junk files
    for file in filelist:
        print "deleting file {0}".format(file)
        os.remove(file)
    exit()

if __name__ == "__main__":
    main(sys.argv)
    exit()
