#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.append("../choose_your_own/")

from prep_terrain_data import makeTerrainData

from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()

values = []

for step in range(10):

    ### the classify() function in classifyDT
    clf = classify(features_train, labels_train)

    ### predict
    pred = clf.predict(features_test)

    ### compute accuracy
    acc = clf.score(features_test, labels_test)

    ### add value to the list
    if(values.count(acc) == 0):
        values.append(acc)

    print step, " ", acc

# check stability
if(len(values) == 1):
    print "computed successfully"
else:
    print "instability detected", len(values), "accuracy values computed!"






