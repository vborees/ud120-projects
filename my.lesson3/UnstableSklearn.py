#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.append("../choose_your_own/")
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import shutil as shutil
from classifyDT import classify
from sklearn import tree

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

### check stability
if(len(values) == 1):
    print "computed successfully"
else:
    print "instability detected", len(values), "accuracy values computed!"






