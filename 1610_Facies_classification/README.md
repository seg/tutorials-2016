## Facies Classification
This [notebook](https://github.com/brendonhall/facies_classification/blob/master/Facies%20Classification%20-%20SVM.ipynb) is a demonstration of using a machine learning algorithm (support vector machine) to assign facies to well log data.  Training data has been assembled based on expert core description combined with wireline data from nine wells.  This data is used to train a support vector machine to indentify facies based only on wireline data.  This is based on a class exercise from the University of Kansas that can be [found online](http://www.people.ku.edu/~gbohling/EECS833/).

## Erratum

> For example, if the classifier predicts that an interval is fine siltstone (FSiS), there is an 62% probability that the interval is actually **sandstone** (precision).

...should read...

> For example, if the classifier predicts that an interval is fine siltstone (FSiS), there is an 62% probability that the interval is actually **that lithology** (precision).

This has been corrected in [the version on GitHub](https://github.com/seg/tutorials-2016/blob/master/1610_Facies_classification/manuscript/facies_classification.md).
