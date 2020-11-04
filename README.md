# NAIVE BAYES CLASSIFIER

This code implements the Naive Bayes Classifier which is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set.

## Aim
Given 1000 training documents from each group. Learn to classify new documents according to which newsgroup it came from and then evaluate the performance of the classifier. Use both Maximum Likelihood Estimator and Bayesian Estimator and indicate which performed better.

## Experimental Results

* Run time of the program was 1 minute 17 seconds
* Probability that a word wk is found in a given class j using MLE, PMLE(wk/ωj) = nk/n. Thus, if the value of nk which is the number of times work wk occurs in all documents in class ωj is 0, the probability also becomes zero.
* On the other hand, probability that a word w is found in a given class j using BE, PBE(wk/ωj) = (nk + 1)/(n + |vocabulary|). Here, even if the value of nk is 0, the probability is not zero due the presence of 1 in the numerator.
* From the experimental results, we observe that the results or the predictions for the training data set were better than the test data set. This is expected as the model or the classifier is itself built by the training data set and hence, it will have the highest performance on it.
* If we compare the results for Bayesian and Maximum Likelihood Estimators, we see that the MLE works better than BE only on training data set. On the test data set, MLE performs the worst, whereas there is very slight difference in accuracy of Bayesian Estimator on training as well as test data.
* A high-level idea behind this is that when we see a particular word which is not present in a class but is present in other classes, for MLE, we assume that the probability of that work occurring in that class is always 0, which is true for the training data set but is not for every other test data sets. In case of BE, we take the weighted probability of that word occurring in that particular class, even if the word is not present in that class for the training data set. This improves the performance or the accuracy of the classifier when we use the Bayesian Estimator. Thus, the Bayesian Estimator is better!
   
   
