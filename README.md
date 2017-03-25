# Babel_SGD
Data Analysis (ML) on Babel data set

Some code to do various ML techniques on a phone based data set (speech recognition) (360 feature data set).

Techniques shown include:

1)	SGD with two sets of beta parameters, one for mapping down to the 42 phones, and the other for mapping to the 1000 states.

2)	A SGD tree where we first map to one of the 42 phones, and then map to one of the states associated with that specific phone. This was done by running SGD independently for each of the 42 phones, as well as at a top level to map from the features to the 42 phones, and then combining these 42 + 1 sets of parameters to generate log likelihoods.

3)	Nearest neighbours for the PCA'ed data. Generally whilst the LSH approach is fast, there is quite a big tradeoff between speed, and accuracy in results.
