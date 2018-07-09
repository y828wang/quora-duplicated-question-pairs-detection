# Classify Duplicated Quora Question Pairs

[The Kaggle competition](https://www.kaggle.com/c/quora-question-pairs)

* [deep_learning/](deep_learning): Deep neural networks which extract feature vectors from Quora questions.
  * [cnn.py](deep_learning/cnn.py) is a convolutional neural network. 
  * [lstm.py](deep_learning/cnn.py) is a recurrent neural network. 
* [feature_engineering/](feature_engineering): 60 hand crafted features on a pair of documents, Quora questions in this case. [xgboost decision trees](feature_engineering/feat_xgboost.py) ingests the features and classify whether the two questions are similiar. [plots.ipynb](plots.ipynb) shows the feature importance of trained xgboost trees. 
* [matrix_fact/](matrix_fact): Matrix Factorization based models to extract feature vectors from Quora questions, such as LSA.
