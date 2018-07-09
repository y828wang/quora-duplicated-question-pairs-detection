#!/usr/bin/env bash

sudo apt-get update
sudo apt-get install build-essential
sudo apt-get -y install python3-pip
sudo -H pip3 install --upgrade pip

sudo -H pip3 install gensim
sudo -H pip3 install nltk
sudo -H pip3 install scikit-learn
sudo -H pip3 install pandas
sudo -H pip3 install numpy
sudo -H pip3 install fuzzywuzzy
sudo -H pip3 install python-Levenshtein
sudo -H pip3 install Cython

sudo python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# install fastFM
#cd /output/fastFM; pip3 install -r ./requirements.txt; sudo pip3 install .

# install xgboost
sudo cd /allen/xgboost/python-package; sudo python3 setup.py install
sudo echo 'export PYTHONPATH=/output/xgboost/python-package' >> ~/.bashrc
#sudo echo 'export PYTHONPATH=/output/fastFM/fastFM' >> ~/.bashrc

pip3 install tensorflow-gpu

