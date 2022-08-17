
import json
import re
import os
import nltk
nltk.download('words')
import emoji
import shutil
words = set(nltk.corpus.words.words())
path = '/data/'
import numpy as np


def preprocess (path):
    for dir_ in os.listdir(path):
        dir_=path+dir_
        extract_text(dir_)

def extract_text(path_to_json):
  data=json.load(open(path_to_json))
  ent=data[0]['class']
  if not os.path.exists(os.path.join(path,ent)):
    os.makedirs(os.path.join(path,ent))
  path = os.path.join(path,ent)
  for i,dat in enumerate(data):
    with open(os.path.join(path,str(i)+'.txt'),'w') as f:
          f.write( process_text(dat['full_text']))

def process_text(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign 
    # tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
    #         if w.lower() in words or not w.isalpha())
    return tweet

def extract_tweet_into_text(path):
    for files in os.listdir(path):
        if files not in ['train','test','val'] and 'tweet' not in files:
            print(files)
            files_full=os.path.join(path,files)
            list_of_file = [x for x in os.listdir(files_full) if x.endswith('txt')]
            np.random.shuffle(list_of_file)
            count = 0
            for i,txt in enumerate(list_of_file):
                if count<7000:
                    if not os.path.exists(os.path.join(os.path.join(path,'train'),files)):
                        os.makedirs(os.path.join(os.path.join(path,'train'),files))
                        target = os.path.abspath(os.path.join(os.path.join(path,'train'),files))
                elif count>=7000 and count<8500:
                    if not os.path.exists(os.path.join(os.path.join(path,'val'),files)):
                        os.makedirs(os.path.join(os.path.join(path,'val'),files))
                        target = os.path.abspath(os.path.join(os.path.join(path,'val'),files))
                elif count>=8500 and count<10000:
                    if not os.path.exists(os.path.join(os.path.join(path,'test'),files)):
                        os.makedirs(os.path.join(os.path.join(path,'test'),files))
                        target = os.path.abspath(os.path.join(os.path.join(path,'test'),files))
                else:
                    break
                source = os.path.join(files_full,txt)
                destination = os.path.join(target,txt)
                shutil.copy(source, destination)
                count=count+1
