import warnings
warnings.filterwarnings("ignore")

import pandas as pd

import nltk
from nltk.corpus import stopwords
import emoji
import re

nltk.download('stopwords')
STOPWORDS = list(set(stopwords.words('english')))
STOPWORDS += [".", "!", "?", ",", ";", ":", "[", "]", "{", "}", "-", "+", 
    "_", "/", "#", "@", "$", "%", "^", "&", "*", "(", ")", "<", ">", "|", "=",
    ".-", ".,", "'", '"', ',"', ".>", ".<"]

def preprocessing(df, lowercase=False, stopwords=False, links=False, tags=False, numbers=False, emojis=False, hashtag=False,
                 rt=False):
    
    new_df = df.copy()
    text = new_df['text']
    
    # lowercasing everything
    if lowercase:
        text = text.apply(lambda x: str.lower(x))
    
    # removing stopword
    if stopwords:
        # we have to look at the lowercase words, since the stopwords are lowercase
        text = text.apply(lambda x: " ".join([word for word in x.split() if str.lower(word) not in STOPWORDS]))
    
    # removing links
    if links:
        text = text.apply(lambda x: " ".join([word for word in x.split() if 'http' not in word]))
    
    # removing tags
    if tags:
        text = text.apply(lambda x: " ".join([word for word in x.split() if '@' not in word]))
    
    # removing numbers only if the whole word is numeric - eg. we remove 1123 but not 1123a
    if numbers:
        text = text.apply(lambda x: " ".join([word for word in x.split() if not word.isnumeric()]))
    
    # removing emojis (whole word if it contains an emoji)
    if emojis:
        text = text.apply(lambda x: " ".join([word for word in x.split() if not any(i in word for i in emoji.EMOJI_DATA)]))
    
    # removing hashtags
    if hashtag:
        text = text.apply(lambda x: " ".join([word for word in x.split() if '#' not in word]))
        
    # removing rt from the beginning
    if rt:
        text = text.apply(lambda x: " ".join([word for i,word in enumerate(x.split()) if not (i==0 and str.lower(word)=='rt')]))
    
    new_df['text'] = text
    
    return new_df