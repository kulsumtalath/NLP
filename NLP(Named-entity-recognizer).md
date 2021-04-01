```python
import nltk
```


```python
import re
import string
```


```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
```


```python
import nltk
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\Kulsum\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
import nltk
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\Kulsum\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    




    True




```python
def rem_stopwords(text):
    stop_words=set(stopwords.words("english"))
    word_tokens=word_tokenize(text)
    filtered_text=[word for word in word_tokens if word not in stop_words]
    return filtered_text
example_text="This is a sample sentence and we are going to remove the stop words"
rem_stopwords(example_text)
    
```




    ['This', 'sample', 'sentence', 'going', 'remove', 'stop', 'words']



# Stemmed words


```python
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer=PorterStemmer()
def stem_words(text):
    word_tokens=word_tokenize(text)
    stems=[stemmer.stem(word) for word in word_tokens]
    return stems
text="friend friends friended friendly scientific methods"
stem_words(text)
```




    ['friend', 'friend', 'friend', 'friendli', 'scientif', 'method']



# Lemmatizer


```python
import nltk
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\Kulsum\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    




    True




```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer=WordNetLemmatizer()
def lemmatize_words(text):
    word_tokens=word_tokenize(text)
    lemmas=[lemmatizer.lemmatize(word) for word in word_tokens]
    return lemmas
text="data science uses scientific methods algorithms and many types of processes"
lemmatize_words(text)
```




    ['data',
     'science',
     'us',
     'scientific',
     'method',
     'algorithm',
     'and',
     'many',
     'type',
     'of',
     'process']



# partsofspeech


```python
import nltk
nltk.download('averaged_perceptron_tagger')
```

    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     C:\Users\Kulsum\AppData\Roaming\nltk_data...
    [nltk_data]   Package averaged_perceptron_tagger is already up-to-
    [nltk_data]       date!
    




    True




```python
from nltk.tokenize import word_tokenize
from nltk import pos_tag
```


```python
def pos_tagging(text):
    word_token=word_tokenize(text)
    return pos_tag(word_token)
pos_tagging("You just gave me a chocolate")
```




    [('You', 'PRP'),
     ('just', 'RB'),
     ('gave', 'VBD'),
     ('me', 'PRP'),
     ('a', 'DT'),
     ('chocolate', 'NN')]



In the given example, PRP stands for personal pronoun, RB for adverb, VBD for verb past tense, DT for determiner and NN for noun.

# chunking


```python
def chunking(text,grammar):
    word_tokens=word_tokenize(text)
    word_pos=pos_tag(word_tokens)
    chunkParser=nltk.RegexpParser(grammar)
    tree=chunkParser.parse(word_pos)
    for subtree in tree.subtrees():
        print(subtree)
    tree.draw()
sentence = 'the little yellow bird is flying in the sky'
grammar = "NP: {<DT>?<JJ>*<NN>}"
chunking(sentence, grammar)    
```

    (S
      (NP the/DT little/JJ yellow/JJ bird/NN)
      is/VBZ
      flying/VBG
      in/IN
      (NP the/DT sky/NN))
    (NP the/DT little/JJ yellow/JJ bird/NN)
    (NP the/DT sky/NN)
    

In the given example, grammar, which is defined using a simple regular expression rule. This rule says that an NP (Noun Phrase) chunk should be formed whenever the chunker finds an optional determiner (DT) followed by any number of adjectives (JJ) and then a noun (NN).

# Name Entity Recognizer


```python
import nltk
nltk.download('maxent_ne_chunker')
```

    [nltk_data] Downloading package maxent_ne_chunker to
    [nltk_data]     C:\Users\Kulsum\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping chunkers\maxent_ne_chunker.zip.
    




    True




```python
import nltk
nltk.download('words')
```

    [nltk_data] Downloading package words to
    [nltk_data]     C:\Users\Kulsum\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping corpora\words.zip.
    




    True




```python
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
from nltk import pos_tag
```


```python
def named_entity_recognizer(text):
    word_token=word_tokenize(text)
    word_pos=pos_tag(word_token)
    print(ne_chunk(word_pos))
text="Raj works for Amazon so he went ot Bangalore"
named_entity_recognizer(text)
```

    (S
      (GPE Raj/NNP)
      works/VBZ
      for/IN
      (PERSON Amazon/NNP)
      so/RB
      he/PRP
      went/VBD
      ot/PRP
      Bangalore/NNP)
    


```python

```


```python

```
