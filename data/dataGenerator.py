"""
Module contains methods to generate dataset for training and testing epidemiological corpus taggers found in models.py

"""
import pandas as pd
from datasets import Dataset, load_dataset
import re


__author__ = "Edmond Menya"
__email__ = "edmondmenya@gmail.com"

def data_generator(rel_url,irr_url,corpus_type):
    """
    Crawls data folder for corpus
    :param rel_url: location of relevant corpus folder
    :param irr_url: location of irrelevant corpus folder
    :return: Shuffled dataframe containing both corpus and label in two columns
    """

    relevant_data=pd.read_csv(rel_url)
    irrelevant_data=pd.read_csv(irr_url)

    #Label Corpus in data
    relevant_data['Label']='relevant'
    irrelevant_data['Label']='irrelevant'

    #Select Corpus and Label from data
    relevant_data=relevant_data[[corpus_type,'Label']]
    irrelevant_data=irrelevant_data[[corpus_type,'Label']]

    #Drop Last Empty Columns
    relevant_data.drop(relevant_data.tail(1).index,inplace=True)
    irrelevant_data.drop(irrelevant_data.tail(1).index,inplace=True)

    #Merge and Shuffle Data
    data=pd.concat([relevant_data,irrelevant_data],ignore_index=True)
  
    return data.sample(frac=1).reset_index(drop=True)

def text_cleaner(corpus):
    punctuation = '!"#$%&()*+-<=>?@[\\]^_`{|}~©'

    corpus = corpus.apply(lambda x: re.sub(r'http\S+', '', x)) #remove urls
    corpus = corpus.apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation))) #remove punctuation marks
    corpus = corpus.str.lower() #convert all words to lowercase
    corpus = corpus.apply(lambda x:' '.join(x.split())) #remove whitespaces

    return corpus
  
  
def tok_with_labels(doc, label, tokenizer):
    """
    Tokenizes corpus content using supplied pretrained tokenizer
    :param doc: corpus to be tokenized
    :param label: corpus class label
    :param tokenizer: object of pretrained tokenizer
    :return: tokenized corpus, corpus class label in a list
    """
  
    tok_doc=tokenizer.tokenize(doc)

    return tok_doc, [label]

def preprocess_function(doc):
    """
    Tokenizes corpus content using supplied pretrained tokenizer
    :param doc: corpus to be tokenized
    :param tokenizer: object of pretrained tokenizer
    :return: tokenized corpus, corpus class label in a list
    """
    label = doc["score"] 
    tok_doc = tokenizer(doc["text"], truncation=True, padding="max_length", max_length=256)
    
    tok_doc["labels"] = [float(s) for s in label]

    return tok_doc


def corpus_tok(dataset, tokenizer):
    tokenizer = tokenizer
    tokenized_datasets = dataset.map(preprocess_function, remove_columns=["id", "uuid", "text", "score"], batched=True)

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8)
    test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=8)

    return train_dataloader, test_dataloader

