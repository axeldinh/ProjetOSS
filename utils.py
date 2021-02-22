import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd
import time
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_model(directory='DialoGTP_Oss_100Epoch'):
    
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelWithLMHead.from_pretrained(directory)
    
    return model, tokenizer

def get_answer(model, tokenizer, sentence, device, max_length = 100):
    model.eval()
    input_ids = tokenizer.encode(sentence + tokenizer.eos_token, return_tensors='pt').to(device)
    answer_ids = model.generate(input_ids, eos_token_id = tokenizer.eos_token_id,
                              pad_token_id = tokenizer.eos_token_id, max_length = max_length)[0][input_ids.size(-1):]
    return tokenizer.decode(answer_ids, skip_special_tokens=True)

def conversation(model, tokenizer, max_length, device):
  
    conversation = ''
    while tokenizer.encode(conversation, return_tensors='pt').size(-1) < max_length:
        sentence = input('Vous : ') + tokenizer.eos_token
        conversation = conversation + ' ' + sentence
        answer = get_answer(model, tokenizer, conversation, device, max_length)
        print('Oss : {}'.format(answer))
        conversation = conversation + answer + tokenizer.eos_token
    print('The conversation has exceeded the max_length parameter.')
    
    return conversation

def conversation_with_himself(model, tokenizer, first_sentence, max_length, device):
  
    conversation = first_sentence + tokenizer.eos_token
    answer = get_answer(model, tokenizer, conversation, device, max_length)

    print('Oss1: {}'.format(sentence))
    print('Oss2: {}'.format(answer))

    while tokenizer.encode(conversation, return_tensors='pt').size(-1) < max_length:
    
        answer1 = get_answer(model, tokenizer, conversation, device, max_length)
        print('Oss1: {}'.format(answer1))
        conversation = conversation + ' ' + answer1 + tokenizer.eos_token
        answer2 = get_answer(model, tokenizer, conversation, device, max_length)
        print('Oss2: {}'.format(answer2))
        conversation = conversation + ' ' + answer2 + tokenizer.eos_token
  
    print('The conversation has exceeded the max_length parameter.')

    return conversation

def get_oss_answer(model, tokenizer, sentence, refs_tfidf, vectorizer, References, device, max_length = 100, print_true_answer = False):

    answer = get_answer(model, tokenizer, sentence, device, max_length)
    answer_tfidf = vectorizer.transform([answer])

    if print_true_answer:
        print(answer)

    best_similarity = 0
    best_line_id = 0

    for i, line_tfidf in enumerate(refs_tfidf):

        similarity = cosine_similarity(line_tfidf, answer_tfidf)
        if similarity > best_similarity:
            best_similarity = similarity
            best_line_id = i

    return best_line_id, References.iloc[best_line_id].Line
  
def get_refs_tfidf_vectorizer(References, tokenizer):
    vectorizer = TfidfVectorizer(tokenizer = tokenizer.tokenize)
    
    return vectorizer, vectorizer.fit_transform(References.Line)