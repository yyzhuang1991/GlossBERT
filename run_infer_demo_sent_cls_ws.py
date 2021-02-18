import pandas as pd
from nltk import wordnet as wn
import argparse
from tokenization import BertTokenizer
from modeling import BertForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_context_gloss_pairs(input, target_start_id, target_end_id, lemma):
    """
    construct context gloss pairs like sent_cls_ws
    :param input: str, a sentence
    :param target_start_id: int
    :param target_end_id: int
    :param lemma: lemma of the target word
    :return: candidate lists
    """
    sent = input.split(" ")
    assert 0 <= target_start_id and target_start_id < target_end_id  and target_end_id <= len(sent)
    target = " ".join(sent[target_start_id:target_end_id])
    if len(sent) > target_end_id:
        sent = sent[:target_start_id] + ['"'] + sent[target_start_id:target_end_id] + ['"'] + sent[target_end_id:]
    else:
        sent = sent[:target_start_id] + ['"'] + sent[target_start_id:target_end_id] + ['"']

    sent = " ".join(sent)
    lemma = lemma


    sense_data = pd.read_csv("./wordnet/index.sense.gloss",sep="\t",header=None).values
    d = dict()
    for i in range(len(sense_data)):
        s = sense_data[i][0]
        pos = s.find("%")
        try:
            d[s[:pos + 2]].append((sense_data[i][0],sense_data[i][-1]))
        except:
            d[s[:pos + 2]]=[(sense_data[i][0], sense_data[i][-1])]

    # print(len(d))
    # print(len(d["happy%3"]))
    # print(d["happy%3"])

    candidate = []
    for category in ["%1", "%2", "%3", "%4", "%5"]:
        query = lemma + category
        try:
            sents = d[query]
            for sense_key, gloss in sents:
                candidate.append((sent, f"{target} : {gloss}", target, lemma, sense_key, gloss))
        except:
            pass
    if len(candidate) == 0:
        return None
    # assert len(candidate) != 0, f'there is no candidate sense of "{lemma}" in WordNet, please check'
    # print(f'there are {len(candidate)} candidate senses of "{lemma}"')


    return candidate


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_to_features(candidate, tokenizer, max_seq_length=512):

    candidate_results = []
    features = []
    for item in candidate:
        text_a = item[0] # sentence
        text_b = item[1] # gloss
        candidate_results.append((item[-2], item[-1])) # (sense_key, gloss)


        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))


    return features, candidate_results



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



def infer(model, tokenizer, input, target_start_id, target_end_id, lemma, args):

    print(f"input: {input}\nlemma: {lemma}")
    examples = construct_context_gloss_pairs(input, target_start_id, target_end_id, lemma)
    if examples is None:
        return None
    eval_features, candidate_results = convert_to_features(examples, tokenizer)
    input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)


    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
    logits_ = F.softmax(logits, dim=-1)
    logits_ = logits_.detach().cpu().numpy()
    output = np.argmax(logits_, axis=0)[1]
    # print(f"results:\nsense_key: {candidate_results[output][0]}\ngloss: {candidate_results[output][1]}")
    sense_key = candidate_results[output][0]
    gloss = candidate_results[output][1]
    return sense_key, gloss


def infer_for_examples(sentences, target_starts, target_ends, lemmas, args):
    # sentences: list of str, each str should have already been tokenized and concat by space
    tokenizer = BertTokenizer.from_pretrained(model_folder, do_lower_case=True)
    label_list = ["0", "1"]
    num_labels = len(label_list)
                                     
    model = BertForSequenceClassification.from_pretrained(model_folder,  num_labels=num_labels)
    model.to(device)
    model.eval()

    sense_keys = []
    glosses = []
    
    for sent_str, target_start_id, target_end_id, lemma in zip(sentences, target_starts, target_ends, lemmas):
        item = infer(model, tokenizer, sent_str, target_start_id, target_end_id, lemma, args)
        if item is None: # no matched word
            sense_key = None
            gloss = None 
        else:
             sense_key, gloss = item
        sense_keys.append(sense_key)
        glosses.append(gloss)
    return sense_keys, glosses



def load_glossbert(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    label_list = ["0", "1"]
    num_labels = len(label_list)
                                     
    model = BertForSequenceClassification.from_pretrained(args.bert_model,  num_labels=num_labels)
    
    model.to(device)
    model.eval()
    return model, tokenizer

if  __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("--bert_model", default="./Sent_CLS_WS", type=str)
    parser.add_argument("--bert_model", default="/uusoc/exports/scratch/yyzhuang/glossbert/checkpoint/", type=str)
    
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")

    args = parser.parse_args()

    input = "I went to the bank and talked with the banker"
    target_start_id = 4
    target_end_id = 5
    lemma = "bank"
    model, tokenizer = load_glossbert(args)

    sense_key, gloss = infer(model, tokenizer, input, target_start_id, target_end_id, lemma, args)
    print(f"output sense key: {sense_key}, gloss = {gloss}")
