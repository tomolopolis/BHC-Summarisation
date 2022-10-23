import pickle
import argparse
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from datasets import load_metric

metric = load_metric('rouge')

tqdm.pandas()

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', choices=('wv', 'sbert'))
parser.add_argument('-d', '--dataset', choices=('train', 'val', 'test'))
parser.add_argument('-sl', '--sent_limits', default="1,2,3,5,10,15")


def main():
    args = parser.parse_args()
    sent_limits = [int(l) for l in args.sent_limits.split(',')]
                    
    if args.model == 'sbert':
        model = sent_bert_textrank
    elif args.model == 'wv':
        model = wv_textrank
    
    if args.dataset == 'train':
        docs = pd.read_json('mimic_summarisation/mimic_3_train.json', lines=True).text
        ref_sums = pickle.load(open('train_ref_summmaries.pickle', 'rb'))
    elif args.dataset == 'val':
        docs = pd.read_json('mimic_summarisation/mimic_3_val.json', lines=True).text
        ref_sums = pickle.load(open('val_ref_summaries.pickle', 'rb'))
    elif args.dataset == 'test':
        docs = pd.read_json('mimic_summarisation/mimic_3_test.json', lines=True).text
        ref_sums = pickle.load(open('test_ref_summaries.pickle', 'rb'))
    
    #docs = docs[0:3] 
    sent_limits_gen_sums = model(docs, ref_sums, sent_limits)
    scores = {}
    for (limit_gen, gen_sums), (limit_ref, ref_sums) in zip(sent_limits_gen_sums.items(), ref_sums.items()):
        scores[limit_gen] = rouge_score(gen_sums, ref_sums)
    
    print(scores)
    json.dump(scores, open(f'outputs/{args.model}_{args.dataset}_scores.json', 'w'))

    
def wv_textrank(docs, refs, sent_limits):
    nlp = spacy.load('en_core_web_md')
    nlp.add_pipe('textrank')
    sent_limits_gen_sums = defaultdict(list)
    for doc in tqdm(docs):
        doc = nlp(doc)
        for limit in sent_limits:
            sent_limits_gen_sums[limit].append(''.join(sent.text for sent in doc._.textrank.summary(limit_phrases=15, limit_sentences=limit)))
    return sent_limits_gen_sums


def sent_bert_textrank(docs: dict, refs, sent_limits):
    errors = []
    sent_limits_gen_sums = defaultdict(list)
    docs = docs.str.replace('\n\n' ,'\n').str.replace(r'\s{2+}', ' ').str.replace(r'\t', ' ')
    nlp = spacy.load('en_core_web_md')
    model = SentenceTransformer('all-MiniLM-L12-v2')
    for i, doc in enumerate(tqdm(docs)):
        # checkpoint every 100 docs, the generated_sums
        if i % 200 == 0:
            print(f'saving generated sums: iter:{i}')
            pickle.dump(sent_limits_gen_sums, open('outputs/checkpoint_sbert_generated_sums.pickle', 'wb'))
        doc = nlp(doc)
        sents = [sent.text for sent in doc.sents if len(sent.text) > 3]
        vecs = [model.encode(sent) for sent in sents]
        vecs = np.array(vecs)
        cs_mtx = cosine_similarity(vecs, vecs)
        zero_ident_mtx = np.ones(cs_mtx.shape, int)
        np.fill_diagonal(zero_ident_mtx, 0)
        cs_mtx = cs_mtx * zero_ident_mtx
        graph = nx.from_numpy_array(cs_mtx)
        try:
            nx.pagerank(graph, max_iter=500)
            ranked_sents = pd.Series(nx.pagerank(graph, max_iter=500)).sort_values(ascending=False).index.tolist()
            for limit in sent_limits:
                gen_sum = ''.join([sents[ranked_sents[i]] for i in range(limit)])
                sent_limits_gen_sums[limit].append(gen_sum)
        except:
            errors.append((i, doc.text[0:30]))
            for limit in sent_limits:
                sent_limits_gen_sums[limit].append(None)
    for err in errors:
        print(errors)
    return sent_limits_gen_sums


def _parse_score(lvl, scores):
    return (lvl, scores[lvl].mid.precision, scores[lvl].mid.recall, scores[lvl].mid.fmeasure)


def rouge_score(gens, refs):
    for gen, ref in zip(gens, refs):
        if ref is not None and gen is not None:
            metric.add(prediction=gen, reference=ref)
    scores = metric.compute()
    return _parse_score('rouge1', scores), _parse_score('rouge2', scores), _parse_score('rougeLsum', scores)
            

if __name__ == '__main__':
    main()
