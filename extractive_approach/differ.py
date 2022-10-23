from typing import List, Any, Callable, Tuple, Union
from itertools import zip_longest
import re
import difflib
import html
import numpy as np
import pandas as pd
from datasets import list_metrics, load_metric
from collections import defaultdict
from itertools import chain
import torch
from tqdm import tqdm


try:
    from IPython.display import display
    from IPython.display import HTML
except:
    print("No IPython detected, cannot show diffs using show_diffs()")


Token = str
TokenList = List[Token]
whitespace = re.compile('\s+')
end_sentence = re.compile('[.!?]\s+')

def tokenize(s:str) -> TokenList:
    '''Split a string into tokens'''
    return whitespace.split(s)


def untokenize(ts:TokenList) -> str:
    '''Join a list of tokens into a string'''
    return ' '.join(ts)


def sentencize(s:str) -> TokenList:
    '''Split a string into a list of sentences'''
    return end_sentence.split(s)


def unsentencise(ts:TokenList) -> str:
    '''Join a list of sentences into a string'''
    return '. '.join(ts)


def html_unsentencise(ts:TokenList) -> str:
    '''Joing a list of sentences into HTML for display'''
    return ''.join(f'<p>{t}</p>' for t in ts)


def mark_text(text:str) -> str:
    return f'<span style="color: red;">{text}</span>'


def mark_span(text:TokenList) -> TokenList:
    if len(text) > 0:
        text[0] = '<span style="background: #69E2FB;">' + text[0]
        text[-1] += '</span>'
    return text


def markup_diff(a:TokenList, b:TokenList,
                mark:Callable[[TokenList], TokenList]=mark_span,
                default_mark: Callable[[TokenList], TokenList] = lambda x: x,
                isjunk:Union[None, Callable[[Token], bool]]=None) -> Tuple[TokenList, TokenList]:
    """Returns a and b with any differences processed by mark

    Junk is ignored by the differ
    """
    seqmatcher = difflib.SequenceMatcher(isjunk=isjunk, a=a, b=b, autojunk=False)
    out_a, out_b = [], []
    for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
        markup = default_mark if tag == 'equal' else mark
        out_a += markup(a[a0:a1])
        out_b += markup(b[b0:b1])
    assert len(out_a) == len(a)
    assert len(out_b) == len(b)
    return out_a, out_b


def align_seqs(a: TokenList, b: TokenList, fill:Token='') -> Tuple[TokenList, TokenList]:
    out_a, out_b = [], []
    seqmatcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
        delta = (a1 - a0) - (b1 - b0)
        out_a += a[a0:a1] + [fill] * max(-delta, 0)
        out_b += b[b0:b1] + [fill] * max(delta, 0)
    assert len(out_a) == len(out_b)
    return out_a, out_b


def diff_ratio(a: str, b: str, fill:Token='',       
               isjunk:Union[None, Callable[[Token], bool]]=None) -> float:
    ratios = []
    redundant_tokens = []
    total_tokens = []
    for sent_a, sent_b in zip_longest(*align_seqs(sentencize(a), sentencize(b))):
        tok_sent_a, tok_sent_b = tokenize(sent_a), tokenize(sent_b)
        sent_ratio = difflib.SequenceMatcher(isjunk=isjunk, a=tok_sent_a, 
                                             b=tok_sent_b, autojunk=False).ratio()
        ratios.append(sent_ratio)
        total_tokens.append(len(tok_sent_a))
        # inverse of ratio calc as shown: 
        # https://docs.python.org/3.7/library/difflib.html#difflib.SequenceMatcher.ratio
        redundant_tokens.append(sent_ratio / 2 * (len(tok_sent_a) + len(tok_sent_b)))
        
    return sum(ratios) / len(ratios), np.std(ratios), max(ratios), min(ratios), sum(redundant_tokens), sum(total_tokens)


def compute_sequential_diff_metrics(notes: pd.DataFrame, sort_cols=['hadm_id', 'chartdate', 'charttime', 'category', 'description'], groupby_cols=['hadm_id', 'category', 'description']) -> pd.DataFrame:
    """
    Given a DataFrame of notes (with mimic3 formatted columns) compute token, rouge, and bert-score sequential diffs.
    """
    internote_sort_cols = sort_cols
    internote_group_cols = groupby_cols
    results = defaultdict(list)
    import os;print(os.getcwd())
    rouge_metric = load_metric('./rouge.py')
    bert_metric = load_metric('./bertscore.py')
    print(notes.columns)
    notes = notes.sort_values(sort_cols)
    for k, g in tqdm(notes.groupby(internote_group_cols)):
        if g.shape[0] > 1:
            ratios, std_devs, max_ratios, min_ratios, redundant_toks, total_tokens = [], [], [], [], [], []
            for i in range(len(g.text.tolist()) - 1):
                ratio, std_dev, max_ratio, min_ratio, redundant_toks_len, total_toks = \
                    diff_ratio(g.text.iloc[i], g.text.iloc[i+1])
                rouge_metric.add(prediction=g.text.iloc[i+1], reference=g.text.iloc[i])
                bert_metric.add(prediction=g.text.iloc[i+1], reference=g.text.iloc[i])
                ratios.append(ratio)
                std_devs.append(std_dev)
                max_ratios.append(max_ratio)
                min_ratios.append(min_ratio)
                redundant_toks.append(redundant_toks_len)
                total_tokens.append(total_toks)
            results['diff_ratios'].append(ratios)
            results['max_ratios'].append(max(max_ratios))
            results['min_ratios'].append(min(min_ratios))
            results['redundant_toks'].append(sum(redundant_toks))
            results['total_tokens'].append(sum(total_tokens))
            results['avg_diff_ratio'].append(sum(ratios) / len(ratios))
            txt_lens = g.text.apply(len)
            results['avg_txt_len'].append(sum(txt_lens) / len(txt_lens))
            results['hadm_id'].append(k[0])
            results['category'].append(k[1])
            results['description'].append(k[2] if len(k) == 1 else '')
            # compute batched summarisation metrics
            rougeL = rouge_metric.compute(rouge_types=['rougeL'], use_agregator=False)['rougeL']
            try:
                bert_scores = bert_metric.compute(lang='en', rescale_with_baseline=True, model_type='./xlnet-base-cased')
                _compute_bert_score_stats(bert_scores, 'recall', results)
                _compute_bert_score_stats(bert_scores, 'precision', results)
            except Exception as e:
                # Failure due to cuda memory..
                print(f"Failure computing bert-score for group {k}: {e}")
                _add_nan_bert_scores('precision', results)
                _add_nan_bert_scores('recall',  results)
            _compute_rouge_stats(rougeL, 'recall', results)
            _compute_rouge_stats(rougeL, 'precision', results)
            torch.cuda.empty_cache()
           
    return pd.DataFrame(results)

def _compute_rouge_stats(scores: list, prop: str, results: dict):
    measure = [getattr(l, prop) for l in scores]
    results[f'rg_{prop}'].append(measure)
    results[f'rg_{prop}_avg'].append(np.average(measure))
    results[f'rg_{prop}_med'].append(np.median(measure))
    results[f'rg_{prop}_iqr'].append(np.subtract(*np.percentile(measure, [75, 25])))

def _compute_bert_score_stats(scores: list, prop: str, results):
    results[f'bs_{prop}'].append(scores[prop].detach().cpu().numpy())
    results[f'bs_{prop}_avg'].append(np.average(scores[prop]))
    results[f'bs_{prop}_med'].append(np.median(scores[prop]))
    results[f'bs_{prop}_iqr'].append(np.subtract(*np.percentile(scores[prop], [75, 25])))

def _add_nan_bert_scores(prop: str, results: dict):
    results[f'bs_{prop}'].append(np.nan)
    results[f'bs_{prop}_avg'].append(np.nan)
    results[f'bs_{prop}_med'].append(np.nan)
    results[f'bs_{prop}_iqr'].append(np.nan)
    

def split_notes(notes, splitter='------\n------\n------\n'):
    care_notes = defaultdict(list)
    discharge_notes = defaultdict(list)
    for adm, df in notes.sort_values(['hadm_id', 'chartdate', 'charttime']).groupby('hadm_id'):
        dis_notes = df[(df.category == 'Discharge summary') & (df.description != 'Addendum')].text.tolist()
        care_df = df[df.category != 'Discharge summary']
        if len(dis_notes) == 0 or care_df.shape[0] == 0:
            continue

        hadm_id = care_df.hadm_id.iloc[0]
        # pick first as there are rarely 2 summaries??

        discharge_notes['hadm_id'].append(hadm_id)
        discharge_notes['text'].append(dis_notes[0])
        # ignore addendums - as not sure what section they are in.

        care_notes['hadm_id'].append(hadm_id)
        care_notes['text'].append(care_df.text.str.cat(sep=splitter))
        care_notes['first_time'].append(care_df.iloc[0].charttime or care_df.iloc[0].chartdate)
        care_notes['last_time'].append(care_df.iloc[-1].charttime or care_df.iloc[-1].chartdate)
        care_notes['categories'].append(care_df.category.tolist())
        care_notes['descriptions'].append(care_df.description.tolist())
        care_notes['icd_code'].append(care_df.icd9_code.iloc[0])
    care_notes_df = pd.DataFrame(care_notes)
    discharge_notes_df = pd.DataFrame(discharge_notes)
    return care_notes_df, discharge_notes_df


def compute_avgs(results_df: pd.DataFrame) -> pd.DataFrame:
    cat_desc_avg = defaultdict(list)
    # remove where no bert_score is found
    results_df = results_df[~pd.isna(results_df.bs_recall)]
    for k, df in results_df.groupby(['category', 'description']):
        cat_desc_avg['cat_desc'].append(f'{k[0]}:{k[1]}')
        cat_desc_avg['redundant_toks'].append(sum(df.redundant_toks))
        cat_desc_avg['total_toks'].append(sum(df.total_tokens))
        cat_desc_avg['avg_txt_len'].append(np.average(df.avg_txt_len))
        cat_desc_avg['macro_avg'].append(np.average(df.avg_diff_ratio))
        
        d_r = list(chain.from_iterable(df.diff_ratios))
        cat_desc_avg['micro_avg'].append(np.average(d_r))
        cat_desc_avg['num_instances'].append(len(d_r))
        # micro avgs of median / iqr
        # rg avgs
        rg_rec = list(chain.from_iterable(df.rg_recall))
        rg_prec = list(chain.from_iterable(df.rg_precision))
        cat_desc_avg['rg_rec_avg'].append(np.average(rg_rec))
        cat_desc_avg['rg_rec_med'].append(np.median(rg_rec))
        cat_desc_avg['rg_rec_iqr'].append(np.subtract(*np.percentile(rg_rec, [75, 25])))
        cat_desc_avg['rg_prec_avg'].append(np.average(rg_prec))
        cat_desc_avg['rg_prec_med'].append(np.median(rg_prec))
        cat_desc_avg['rg_prec_iqr'].append(np.subtract(*np.percentile(rg_prec, [75, 25])))
        # bs avgs
        bs_rec = list(chain.from_iterable(df.bs_recall))
        bs_prec = list(chain.from_iterable(df.bs_precision))
        cat_desc_avg['bs_rec_avg'].append(np.average(bs_rec))
        cat_desc_avg['bs_rec_med'].append(np.median(bs_rec))
        cat_desc_avg['bs_rec_iqr'].append(np.subtract(*np.percentile(bs_rec, [75, 25])))
        cat_desc_avg['bs_prec_avg'].append(np.average(bs_prec))
        cat_desc_avg['bs_prec_med'].append(np.median(bs_prec))
        cat_desc_avg['bs_prec_iqr'].append(np.subtract(*np.percentile(bs_prec, [75, 25])))
        
    group_avgs = pd.DataFrame(cat_desc_avg)
    group_avgs = group_avgs[~group_avgs.cat_desc.str.contains('Discharge summary')]
    group_avgs = group_avgs[group_avgs['num_instances'] > 5]
    group_avgs = group_avgs.sort_values('num_instances', ascending=False).reset_index(drop=True).head(20)
    return group_avgs


def html_sidebyside(a, b):
    # Set the panel display
    out = '<div>'
    for left, right in zip_longest(a, b, fillvalue=''):
        out += f'<div style="display: inline-block; width: calc(50% - 10px)">{left}</div>'
        out += '<div style="width: 20px; display: inline-block"></div>'
        out += f'<div style="display: inline-block; width: calc(50% - 10px)">{right}</div>'
    out += '</div>'
    return out

def html_diffs(a, b):
    a = html.escape(a)
    b = html.escape(b)

    out_a, out_b = [], []
    for sent_a, sent_b in zip(*align_seqs(sentencize(a), sentencize(b))):
        mark_a, mark_b = markup_diff(tokenize(sent_a), tokenize(sent_b))
        out_a.append(untokenize(mark_a))
        out_b.append(untokenize(mark_b))

    return html_sidebyside(out_a, out_b)


def show_diffs(a, b):
    display(HTML(html_diffs(a,b)))