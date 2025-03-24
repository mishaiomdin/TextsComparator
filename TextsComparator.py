import pandas as pd
import numpy as np

from tqdm.auto import tqdm
import re
import collections

import matplotlib.pyplot as plt
import seaborn as sns

# includes Latin, Cyrillic, Greek, -
non_alphabet_re1 = r"""[^a-zA-Z\u00C0-\u024F\u0400-\u04FF\- ]"""

non_alphabet_re2 = r"""[^a-zA-Z\u00C0-\u024F\u0400-\u04FF\-\'’ ]"""


""" WORKING WITH ONE TEXT """

def get_text(from_file, lang):
    with open(from_file + lang + '.txt') as file:
        text = file.read()
    return text

def get_frequency_old(text_file, non_alphabet_re=non_alphabet_re1, lower=True):
    """ Returns table of word frequencies in a text """
    with open(text_file) as file:
        text = file.read()
    
    dictionary = {}
    
    for _line in text.split('\n'):
        line = re.sub(non_alphabet_re, '', _line)
        if lower:
            line = line.lower()
        for word in line.split(' '):
            if word:
                if word not in dictionary:
                    dictionary[word] = 0
                dictionary[word] += 1
    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={0: 'count', 'index': 'word'}, inplace=True)
    df.sort_values('count', ascending=False, inplace=True)
    df = df[df['word'] != '-']
    df.reset_index(drop=True, inplace=True)
    df['hapax_legomenon'] = (df['count'] == 1)
    df['length'] = df['word'].str.len()
    return df

def get_frequency(text_file, non_alphabet_re=non_alphabet_re2, lower=True):
    with open(text_file) as file:
        text = file.read()
    words = text_to_words(text, non_alphabet_re=non_alphabet_re)
    dictionary = {}
    for word in words:
        if word:
            if word not in dictionary:
                dictionary[word] = 0
            dictionary[word] += 1
    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={0: 'count', 'index': 'word'}, inplace=True)
    df.sort_values('count', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_char_frequency(text_file, non_alphabet_re=non_alphabet_re2, lower=True):
    with open(text_file) as file:
        text = file.read()
    chars = text_to_chars(text, non_alphabet_re=non_alphabet_re)
    dictionary = {}
    for char in chars:
        if char not in dictionary:
            dictionary[char] = 0
        dictionary[char] += 1
    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={0: 'count', 'index': 'char'}, inplace=True)
    df.sort_values('count', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def text_to_words(text, to_delete=None, non_alphabet_re=non_alphabet_re2):
    return clean_text(text, to_delete=to_delete, non_alphabet_re=non_alphabet_re).split()

def text_to_chars(text, to_delete=None, non_alphabet_re=non_alphabet_re2):
    return re.sub(non_alphabet_re, '', text)

def clean_text(text, to_delete=None, non_alphabet_re=non_alphabet_re2):
    """ Deletes all punctuation """
    if to_delete:
        text = text.replace(to_delete, '')
    # apostrophes
    text = re.sub(r"(?<![a-zA-Z])’|’(?![a-zA-Z])", '', text) # deletes all ’ those not between letters
    text = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '', text) # deletes all ' those not between letters
    text = re.sub(r"(?<![a-zA-Z])\-|\-(?![a-zA-Z])", '', text) # deletes all - those not between letters
    text = text.replace('\n', ' ')
    text = re.sub(non_alphabet_re, '', text) # ’'- are saved
    text = re.sub(' +', ' ', text).lower()
    return text

def get_ngrams_df(text, n):
    """ Returns a table of ngrams for given n, sorted by frequency """
    ngrams = {}
    words = text.split()
    for i in range(len(words) - n):
        ngram = ' '.join(words[i:i + n])
        if ngram in ngrams:
            ngrams[ngram] += 1
        else:
            ngrams[ngram] = 1
    res = [[k, v, n] for k, v in sorted(ngrams.items(), key=lambda item: -item[1])]
    return pd.DataFrame(res, columns=['ngram', 'count', 'length'])

""" WORKING WITH A GROUP OF TEXTS """

def save_all(from_file, to_file, langs):
    """ Calculates frequencies for each; saves to csv """
    for lang in langs:
        freq = get_frequency(from_file + lang + '.txt')
        freq.to_csv(to_file + lang + '.csv', index=False)

def get_all(from_file, langs):
    """ Returns a dictionary for all word frequencies of languages """
    res = {}
    for lang in langs:
        res[lang] = pd.read_csv(from_file + lang + '.csv')
    return res

def get_all_texts(from_file, langs):
    """ Returns a dictionary for all texts of languages """
    res = {}
    for lang in langs:
        with open(from_file + lang + '.txt') as file:
            res[lang] = file.read()
    return res


""" HEATMAP """

def heatmap(df, save_to, triangle=True, black=True, labels=False, annot=True, fmt='.0f'):
    """ saves a heatmap for given df, triangle by default """
    # setting background color
    if black:
        sns.set(rc={'axes.facecolor':'black', 'figure.facecolor':'black', 'axes.edgecolor': 'black',
                    'xtick.color': 'white', 'ytick.color': 'white'})
    else:
        sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'axes.edgecolor': 'white',
                   'xtick.color': 'black', 'ytick.color': 'black'})
    if triangle:
        mask = np.array([[False] * i + [True] * (len(df) - i) for i in range(len(df))])
        sns.heatmap(df, annot=annot, mask=mask, fmt=fmt, xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(df, annot=annot, fmt=fmt, xticklabels=labels, yticklabels=labels)
    plt.savefig(save_to)
    plt.show()

""" COMPARE SEVERAL / TWO LANGUAGES """

def compare_frequent(freq, *langs, limit=10):
    """ Returns an UNMATCHED table of most frequent words in any number of languages """
    return pd.DataFrame({
        lang: freq[lang]['word'] for lang in langs
    }).head(limit)

def compare_ngrams(n, texts, *langs, limit=10):
    """ Returns an UNMATCHED table of most frequent ngrams in any number of languages """
    return pd.concat(
        [get_ngrams_df(texts[lang], n) for lang in langs]
    ).head(limit)

def check_different_langs(lang1, lang2):
    """ checks that two different languages are given """
    if lang1 == lang2:
        raise Exception("Can't compare a language with itself")

def calculate_intersection(freq, lang1, lang2):
    """ calculates in_lang1 and in_lang2 if not yet; returns set of intersected words """
    check_different_langs(lang1, lang2)
    intersection = set(freq[lang1]['word']).intersection(set(freq[lang2]['word']))
    if f"in_{lang2}" not in freq[lang1].columns:
        freq[lang1][f'in_{lang2}'] = freq[lang1]['word'].apply(lambda x: x in intersection)
    if f"in_{lang1}" not in freq[lang2].columns:
        freq[lang2][f'in_{lang1}'] = freq[lang2]['word'].apply(lambda x: x in intersection)
    return intersection

def get_specifics(freq, lang1, lang2):
    """ returns words specific to lang1, lang2"""
    check_different_langs(lang1, lang2)
    if f"in_{lang2}" not in freq[lang1].columns or f"in_{lang1}" not in freq[lang2].columns:
        calculate_intersection(freq, lang1, lang2)
    return freq[lang1][~freq[lang1][f'in_{lang2}']].reset_index(), freq[lang2][~freq[lang2][f'in_{lang1}']].reset_index()

def get_specifics_three(freq, lang1, lang2, lang3):
    """ returns words specific to lang1, lang2, lang3"""
    check_different_langs(lang1, lang2)
    check_different_langs(lang2, lang3)
    check_different_langs(lang1, lang3)
    if f"in_{lang2}" not in freq[lang1].columns or f"in_{lang3}" not in freq[lang1].columns:
        calculate_intersection(freq, lang1, lang2)
        calculate_intersection(freq, lang1, lang3)
        calculate_intersection(freq, lang2, lang3)
    return freq[lang1][(~freq[lang1][f'in_{lang2}']) & (~freq[lang1][f'in_{lang3}'])].reset_index(),\
    freq[lang2][(~freq[lang2][f'in_{lang1}']) & (~freq[lang2][f'in_{lang3}'])].reset_index(),\
    freq[lang3][(~freq[lang3][f'in_{lang1}']) & (~freq[lang3][f'in_{lang2}'])].reset_index()

def get_specifics_df(freq, lang1, lang2):
    """ returns words specific to lang1, lang2 in one table, NOT MATCHED"""
    check_different_langs(lang1, lang2)
    specific1, specific2 = get_specifics(freq, lang1, lang2)
    df = pd.DataFrame({
        f'{lang1}_specific': specific1['word'],
        f'{lang1}_count': specific1['count'],
        f'{lang2}_specific': specific2['word'],
        f'{lang2}_count': specific2['count']
    })
    return df

def get_specifics_df_three(freq, lang1, lang2, lang3):
    """ returns words specific to lang1, lang2, lang3 in one table, NOT MATCHED"""
    check_different_langs(lang1, lang2)
    specific1, specific2, specific3 = get_specifics_three(freq, lang1, lang2, lang3)
    df = pd.DataFrame({
        f'{lang1}_specific': specific1['word'],
        f'{lang1}_count': specific1['count'],
        f'{lang2}_specific': specific2['word'],
        f'{lang2}_count': specific2['count'],
        f'{lang3}_specific': specific3['word'],
        f'{lang3}_count': specific3['count']
    })
    return df

def get_intersection_df(freq, lang1, lang2):
    """ returns table of mutual words between lang1 and lang2, sorted by sum of frequencies"""
    check_different_langs(lang1, lang2)
    df = freq[lang1].merge(freq[lang2], on='word', how='inner', suffixes=(f'_{lang1}', f'_{lang2}'))
    df['sum_count'] = df[f'count_{lang1}'] + df[f'count_{lang2}']
    df.sort_values('sum_count', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df[['word', f'count_{lang1}', f'count_{lang2}', 'sum_count']]

""" IOU METRIC FOR VOCABULARY ON A GROUP OF LANGUAGES """

def get_iou(freq, langs, round=None, percentage=False, also_intersection=False):
    """ Calculates IOU for words of given languages """
    iou = {lang:{} for lang in langs}
    intersection = {lang:{} for lang in langs}
    for i1 in tqdm(range(len(langs))):
        lang1 = langs[i1]
        
        # with itself
        iou[lang1][lang1] = 1
        intersection[lang1][lang1] = len(freq[lang1])

        # with others
        for i2 in range(i1 + 1, len(langs)):
            lang2 = langs[i2]
            intersection[lang1][lang2] = freq[lang1]['word'].apply(
                lambda x: x in freq[lang2]['word'].values).sum()
            cur_union = len(freq[lang1]) + len(freq[lang2]) - intersection[lang1][lang2]
            iou[lang1][lang2] = intersection[lang1][lang2] / cur_union
            
            # other way around
            iou[lang2][lang1] = iou[lang1][lang2]
            intersection[lang2][lang1] = intersection[lang1][lang2]
            
    df_iou = pd.DataFrame(iou) * (100 if percentage else 1)
    df_intersection = pd.DataFrame(intersection)
    if round is not None:
        df_iou = df_iou.round(round)
    if also_intersection:
        return df_iou, df_intersection
    return df_iou

def iou_heatmap(freq, langs, save_to, round=None, percentage=True, triangle=True, black=True, labels=False, annot=True, fmt='.0f'):
    """ saves and shows a heatmap for IOU, triangle by default """
    df_iou = get_iou(freq, langs, round, percentage)
    heatmap(df_iou, save_to, triangle, black, labels, annot, fmt)


""" NGRAMS """

def longest_ngrams_text(text1, text2, with_tqdm=True, upto=1000):
    """ Returns a list of longest common ngrams between two texts, punctuation is ignored """
    """ this one uses sets """
    words1 = text_to_words(text1)
    words2 = text_to_words(text2)
    last_ngrams = set([])
    new_ngrams = set([])
    cycle = range(1, upto)
    if with_tqdm:
        cycle = tqdm(cycle)
    for n in cycle:
        ngrams1 = set([])
        ngrams2 = set([])
        for i in range(len(words1) - n):
            ngram = ' '.join(words1[i:(i + n)])
            ngrams1.add(ngram)
        for i in range(len(words2) - n):
            ngram = ' '.join(words2[i:(i + n)])
            ngrams2.add(ngram)
        new_ngrams = ngrams1.intersection(ngrams2)
        if new_ngrams:
            last_ngrams = new_ngrams
            new_ngrams = set([])
        else:
            return list(last_ngrams)
    return list(last_ngrams)

def longest_ngrams(from_file, lang1, lang2, with_tqdm=True):
    text1 = get_text(from_file, lang1)
    text2 = get_text(from_file, lang2)
    return longest_ngrams_text(text1=text1, text2=text2, with_tqdm=with_tqdm)

def one_longest_ngram_text(text1, text2, with_tqdm=True):
    """ Returns one of the longest ngrams between two texts, punctuation is ignored """
    res = longest_ngrams_text(text1, text2, with_tqdm=with_tqdm)
    if res:
        return res[0]
    return ''

def one_longest_ngram(from_file, lang1, lang2, with_tqdm=True):
    text1 = get_text(from_file, lang1)
    text2 = get_text(from_file, lang2)
    return one_longest_ngram_text(text1=text1, text2=text2, with_tqdm=with_tqdm)

def words_count(phrase):
    """ Counts words in a phrase """
    return len(phrase.split())

def longest_ngram_table(from_file, langs, all=True, for_same='', also_length=False):
    """ if all=True, each cell is a list of ngrams or []; otherwise one or '' """
    longest_ngram = {lang:{} for lang in langs}
    for i1 in tqdm(range(len(langs))):
        lang1 = langs[i1]
        
        # with itself
        longest_ngram[lang1][lang1] = for_same

        # with others
        for i2 in range(i1 + 1, len(langs)):
            lang2 = langs[i2]
            if all:
                ngrams = longest_ngrams(from_file, lang1, lang2, with_tqdm=False)
            else:
                ngrams = one_longest_ngram(from_file, lang1, lang2, with_tqdm=False)
            longest_ngram[lang1][lang2] = ngrams
            
            # other way around
            longest_ngram[lang2][lang1] = ngrams
    df_ngrams = pd.DataFrame(longest_ngram)
    if also_length:
        if all:
            df_length = df_ngrams.map(lambda x: words_count(x[0]) if x else 0)
        else:
            df_length = df_ngrams.map(lambda x: words_count(x) if x else 0)
        return df_ngrams, df_length
    else:
        return df_ngrams



