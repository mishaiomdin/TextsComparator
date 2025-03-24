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

def get_text(from_file, text_name):
    with open(from_file + text_name + '.txt') as file:
        text = file.read()
    return text

def get_word_frequency(text_file, non_alphabet_re=non_alphabet_re2, lower=True):
    """ Returns table of word frequencies in a text """
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
    """ Returns table of character frequencies in a text """
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

def get_ngrams_frequency(text, n):
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

def save_all(from_file, to_file, text_names):
    """ Calculates frequencies for each; saves to csv """
    for text_name in text_names:
        freq = get_word_frequency(from_file + text_name + '.txt')
        freq.to_csv(to_file + text_name + '.csv', index=False)

def get_all(from_file, text_names):
    """ Returns a dictionary for all word frequencies of a group of texts """
    res = {}
    for text_name in text_names:
        res[text_name] = pd.read_csv(from_file + text_name + '.csv')
    return res

def get_all_texts(from_file, text_names):
    """ Returns a dictionary for a group of texts """
    res = {}
    for text_name in text_names:
        with open(from_file + text_name + '.txt') as file:
            res[text_name] = file.read()
    return res

def get_group_frequencies(from_file, texts):
    """ Returns a dictionary for a group of texts """
    res = {}
    for text_name in texts:
        res[text_name] = get_word_frequency(texts[text_name])
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

""" COMPARE SEVERAL TEXTS """

def compare_frequent(freq, *text_names, limit=10):
    """ Returns an UNMATCHED table of most frequent words in any number of text_nameuages """
    return pd.DataFrame({
        text_name: freq[text_name]['word'] for text_name in text_names
    }).head(limit)

def compare_ngrams(n, texts, *text_names, limit=10):
    """ Returns an UNMATCHED table of most frequent ngrams in any number of text_nameuages """
    return pd.concat(
        [get_ngrams_frequency(texts[text_name], n) for text_name in text_names]
    ).head(limit)

def check_different_text_names(text_name1, text_name2):
    """ checks that two different text_nameuages are given """
    if text_name1 == text_name2:
        raise Exception("Can't compare a text_nameuage with itself")

def calculate_intersection(freq, text_name1, text_name2):
    """ calculates in_text_name1 and in_text_name2 if not yet; returns set of intersected words """
    check_different_text_names(text_name1, text_name2)
    intersection = set(freq[text_name1]['word']).intersection(set(freq[text_name2]['word']))
    if f"in_{text_name2}" not in freq[text_name1].columns:
        freq[text_name1][f'in_{text_name2}'] = freq[text_name1]['word'].apply(lambda x: x in intersection)
    if f"in_{text_name1}" not in freq[text_name2].columns:
        freq[text_name2][f'in_{text_name1}'] = freq[text_name2]['word'].apply(lambda x: x in intersection)
    return intersection

def get_specifics(freq, text_name1, text_name2):
    """ returns words specific to text_name1, text_name2"""
    check_different_text_names(text_name1, text_name2)
    if f"in_{text_name2}" not in freq[text_name1].columns or f"in_{text_name1}" not in freq[text_name2].columns:
        calculate_intersection(freq, text_name1, text_name2)
    return freq[text_name1][~freq[text_name1][f'in_{text_name2}']].reset_index(), freq[text_name2][~freq[text_name2][f'in_{text_name1}']].reset_index()

def get_specifics_three(freq, text_name1, text_name2, text_name3):
    """ returns words specific to text_name1, text_name2, text_name3"""
    check_different_text_names(text_name1, text_name2)
    check_different_text_names(text_name2, text_name3)
    check_different_text_names(text_name1, text_name3)
    if f"in_{text_name2}" not in freq[text_name1].columns or f"in_{text_name3}" not in freq[text_name1].columns:
        calculate_intersection(freq, text_name1, text_name2)
        calculate_intersection(freq, text_name1, text_name3)
        calculate_intersection(freq, text_name2, text_name3)
    return freq[text_name1][(~freq[text_name1][f'in_{text_name2}']) & (~freq[text_name1][f'in_{text_name3}'])].reset_index(),\
    freq[text_name2][(~freq[text_name2][f'in_{text_name1}']) & (~freq[text_name2][f'in_{text_name3}'])].reset_index(),\
    freq[text_name3][(~freq[text_name3][f'in_{text_name1}']) & (~freq[text_name3][f'in_{text_name2}'])].reset_index()

def get_specifics_df(freq, text_name1, text_name2):
    """ returns words specific to text_name1, text_name2 in one table, NOT MATCHED"""
    check_different_text_names(text_name1, text_name2)
    specific1, specific2 = get_specifics(freq, text_name1, text_name2)
    df = pd.DataFrame({
        f'{text_name1}_specific': specific1['word'],
        f'{text_name1}_count': specific1['count'],
        f'{text_name2}_specific': specific2['word'],
        f'{text_name2}_count': specific2['count']
    })
    return df

def get_specifics_df_three(freq, text_name1, text_name2, text_name3):
    """ returns words specific to text_name1, text_name2, text_name3 in one table, NOT MATCHED"""
    check_different_text_names(text_name1, text_name2)
    specific1, specific2, specific3 = get_specifics_three(freq, text_name1, text_name2, text_name3)
    df = pd.DataFrame({
        f'{text_name1}_specific': specific1['word'],
        f'{text_name1}_count': specific1['count'],
        f'{text_name2}_specific': specific2['word'],
        f'{text_name2}_count': specific2['count'],
        f'{text_name3}_specific': specific3['word'],
        f'{text_name3}_count': specific3['count']
    })
    return df

def get_intersection_df(freq, text_name1, text_name2):
    """ returns table of mutual words between text_name1 and text_name2, sorted by sum of frequencies"""
    check_different_text_names(text_name1, text_name2)
    df = freq[text_name1].merge(freq[text_name2], on='word', how='inner', suffixes=(f'_{text_name1}', f'_{text_name2}'))
    df['sum_count'] = df[f'count_{text_name1}'] + df[f'count_{text_name2}']
    df.sort_values('sum_count', inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df[['word', f'count_{text_name1}', f'count_{text_name2}', 'sum_count']]

""" IOU METRIC FOR VOCABULARY ON A GROUP OF text_nameUAGES """

def get_iou(freq, text_names, round=None, percentage=False, also_intersection=False):
    """ Calculates IOU for words of given text_nameuages """
    iou = {text_name:{} for text_name in text_names}
    intersection = {text_name:{} for text_name in text_names}
    for i1 in tqdm(range(len(text_names))):
        text_name1 = text_names[i1]
        
        # with itself
        iou[text_name1][text_name1] = 1
        intersection[text_name1][text_name1] = len(freq[text_name1])

        # with others
        for i2 in range(i1 + 1, len(text_names)):
            text_name2 = text_names[i2]
            intersection[text_name1][text_name2] = freq[text_name1]['word'].apply(
                lambda x: x in freq[text_name2]['word'].values).sum()
            cur_union = len(freq[text_name1]) + len(freq[text_name2]) - intersection[text_name1][text_name2]
            iou[text_name1][text_name2] = intersection[text_name1][text_name2] / cur_union
            
            # other way around
            iou[text_name2][text_name1] = iou[text_name1][text_name2]
            intersection[text_name2][text_name1] = intersection[text_name1][text_name2]
            
    df_iou = pd.DataFrame(iou) * (100 if percentage else 1)
    df_intersection = pd.DataFrame(intersection)
    if round is not None:
        df_iou = df_iou.round(round)
    if also_intersection:
        return df_iou, df_intersection
    return df_iou

def iou_heatmap(freq, text_names, save_to, round=None, percentage=True, triangle=True, black=True, labels=False, annot=True, fmt='.0f'):
    """ saves and shows a heatmap for IOU, triangle by default """
    df_iou = get_iou(freq, text_names, round, percentage)
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

def longest_ngrams(from_file, text_name1, text_name2, with_tqdm=True):
    text1 = get_text(from_file, text_name1)
    text2 = get_text(from_file, text_name2)
    return longest_ngrams_text(text1=text1, text2=text2, with_tqdm=with_tqdm)

def one_longest_ngram_text(text1, text2, with_tqdm=True):
    """ Returns one of the longest ngrams between two texts, punctuation is ignored """
    res = longest_ngrams_text(text1, text2, with_tqdm=with_tqdm)
    if res:
        return res[0]
    return ''

def one_longest_ngram(from_file, text_name1, text_name2, with_tqdm=True):
    text1 = get_text(from_file, text_name1)
    text2 = get_text(from_file, text_name2)
    return one_longest_ngram_text(text1=text1, text2=text2, with_tqdm=with_tqdm)

def words_count(phrase):
    """ Counts words in a phrase """
    return len(phrase.split())

def longest_ngram_table(from_file, text_names, all=True, for_same='', also_length=False):
    """ if all=True, each cell is a list of ngrams or []; otherwise one or '' """
    longest_ngram = {text_name:{} for text_name in text_names}
    for i1 in tqdm(range(len(text_names))):
        text_name1 = text_names[i1]
        
        # with itself
        longest_ngram[text_name1][text_name1] = for_same

        # with others
        for i2 in range(i1 + 1, len(text_names)):
            text_name2 = text_names[i2]
            if all:
                ngrams = longest_ngrams(from_file, text_name1, text_name2, with_tqdm=False)
            else:
                ngrams = one_longest_ngram(from_file, text_name1, text_name2, with_tqdm=False)
            longest_ngram[text_name1][text_name2] = ngrams
            
            # other way around
            longest_ngram[text_name2][text_name1] = ngrams
    df_ngrams = pd.DataFrame(longest_ngram)
    if also_length:
        if all:
            df_length = df_ngrams.map(lambda x: words_count(x[0]) if x else 0)
        else:
            df_length = df_ngrams.map(lambda x: words_count(x) if x else 0)
        return df_ngrams, df_length
    else:
        return df_ngrams



