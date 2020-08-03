import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

from Function_file import *

import lir as liar


speakers_path = 'JSON/speakers_FINAL.json'

if os.path.exists(speakers_path):
    print('loading', speakers_path)
    speakers_wordlist = load_data(speakers_path)
else:
    speakers_wordlist = compile_data('SHA256_textfiles/FINALdata.txt')
    store_data(speakers_path, speakers_wordlist)

sample_size = 100
n_freq = 50

wordlist_test = get_frequent_table(speakers_wordlist, speakers_wordlist, n_freq)
    #print('##############################################################\n ##############################################################')
    #wordlist = get_frequent_words(speakers_wordlist_CGN, n_freq)
    #wordlist_test = get_frequent_table(speakers_wordlist, speakers_wordlist_CGN, n_freq)
