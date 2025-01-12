# =============================================================================
# import libraries and data
# =============================================================================
import os
import pandas as pd
import re
from nltk import word_tokenize, everygrams, FreqDist, pos_tag, stem
import stop_words
import hunspell
import collections
import spacy
import numpy as np
import random as rd
import time

# --- grab tbl and modify
df_01 = pd.DataFrame(
    {
     'str_orig': [
        "Belavi Outdoor Fire Pit",
        "Maeil My Cafe Latte",
        "MLB Bandana",
        "Shark FlexBreeze Cordless Fan",
        "Mini Sweet Peppers",
        "Quilted Northern Toilet Paper Mega Rolls",
        "Topo Chico Sparkling Water" ,
        "Meijer Double Zipper Sandwich Bags",
        "Reddy Stainless Steel Mango Wood Dog Bowl",
        "Bellamy 68"" Traditional Console",
        "Little Tikes Foldable Slide",
        "Jimmy Dean Sliced Bacon",
        "Blue Bunny Mini Swirls",
        "Kroger Large Eggs",
        "Dawn Powerwash Dish Spray Starter Kit",
        "Father's Day Mug",
        "General Mills Honey Nut Cheerios Large Size Cereal",
        "Yardistry Gazebo with Aluminum Roof",
        "Thomasville Ash Fabric Swivel Chair",
        "Member's Mark Kids' Swim Set",
        "A black dog is retrieving a ball in water .",
        "two dogs play together .",
        "A basketball player in a red uniform is trying to make a basket but is being blocked by the opposing team .",
        "Three Oklahoma Sooners playing football against another team , one of the sooners with the ball in their possession .",
        "A woman with green hair hula-hoops in a flowered orange top .",
        "Two people playing baseball .",
        "Two men walk down a stone stairway , one is riding a unicycle .",
        "Men with surveying equipment are in the background of landscape .",
        "Two football players talk during a game .",
        "A dog runs on brown grass .",
        "A couple in an embrace looking at each other .",
        "A group of people seated in a line of seats at an event .",
        "Someone is parasailing on a blue and yellow parasail .",
        "The man in the blue shirt is surrounded by three dogs in the grass .",
        "A child wearing black and purple playing baseball .",
        "Skiiers on a snowy mountain .",
        "The basketball player in white holds the ball .",
        "two young girls wearing bonnets standing by a tree watching a snake nearby .",
        "There is a jeep stuck in mud up to the doors .",
        "A man sits on the gravel by an ocean ."
        ]    
    }
)

# =============================================================================
# functions 01
# =============================================================================
# --- cleans post content data
def string_mod(StrInput):
    '''
    Parameters
    ----------
    str_x : TYPE string
        DESCRIPTION. a string that needs to be cleaned
        
    Returns
    -------
    str_mod : TYPE string
        DESCRIPTION. a cleaned version of the string
    '''   
    # -- remove all html
    str_mod = re.sub('\'', '', StrInput)
    # -- remove everything but space and alpha
    str_mod = re.sub(r'[^a-zA-Z ]', ' ', str_mod)
    # -- any space 2 or greater change to one
    str_mod = re.sub(r' {2,}', ' ', str_mod)
    # -- strip and lower
    str_mod = str_mod.strip().lower()   
    # -- return final string
    return str_mod

# --- function to get base dictionary
def base_dictionary(StringListInput, DictInput, StopWordInput):
    '''
    Parameters
    ----------
    StringListInput : TYPE list
        DESCRIPTION. a list of strings
    DictInput : TYPE hunspell.HunspellWrap
        DESCRIPTION. a dictionary from huspell library
    StopWordInput : TYPE list
        DESCRIPTION. a list of stop words
        
    Returns
    -------
    wrd_dct : TYPE dictionary
        DESCRIPTION. a dictionary of dictionaries
    '''
    # -- list to append to
    tmp_word_lst = []
    # -- get all the words and place in list
    for stc in [string_mod(i) for i in StringListInput]:
        try:
            [tmp_word_lst.append(wrd) for wrd in word_tokenize(stc)]
        except:
            continue
    # -- this is our first word dictionary with counts
    wrd_dct_counts = collections.Counter(tmp_word_lst)
    # -- our primary word dictionary
    wrd_dct = {}
    # -- loop through all words
    for wrd in wrd_dct_counts:
        # -- default values
        wrd_dct_loop = {
            'count':wrd_dct_counts[wrd],
            'pos':'',
            'all_stem':'',
            'short_stem':'',
            'can_spell':False,
            'nchar':len(wrd),
            'is_stop_word':False
        }
        # -- try to get pos
        try:
            tmp_pos = pos_tag([wrd])[0][1]
            wrd_dct_loop['pos'] = tmp_pos
        except:
            pass
        # -- try to get all stem and short stem
        try:
            tmp_stem_lst = list(DictInput.stem(wrd))
            tmp_short_stem = [
                i for i in tmp_stem_lst if len(i) == min([len(i) for i in tmp_stem_lst])
            ][0]
            wrd_dct_loop['all_stem'] = tmp_stem_lst
            wrd_dct_loop['short_stem'] = tmp_short_stem
        except:
            pass
        # -- see if spellable
        try:
            wrd_dct_loop['can_spell'] = DictInput.spell(wrd)
        except:
            pass
        # -- check if stop word
        try:
            if wrd in StopWordInput:
                wrd_dct_loop['is_stop_word'] = True
        except:
            pass                
        # -- set value of wrd dct
        wrd_dct[wrd]=wrd_dct_loop
    # -- output dictionary
    return wrd_dct

# --- function to get cosin similarity based on pretrained vals
def get_cos_sim(WordX, WordY, NlpInput):
    '''
    Parameters
    ----------
    WordX : TYPE string
        DESCRIPTION. one word string
    WordY : TYPE string
        DESCRIPTION. one word string
    NlpInput : TYPE spacy.lang
        DESCRIPTION. word vectorizer
    Returns
    -------
    similarity : TYPE float
        DESCRIPTION. similarity between two words
    '''
    vec_x = NlpInput(WordX).vector 
    vec_y = NlpInput(WordY).vector
    similarity = vec_x.dot(vec_y) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))
    return similarity

# --- gets sub word from dictionary
def get_sub_word(
        WordInput, 
        BaseDictionaryInput,
        NlpInput,
        CosineSimThold = 0.01,
        NcharThold = 0,
        OmitStopWords = [],
        OmitNonSpell = [],
        OmitPosList = []
    ):
    # --- dictionary we are adding sub word to
    tmp_wrd_dct = BaseDictionaryInput[WordInput]
    # --- set default if no conditions met
    tmp_wrd_dct['sub'] = WordInput
    # --- default sub value is short stem word 
    if tmp_wrd_dct['short_stem']:
        if tmp_wrd_dct['short_stem'] == WordInput:
            tmp_wrd_dct['sub'] = WordInput
        else:
            try:
                cos_sim = get_cos_sim(WordInput, tmp_wrd_dct['short_stem'], NlpInput)
            except:
                cos_sim = 0.0
            if cos_sim >= CosineSimThold:
                tmp_wrd_dct['sub'] = tmp_wrd_dct['short_stem']
    # --- if its stop word
    if OmitStopWords and tmp_wrd_dct['is_stop_word']:
        tmp_wrd_dct['sub'] = ''
    # --- if its non spellable
    if OmitNonSpell and tmp_wrd_dct['can_spell'] == False:
        tmp_wrd_dct['sub'] = ''
    # --- if its below char threshold
    if tmp_wrd_dct['nchar'] <= NcharThold:
        tmp_wrd_dct['sub'] = ''
    # --- if its in omit pos list
    if tmp_wrd_dct['pos'] in OmitPosList:
        tmp_wrd_dct['sub'] = ''
    # --- return dictionary with sub word
    return tmp_wrd_dct

# --- use sub word dictionary and output new string
def use_sub_dct(StrInput, SubWrdDctInput):
    # -- clean string
    new_string = string_mod(StrInput)
    # -- split to list
    new_string_split = word_tokenize(new_string)
    # -- sub
    new_string = ' '.join([SubWrdDctInput[i]['sub'] for i in new_string_split])
    # -- re clean to get rid of white space
    string_out = string_mod(new_string)
    # -- return string
    return string_out

# =============================================================================
# apply functions
# =============================================================================
# --- get base dictionary to modify as sub dictionary
base_dct = base_dictionary(
    StringListInput=list(df_01['str_orig']),
    DictInput=hunspell.Hunspell('en_US'),
    StopWordInput=[string_mod(i) for i in stop_words.get_stop_words('en')]
)

# --- the nlp input we will use
nlp_input = spacy.load("en_core_web_lg")

# --- sub dictionary
sub_dct = {}

# --- set values in base dictionary
for word in base_dct:
    try:
        tmp_wrd_dct = get_sub_word(
            WordInput=word, 
            BaseDictionaryInput=base_dct,
            NlpInput=nlp_input,
            CosineSimThold = 0.5,
            NcharThold = 2,
            OmitStopWords = True,
            OmitNonSpell = True,
            OmitPosList =['CC', 'CD', 'DT', 'IN']
        )
        sub_dct[word] = tmp_wrd_dct
    except:
        continue

# --- use sub dictionary
df_01['str_mod'] = df_01['str_orig'].apply(lambda x: use_sub_dct(x, sub_dct))
df_01 = df_01[df_01['str_mod'] != '']

for inx, row in df_01.iterrows():
    orig_string = row['str_orig']
    new_string = row['str_mod']
    str_out = f'''
---------------------------------------------------------------
The old product description:
{orig_string}


the new product description: 
{new_string}
---------------------------------------------------------------
'''
    print(str_out)
    time.sleep(2)
    
   
