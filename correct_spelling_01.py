# =============================================================================
# import libraries and data
# =============================================================================
import re
import stop_words
import hunspell
import spacy
import numpy as np
import warnings
import collections
import math

# --- ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

# --- class to hold all the tools
class sup_dct:
    def __init__(self, nlp, spl, stp):
        self.nlp = nlp
        self.spl = spl
        self.stp = stp
        
# --- the super dictionary
sup_dct_input = sup_dct(
    spacy.load("en_core_web_lg"), 
    hunspell.Hunspell('en_US'), 
    [string_mod(i) for i in stop_words.get_stop_words('en')]
)

# --- vectorizes words based on letter
def word_letter_vec(WrdInput):
    '''
    Parameters
    ----------
    WrdInput : TYPE str
        DESCRIPTION. a cleaned word

    Returns
    -------
    dict_out : TYPE dct
        DESCRIPTION. dictionary of letters and counts
    '''
    dict_out = dict(collections.Counter([i for i in string_mod(WrdInput)]))
    return dict_out

# --- function to get cosin given vector dictionaries
def get_cos_sim_letter(WordX, WordY):
    '''
    Parameters
    ----------
    WordX : TYPE str
    WordY : TYPE str
        DESCRIPTION. simple word
        
    Returns
    -------
    TYPE float
        DESCRIPTION. the cosin similarity between the two word dictionaries
    '''
    StrDictX = word_letter_vec(string_mod(WordX))
    StrDictY = word_letter_vec(string_mod(WordY))
    # -- get set of common words between the two dictionaries
    intersection = set(StrDictX.keys()) & set(StrDictY.keys())
    # -- get numerator for final cosin similarity
    numerator = sum([StrDictX[x] * StrDictY[x] for x in intersection])
    # -- get sum of squares of values (frequency) of words in vector/dictionary
    sumx = sum([StrDictX[x] ** 2 for x in list(StrDictX.keys())])
    sumy = sum([StrDictY[x] ** 2 for x in list(StrDictY.keys())])
    # -- dnominator is then the square root of the first dictionary times the squareroot of the second
    denominator = math.sqrt(sumx) * math.sqrt(sumy)
    # -- if no values available then simply return 0 else return numerator divided by denominator
    if not denominator:
        return 0.0
    else:
        # - denominator should be float
        return (float(numerator) / denominator)*.5

# --- check word if need spelling
def check_wrd(WordInput, SupDctInput):
    '''
    Parameters
    ----------
    WordInput : TYPE str
        DESCRIPTION. A word that needs to be checked for spelling
    SupDctInput : TYPE sup_dct
        DESCRIPTION. An object of class sup_dct that holds all our tools

    Returns
    -------
    needs_check : TYPE boolean
        DESCRIPTION. value designating whether the string passes the spell check
    '''
    # -- set default
    needs_check = False
    # -- check if stop word
    if WordInput not in SupDctInput.stp and len(WordInput) > 1:
        needs_check = not SupDctInput.spl.spell(WordInput)
    # -- return bool
    return needs_check

# --- get list of comparison words
def get_comp_word_lst(StrDctInput, SupDctInput):
    '''
    Parameters
    ----------
    StrDctInput : TYPE dct
        DESCRIPTION. a dictionary of strings
    SupDctInput : TYPE sub_dct
        DESCRIPTION. An object of class sup_dct that holds all our tools

    Returns
    -------
    wrd_comp_lst_out : TYPE list
        DESCRIPTION. A list of words that we will compare similarity with
    '''
    # -- list to append to
    wrd_comp_lst_out = []
    # -- loop through
    for i in StrDctInput:
        wrd_to_check = StrDctInput[i]['clean_word']
        if SupDctInput.spl.spell(wrd_to_check) and wrd_to_check not in SupDctInput.stp and len(wrd_to_check) > 1:
            wrd_comp_lst_out.append(wrd_to_check)
    # -- return list
    return wrd_comp_lst_out

# --- function to get cosin similarity based on pretrained vals
def get_cos_sim(WordX, WordY, SupDctInput):
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
    try:
        vec_x = SupDctInput.nlp(WordX).vector 
        vec_y = SupDctInput.nlp(WordY).vector
        similarity = vec_x.dot(vec_y) / (np.linalg.norm(vec_x) * np.linalg.norm(vec_y))
    except:
        similarity = 0.0
    return similarity

# --- grab suggestion
def get_sug(WrdInput, CompLstInput, SupDctInput):
    '''
    Parameters
    ----------
    WrdInput : TYPE str
        DESCRIPTION. the word that needs to be compared
    CompLstInput : TYPE list
        DESCRIPTION. list of words to compare against
    SupDctInput : TYPE sup_dct
        DESCRIPTION. An object of class sup_dct that holds all our tools

    Returns
    -------
    sug_out : TYPE str
        DESCRIPTION. the suggested word based on choices
    max_cos : TYPE float
        DESCRIPTION. the similarity score
    '''
    # -- list to work with dictionary to append to
    possible_sug_lst = list(SupDctInput.spl.suggest(WrdInput))
    print(possible_sug_lst)
    possible_sug_dct = {}
    # -- loop through
    for wrd in possible_sug_lst:
        tmp_cos_lst = []
        for comp_wrd in CompLstInput:
            try:
                tmp_cos = np.mean([get_cos_sim(wrd, comp_wrd, SupDctInput), get_cos_sim_letter(wrd, comp_wrd)])
                if not np.isnan(tmp_cos):
                    tmp_cos_lst.append(tmp_cos)
                else:
                    tmp_cos_lst.append(0.0)
            except:
                tmp_cos_lst.append(0.0)
        possible_sug_dct[wrd] = np.mean(tmp_cos_lst)
    # -- find word with max
    print(possible_sug_dct)
    max_cos = np.max(list(possible_sug_dct.values()))
    sug_out = [w for w in possible_sug_dct if possible_sug_dct[w] == max_cos][0]
    # -- return suggestion
    return sug_out, max_cos

# --- final function
def correct_spelling(StrInput, SupDctInput, CosinThold = 0.15):
    '''
    Parameters
    ----------
    StrInput : TYPE str
        DESCRIPTION. The string with spelling errors that we need to correct
    SupDctInput : TYPE sup_dct
        DESCRIPTION. An object of class sup_dct that holds all our tools
    CosinThold : TYPE, optional float
        DESCRIPTION. The default is 0.15.

    Returns
    -------
    StrInput : TYPE str
        DESCRIPTION. the original string input
    str_out : TYPE str
        DESCRIPTION. the modified string
    '''
    try:
        # --- convert string to list
        str_lst = [i for i in re.split(r"[,\s]+", StrInput)]
        # --- dictionary to append to
        str_dct = {}
        # --- convert to dct
        for i in range(len(str_lst)):
            str_dct[i] = {'orig_word':str_lst[i], 'clean_word':string_mod(str_lst[i]), 'cos':0.0}  
        # --- final list to append to
        lst_out = []
        # --- run through list
        for i in str_dct:
            wrd_to_check = str_dct[i]['clean_word']
            if check_wrd(wrd_to_check, SupDctInput):
                print(wrd_to_check)
                tmp_comp_lst = get_comp_word_lst(str_dct, SupDctInput)
                print(tmp_comp_lst)
                try:
                    tmp_sug, max_cos = get_sug(wrd_to_check, tmp_comp_lst, SupDctInput)
                except:
                    tmp_sug = ''
                    max_cos = 0.0
                str_dct[i]['clean_word'] = tmp_sug
                str_dct[i]['cos'] = max_cos
        # --- loop through dictionary
        for i in str_dct:
            if str_dct[i]['cos'] >= CosinThold:
                lst_out.append(str_dct[i]['clean_word'])
            else:
                lst_out.append(str_dct[i]['orig_word'])
        # --- final string
        str_out = ' '.join(lst_out)
    except:
        str_out = ''
    return StrInput, str_out

# =============================================================================
# apply functions
# =============================================================================
old_string, new_string = correct_spelling(
    'At school I niver lerned how to spelll so I enlosted in the infanty as a soldir', sup_dct_input
)
print(
f'''
OLD:
    {old_string}
    
NEW:
    {new_string}
'''
)

