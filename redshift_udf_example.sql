---------------------------------------- [ generate practice data ] -----
-- be sure to change up schema names
DROP TABLE IF EXISTS whitt11b_02.tmp_product_data;

    CREATE TABLE whitt11b_02.tmp_product_data (
        product VARCHAR(20000)
        ,purchase_type VARCHAR(256)
    );

INSERT INTO whitt11b_02.tmp_product_data VALUES
    ('Creatine HCl Capsules | Muscle, Cognitive, Cellular Energy Support | No Bloating or Cramps | USA Made & NSF Certified | Creatine Pills (90 ct)', 'direct'),
    ('Creatine HCl Gummies for Men & Women | Muscle, Cognitive, Cellular Energy Support | No Bloating or Cramps | USA Made | Berry Zing (60 Count)', 'indirect'),
    ('Sashco - 10018 Big Stretch Acrylic Latex High Performance Caulking Sealant, 10.5 oz Cartridge, Woodtone', 'direct'),
    ('Sashco Log Builder Acrylic Latex Chinking Caulk, 30 oz Cartridge, Tan (Pack of 10)', 'indirect'),
    ('Akfix AS606 White Acrylic Latex Caulk for Painters (12x10.1 fl.oz.) - Siliconized Acrylic Caulk for Window and Door Sealing, Crack Filling, Baseboards & Trim, Odorless, Paintable | 12 Pack', 'indirect'),
    ('The StatQuest Illustrated Guide To Machine Learning Paperback', 'direct'),
    ('No bullshit guide to linear algebra Paperback', 'indirect'),
    ('KUAFU Front Bumper Upper & Lower Grille Grill Compatible with 2017-2021 Chevy Chevrolet Trax Replacement for 42615979, 42537706, 42519611 Chrome Black+Silvery', 'direct'),
    ('KUAFU Front Bumper Upper Grille Grill Compatible with 2017-2021 Chevy Chevrolet Trax Replacement for 94532311 Chrome Black+Silvery', 'indirect');

-- check out the data
SELECT * FROM whitt11b_02.tmp_product_data;

---------------------------------------- [ f_str_repair ] -----
CREATE OR REPLACE FUNCTION f_str_repair (StrInput VARCHAR(20000)) RETURNS VARCHAR(20000) IMMUTABLE as $$

    def str_repair(StrInput):
        import re
        '''
        Parameters
        ----------
        StrInput : TYPE string
            DESCRIPTION. takes a string with capital letters and imposes underscores and then lowers it

        Returns
        -------
        str_out : TYPE string
            DESCRIPTION. the new string we want
        '''
        # -- get rid of all non letters and make everything lower case
        str_out = re.sub('[^a-zA-Z]', ' ', StrInput).lower()
        # -- remove any white space on edges
        str_out = str_out.strip()
        # -- if theres more two or more spaces - lets make that one space
        str_out = re.sub(' {1,}', ' ', str_out)
        return str_out
    return str_repair(StrInput)

$$ LANGUAGE plpythonu;

---------------------------------------- [ prod_02] -----
DROP TABLE IF EXISTS prod_02;

    SELECT DISTINCT
        f_str_repair(product) AS product_clean
        ,purchase_type
    INTO TEMP prod_02
    FROM whitt11b_02.tmp_product_data;

SELECT * FROM prod_02;

---------------------------------------- [ prod_03] -----
DROP TABLE IF EXISTS prod_03;

    SELECT
        p1.product_clean AS direct_product
        ,p2.product_clean AS indirect_product
    INTO TEMP prod_03
    FROM prod_02 p1, prod_02 p2
    WHERE
        p1.purchase_type = 'direct'
        AND p2.purchase_type = 'indirect'
    ORDER BY 1,2;

SELECT * FROM prod_03;

---------------------------------------- [ f_cosin_sim ] -----
CREATE OR REPLACE FUNCTION f_cosin_sim (StrInputX VARCHAR(20000), StrInputY VARCHAR(20000), WordLenLim Integer) RETURNS double precision IMMUTABLE as $$

    # --- transforms sentences into vectors
    def text_to_vector(StrInput, WordLenLim):
        # -- requrired packages
        import re
        '''
        Parameters
        ----------
        StrInput : TYPE String
            DESCRIPTION. unstructured sentence
        WordLenLim : TYPE Int
            DESCRIPTION. omit words with character length less than this
        Returns
        -------
        wrd_vec_dct : TYPE Dictionary
            DESCRIPTION. word count dictionary
        '''
        # -- condense sentence to list of words
        wrd_lst = [i for i in re.split(' ', StrInput) if len(i) > WordLenLim]
        # -- dictionary we will append to
        wrd_vec_dct = dict(zip(list(set(wrd_lst)), [0 for i in list(set(wrd_lst))]))
        # -- now we count the words
        for wrd in wrd_lst:
            wrd_vec_dct[wrd] += 1
        return wrd_vec_dct

    # --- function to get cosin given vector dictionaries
    def get_cosine(StrDictX, StrDictY):
        '''
        Parameters
        ----------
        StrInputX : TYPE dict
        StrInputY : TYPE dict
            DESCRIPTION. dictionary with words and their frequency
        Returns
        -------
        TYPE float
            DESCRIPTION. the cosin similarity between the two word dictionaries
        '''
        import math
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
            return float(numerator) / denominator

    # --- attempt to get sim - if you cant than output 0.0
    try:
        # -- get vectors
        StrDictX = text_to_vector(StrInputX, 2)
        StrDictY = text_to_vector(StrInputY, 2)
        # -- get cosin given vectors
        cosine_out = get_cosine(StrDictX, StrDictY)
    except:
        cosine_out = 0.0
    # --- return final val
    return cosine_out

$$ LANGUAGE plpythonu;

---------------------------------------- [ prod_04] -----
DROP TABLE IF EXISTS prod_04;

    SELECT
        p.direct_product
        ,p.indirect_product
        ,f_cosin_sim(p.direct_product, p.indirect_product, 2) AS product_similarity
    INTO TEMP prod_04
    FROM prod_03 p;

SELECT * FROM prod_04;

---------------------------------------- [ FINAL ] -----
SELECT
    p.indirect_product
    ,p.direct_product
    ,p.product_similarity
FROM prod_04 p
INNER JOIN (
    SELECT
        p.indirect_product
        ,max(p.product_similarity) AS max_product_similarity
    FROM prod_04 p
    GROUP BY 1
) pm ON pm.indirect_product = p.indirect_product AND pm.max_product_similarity = p.product_similarity
ORDER BY 2,1,3
