import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()
from utility import filter_file_and_save, count_nb_rows_in_file

def extract_quotes(names_without_check_set, names_with_categories_dict, input_dir, output_file_all, output_file_cleaned):
  '''
    extract the quotes from a given input directory and outputs two files
    The first output contains the quotes where one of the elements is in the set 
    "names_without_check_set" or in the key set of "names_with_categories_dict"
    The second one filters a bit further, keeping only quotes where one of the 
    elements is in the set "names_without_check_set" and looks if the types of 
    any key of "names_with_categories_dict" corresponds to an element of its keys.

    Parameters:
      names_without_check_set    : set of Names that will be always kept
      names_with_categories_dict : dict of Names associated with their labels, for example 
                                   if we look for Elon Musk we search of Tesla as an organization or product
                                   {"Elon" : {"PERSON"}, "Tesla" : {"ORG","PRODUCT"}}
      input_dir                  : the input file or directory, containing the original quotes
      output_file_all            : the output file of the coarse filter
      output_file_cleaned        : the output file of the fine filter

  '''
  # Basic filter to extract quotes where any of the keywords appears.
  all_keywords = set(names_with_categories_dict.keys()).union(names_without_check_set)

  def contains_one_of(quotation): 
    # returns true if any of the keywords is in the quotation
    return any([e in quotation['quotation'] for e in all_keywords])
  def df_contains_one_of(df):
    # applies "contains_one_of" to the whole DataFrame
    return df[df.apply(contains_one_of, axis = 1)]

  print(f"coarse filtering started with keywords : {all_keywords}")
  # Only keep the lines where the kewords appear
  filter_file_and_save(df_contains_one_of, input_dir, output_file_all, chunksize=500_000)
  print(f"coarse filtering completed and file saved under : {output_file_all}")
  count_nb_rows_in_file(output_file_all)

  print(f"fine filtering started with keywords : {all_keywords}")
  def find_key_words(quotation): 

    #store all key words found in the quotes
    key_words_found = []
    #check for names that don't require checking their categories
    for e in names_without_check_set : 
      if e in quotation['quotation']:
        key_words_found.append(e)
    named_entities = nlp(quotation['quotation']).ents
    #check for names that must be linked to categories
    for e in named_entities : 
      if (e.text in names_with_categories_dict) and (e.label_ in names_with_categories_dict[e.text]):
        key_words_found.append(e.text)
    return key_words_found
    
  def df_contains_one_of(df):
    key_words_found = df.apply(find_key_words,axis = 1)
    filtered_df = df[ key_words_found.str.len() > 0 ]
    filtered_df['key_words'] = key_words_found
    return filtered_df
  
  # filter and save results
  print(f"fine filtering started")
  filter_file_and_save(df_contains_one_of, output_file_all, output_file_cleaned, chunksize=500_000)
  print(f"fine filtering completed")
  count_nb_rows_in_file(output_file_cleaned)