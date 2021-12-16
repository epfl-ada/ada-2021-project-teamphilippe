import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import bz2
from tqdm import tqdm

def apply_to_stream(f, input_file, args=None, chunksize=1_000_000):
  """
    apply func to input_file and save in output_file

    Parameters:
      func :        fonction to apply of type : args = func(df_chunk, args)
      input_file :  input file
      args :        arguments of func 
      chunksize :   chunk size

    Returns:
      args :        arguments at the end of the execution
  """
  with pd.read_json(input_file, lines=True, compression='bz2', chunksize=chunksize) as df_reader:
    for chunk in tqdm(df_reader):
      args = f(chunk, args)
  return args

def write_df_chunk_to_file(df_chunk, file):
  """
    write the chunk in to the file

    Parameters:
      df_chunk :  fonction to apply of type : new_df_chunk, args = func(df_chunk, args)
      file :      file_stream
  """
  file.write(df_chunk.to_json(orient='records', lines=True).encode('utf-8'))

def apply_to_stream_and_save(func, input_file, output_file, args=None, chunksize=1_000_000, override=False):
  """
    apply func to input_file and save in output_file

    Parameters:
      func :        fonction to apply of type : new_df_chunk, args = func(df_chunk, args)
      input_file :  input file
      output_file : output file
      args :        arguments of func 
      chunksize :   chunk size
      override :    override output_file if alredy exist

    Returns:
      args :        arguments at the end of the execution
  """
  if(override or (isfile(input_file) and not isfile(output_file))):
    with pd.read_json(input_file, lines=True, compression='bz2', chunksize=chunksize) as df_reader:
        with bz2.open(output_file, 'wb') as out_file:
            for df_chunk in tqdm(df_reader):
              new_df_chunk, args = func(df_chunk, args)
              write_df_chunk_to_file(new_df_chunk, out_file)
  return args


def apply_to_all_stream_and_save(func, all_input_file, output_file, args=None, chunksize=1_000_000, override=False):
  """
    apply func to all input_file and save in output_file

    Parameters:
      func :            fonction to apply of type : new_df_chunk, args = func(df_chunk, args)
      all_input_file :  input file
      output_file :     output file
      args :            arguments of func 
      chunksize :       chunk size
      override :        override output_file if alredy exist

    Returns:
      args :        arguments at the end of the execution
  """
  # if the source is a single file, use the apply_to_stream_and_save instead
  if((type(all_input_file) == str) and isfile(all_input_file)):
    return apply_to_stream_and_save(func,all_input_file,output_file,args,chunksize,override)

  if(override or not isfile(output_file)):
    # Open only once the output file, so just append all the content to it
    with bz2.open(output_file, 'wb') as out_file:
      for key in all_input_file:
        input_file = all_input_file[key]
        print(f'==> File "{input_file}" start')
        with pd.read_json(input_file, lines=True, compression='bz2', chunksize=chunksize) as df_reader:
          for df_chunk in tqdm(df_reader):
            new_df_chunk, args = func(df_chunk, args)
            write_df_chunk_to_file(new_df_chunk, out_file)
                        
        print(f'==> File "{input_file}" processed')
  return args

def filter_file_and_save(func, input_file, output_file, name='', chunksize=1_000_000, override=False):
  """
    filter input_file and save in output_file

    Parameters:
      func :        fonction to apply of type : new_df_chunk = func(df_chunk)
      input_file :  input file
      output_file : output file
      chunksize :   chunk size
      override :    override output_file if alredy exist
  """
  if(override or not isfile(output_file)):
    def f(df, total_nb_rows):
      new_df = func(df)
      return new_df, (total_nb_rows + df.shape[0] - new_df.shape[0])
    total_nb_rows = apply_to_all_stream_and_save(f, input_file, output_file, 0, chunksize, override)
    print(f'Number of rows dropped {name} : {total_nb_rows}')

def count_nb_rows_in_file(input_file):
  """
    count nb rows in input_file

    Parameters:
      input_file :  input file
  """
  count = 0
  with pd.read_json(input_file, lines=True, compression='bz2', chunksize=750_000) as df_reader:
    for chunk in df_reader:
     count += chunk.shape[0]

  print(f'Total number of records in {input_file}: {count}')

def sample_dataset(input_file, output_file, nb_samples, seed=1, chunksize=750_000):
  """
  Sample some quotes from the dataset
  Parameters:
    - input_file: the file from which we should load the quotes
    - output_file: the file in which we should write the sampled quotes
    - nb_samples: the number of sample quotes we should keep from the data set
    - seed: seed used to fix random operation
    - chunksize: The size of each chunk we should load in memory
  """
  if(isfile(input_file) and not isfile(output_file)):
    # First need to get back the quoteID of all the rows in the file
    quoteIDs = []
    with pd.read_json(input_file, lines=True, compression='bz2', chunksize=chunksize) as df_reader:
      for df_chunk in tqdm(df_reader):
        quoteIDs += df_chunk['quoteID'].tolist()

    print('==> quoteIDs processed')

    # Choose nb_samples uniformly at random
    rng = np.random.default_rng(seed=seed)
    quoteIDs_samples = rng.choice(quoteIDs, nb_samples, replace=False)

    with pd.read_json(input_file, lines=True, compression='bz2', chunksize=chunksize) as df_reader:
      with bz2.open(output_file, 'wb') as out_file:
        for df_chunk in tqdm(df_reader):
          # Keep only chosen rows
          df_result = df_chunk.loc[df_chunk['quoteID'].isin(quoteIDs_samples), :]

          # Write result chunk to file
          write_df_chunk_to_file(df_result, out_file)

    print(f'==> Succesfully sampled {nb_samples} out of {len(quoteIDs)} from file "{input_file}"')

def cast_date_remove_authors_low_proba(df_chunk, nb_rows_dropped):
  """
    cast the date and remove authors with low probability

    Parameters:
      df_chunk :        chunk of the data frame
      nb_rows_dropped : number of rows dropped

    Returns:
      df_chunk :        chunk of the data frame updated
      nb_rows_dropped : number of rows dropped updated
  """
  def create_col_author_highest_proba(row):
    """
    Return the author with the highest probability and its associated probablity
    Parameters:
      row: row of the Quotebank datatest
    Returns:
      Tuple of the author with its associated probability
    """
    max_proba = -1.0
    max_author = None

    for author, proba in row['probas']:
      if float(proba) > max_proba:
        max_proba = float(proba)
        max_author = author

    return max_author, max_proba


  # Cast the date column to datetime
  df_chunk['date'] = pd.to_datetime(df_chunk['date'])

  # Cast the string 'None' for the speaker column to proper np.nan
  df_chunk['speaker'] = df_chunk['speaker'].replace('None', np.nan)

  # Drop all the rows where the author is nan
  df_chunk.dropna(subset=['speaker'], inplace=True)

  tmp = pd.DataFrame()
  # Create 2 new columns with author, proba that has the highest proba
  tmp[['author_highest_proba', 'highest_proba']] = df_chunk.apply(create_col_author_highest_proba, axis=1, result_type='expand')

  # Check if for some rows the author is not the author with the highest proba
  if not df_chunk['speaker'].equals(tmp['author_highest_proba']):
    print('========================================================================')
    print('The column "speaker" is not equal to the column "author_highest_proba" !')
    print('========================================================================')

    # Print where the 2 columns are different
    print(df_chunk[np.argwhere(not df_chunk['speaker'].equals(tmp['author_highest_proba']))])

  # Drop the rows where the highest proba is < 0.5
  mask = tmp['highest_proba'] < 0.5
  nb_rows_dropped += mask.sum()
  df_chunk = df_chunk[~mask]

  return df_chunk, nb_rows_dropped