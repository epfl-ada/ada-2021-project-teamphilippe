import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime

def get_labels_from_QID(wikidata_labels_quotebank):
  """
    return a map from QID to labels

    Parameters: 
      wikidata_labels_quotebank : path of wikidata
    Returns:
      labels_from_QID : a map from QID to labels
      desc_from_QID   : a map from QID to description
  """
  labels_from_QID = {}
  desc_from_QID = {}
  with pd.read_csv(wikidata_labels_quotebank, compression='bz2', chunksize=1_000_000) as df_reader:
    for chunk in df_reader:
      for index, row in chunk.iterrows():
        labels_from_QID[str(row["QID"])] = str(row["Label"])
        desc_from_QID[str(row["QID"])] = str(row["Description"])
  return labels_from_QID, desc_from_QID

def get_Label(QIDs, labels_from_QID):
  """
    get the label from QIDs

    Parameters:
      QIDs :            list of QIDs
      labels_from_QID : map from QID to labels

    Returns:
      labels :          list of labels
  """
  if(str(QIDs) == 'None'):
    return None
  result = [labels_from_QID[str(x)] for x in QIDs if str(x) in labels_from_QID]
  return result if len(result) > 0 else None

def first_if_not_None(element):
  """
    return the first element or None

    Parameters:
      element : list of elements or None

    Returns:
      labels :  the first element or None
  """
  if (str(element) == 'None'):
    return None
  return str(element[0])[1:-10]

def set_academic_degree(row):
  if(str(row) == 'None'):
    return None
  result =  set()
  for element in row:
    e = element.lower()
    if('bachelor' in e or
        e.startswith('bsc') or 
       'licence' in e  or
       'law degree' in e or
       'political science' in e):
      e = 'Bachelor'
    elif('candidate' in e or
          'candidatus' in e or
          'cand.' in e):
      e = 'Candidate'
    elif('doctor' in e or
        'doktor' in e or
        'phd' in e or
        'dr.' in e):
      e = 'Doctor'
    elif('master' in e or
        'laurea' in e or
        e.startswith('ma') or
        'msc' in e or
        'health professional degree' in e or 
        'aggregation of modern literature' in e): 
      e = 'Master'
    elif('professor' in e): 
      e = 'Professor'
    else:
      e = 'Other'
    result.add(e)
  return result if len(result) > 0 else None

def set_continent(nationalitys, continents):
  eu = [
    'Germany',
    'United Kingdom',
    'France',
    'Italy',
    'Spain',
    'Ukraine',
    'Poland',
    'Romania',
    'Netherlands',
    'Belgium',
    'Czech Republic',
    'Greece',
    'Portugal',
    'Sweden',
    'Hungary',
    'Belarus',
    'Austria',
    'Serbia',
    'Switzerland',
    'Bulgaria',
    'Denmark',
    'Finland',
    'Slovakia',
    'Norway',
    'Ireland',
    'Croatia',
    'Moldova',
    'Bosnia',
    'Herzegovina',
    'Albania',
    'Lithuania',
    'North Macedonia',
    'Slovenia',
    'Latvia',
    'Estonia',
    'Montenegro',
    'Luxembourg',
    'Malta',
    'Iceland',
    'Andorra',
    'Monaco',
    'Liechtenstein',
    'San Marino',
    'Holy See']
  am = [
    'Canada',
    'Greenland',
    'Mexico',
    'USA',
    'United States Map',
    'Anguilla',
    'Antigua and Barbuda',
    'Aruba',
    'Bahamas',
    'Barbados',
    'Bermuda',
    'British Virgin Islands',
    'Cayman Islands',
    'Cuba',
    'Curaçao',
    'Dominica',
    'Dominican Republic',
    'Grenada',
    'Guadeloupe',
    'Haiti',
    'Jamaica',
    'Martinique',
    'Montserrat',
    'Puerto Rico',
    'Saint Kitts and Nevis',
    'Saint Lucia',
    'Saint Vincent and the Grenadines',
    'Trinidad and Tobago',
    'US Virgin Islands',
    'Belize',
    'Costa Rica',
    'El Salvador',
    'Guatemala',
    'Honduras',
    'Nicaragua',
    'Panama',
    'Argentina',
    'Bolivia',
    'Brazil',
    'Chile',
    'Colombia',
    'Ecuador',
    'French Guiana',
    'Guyana',
    'Paraguay',
    'Peru',
    'Suriname',
    'Uruguay',
    'Venezuela']
  ai = [
    'Russia',
    'China',
    'India',
    'Indonesia',
    'Pakistan',
    'Bangladesh',
    'Japan',
    'Philippines',
    'Vietnam',
    'Turkey',
    'Iran',
    'Thailand',
    'Myanmar',
    'Iraq',
    'Afghanistan',
    'Saudi Arabia',
    'Uzbekistan',
    'Malaysia',
    'Yemen',
    'Nepal',
    'Korea',
    'Sri Lanka',
    'Kazakhstan',
    'Syria',
    'Cambodia',
    'Jordan',
    'Azerbaijan',
    'United Arab Emirates',
    'Tajikistan',
    'Israel',
    'Laos',
    'Lebanon',
    'Kyrgyzstan',
    'Turkmenistan',
    'Singapore',
    'Oman',
    'State of Palestine',
    'Kuwait',
    'Georgia',
    'Mongolia',
    'Armenia',
    'Qatar',
    'Bahrain',
    'Timor-Leste',
    'Cyprus',
    'Bhutan',
    'Maldives',
    'Brunei']
  af = [
    'Nigeria',
    'Ethiopia',
    'Egypt',
    'Congo',
    'Tanzania',
    'Africa',
    'Kenya',
    'Uganda',
    'Algeria',
    'Sudan',
    'Morocco',
    'Angola',
    'Mozambique',
    'Ghana',
    'Madagascar',
    'Cameroon',
    'Ivoire',
    'Niger',
    'Burkina Faso',
    'Mali',
    'Malawi',
    'Zambia',
    'Senegal',
    'Chad',
    'Somalia',
    'Zimbabwe',
    'Guinea',
    'Rwanda',
    'Benin',
    'Burundi',
    'Tunisia',
    'Togo',
    'Sierra Leone',
    'Libya',
    'Congo',
    'Liberia',
    'Mauritania',
    'Eritrea',
    'Namibia',
    'Gambia',
    'Botswana',
    'Gabon',
    'Lesotho',
    'Guinea-Bissau',
    'Equatorial Guinea',
    'Mauritius',
    'Eswatini',
    'Djibouti',
    'Comoros',
    'Cabo Verde',
    'Sao Tome',
    'Principe',
    'Seychelles',
    'Réunion',
    'Western Sahara',
    'Mayotte',
    'Saint Helena']
  ou = [
    'Australia',
    'Papua New Guinea',
    'New Zealand',
    'Fiji',
    'Solomon Islands',
    'Micronesia',
    'Vanuatu',
    'Samoa',
    'Kiribati',
    'Tonga',
    'Marshall Islands',
    'Palau',
    'Tuvalu',
    'Nauru',
    'New Caledonia',
    'French Polynesia',
    'Guam',
    'Northern Mariana Islands',
    'American Samoa',
    'Cook Islands',
    'Wallis & Futuna',
    'Niue',
    'Tokelau']
  
  if(str(nationalitys) == 'None'):
    return None
  result = set()
  for nationality, continent in zip(nationalitys, continents):
    n = nationality.lower()
    c = continent.lower()
    e = None
    if('america' in c):
      e = 'America'
    elif('europe' in c):
      e = 'Europe'
    elif('asia' in c):
      e = 'Asia'
    elif('oceania' in c):
      e = 'Oceania'
    elif('africa' in c):
      e = 'Africa'
    
    if(e == None):
      for country in am:
        if(country.lower() in n or country.lower() in c):
          e = 'America'
    if(e == None):
      for country in eu:
        if(country.lower() in n or country.lower() in c):
          e = 'Europe'
    if(e == None):
      for country in ai:
        if(country.lower() in n or country.lower() in c):
          e = 'Asia'
    if(e == None):
      for country in ou:
        if(country.lower() in n or country.lower() in c):
          e = 'Oceania'
    if(e == None):
      for country in af:
        if(country.lower() in n or country.lower() in c):
          e = 'Africa'
    
    if(e != None):
      result.add(e)
  return result if len(result) > 0 else None

def get_speaker_atribut(wikidata_labels_quotebank, wikidata_speaker_parquet):
  """
    return a map from QID to speaker atribut

    Parameters: 
      wikidata_labels_quotebank : path of wikidata labels
      wikidata_speaker_parquet  : path of wikidata speaker parquet
    Returns:
      speaker_atribut : a map from QID to speaker atribut
  """
  labels_from_QID, desc_from_QID = get_labels_from_QID(wikidata_labels_quotebank)
  speaker_atribut = {}

  for file in wikidata_speaker_parquet:
    df = pq.read_pandas(file)
    for i in range(len(df['aliases'])):
      result = {
        'name' : str(df['label'][i]),
        'date_of_birth' : first_if_not_None(df['date_of_birth'][i]),
        'gender' : get_Label(df['gender'][i], labels_from_QID),
        'nationality' : get_Label(df['nationality'][i], labels_from_QID),
        'continent' : get_Label(df['nationality'][i], desc_from_QID),
        'ethnic_group' : get_Label(df['ethnic_group'][i], labels_from_QID),
        'occupation' : get_Label(df['occupation'][i], labels_from_QID),
        'party' : get_Label(df['party'][i], labels_from_QID),
        'academic_degree' : get_Label(df['academic_degree'][i], labels_from_QID),
        'candidacy' : get_Label(df['candidacy'][i], labels_from_QID),
        'religion' : get_Label(df['religion'][i], labels_from_QID)
      }
      result['academic_degree_group'] = set_academic_degree(result['academic_degree'])
      result['continent'] = set_continent(result['nationality'], result['continent'])
      if(result['date_of_birth'] != None or 
         result['gender'] != None or
         result['nationality'] != None or
         result['ethnic_group'] != None or
         result['occupation'] != None or
         result['party'] != None or
         result['academic_degree'] != None or
         result['candidacy'] != None or
         result['religion'] != None):
        speaker_atribut[str(df['id'][i])] = result

  return speaker_atribut

def add_speaker_atribut(df_chunk, args):
  """
    add speaker atribut to the data frame 

    Parameters:
      df_chunk :          chunk of the data frame
      args :
        speaker_atribut:  map from QID to speaker atribut
        drops:            number of rows dropped

    Returns:
      new_df_chunk :      chunk of the data frame updated
      args :
        speaker_atribut:  map from QID to speaker atribut
        drops:            number of rows dropped updated
  """
  speaker_atribut, drops = args

  def get_qid(row):
    """
    get qid of the speaker

    Parameters:
      row : row of the data frame

    Returns:
      qid : qid of the speaker
    """
    if('qids' in row):
      qids = row['qids']
      if(str(qids) != 'None'):
        # Filters out misclassifications if key_words is defined
        if('key_words' in row):
          words = row['key_words']
          # check if one of the qids respects the conditions
          for id in qids:
            if(id in speaker_atribut):
              name = speaker_atribut[id]['name']
              valid = True
              for word in words:
                valid &= word not in name
              if(valid):
                return str(id)
            else :
              return str(id)
        else:
          return str(qids[0])
    else:
      return str(row['qid']) if('qid' in row) else None
    return None

  def compute_age(row):
    """
    compute the age of the speaker

    Parameters:
      row : row of the data frame

    Returns:
      age : age of the speaker
    """
    date_of_birth = speaker_atribut[row['qid']]['date_of_birth']
    date = row['date']
    if(date == None or date_of_birth == None):
      return None
    date_of_birth = date_of_birth\
      .replace('-00', '-01')\
      .replace('-02-31', '-02-29')\
      .replace('-06-31', '-06-30')
    date = datetime.strptime(str(date),'%Y-%m-%d %H:%M:%S')
    try:
      date_of_birth = datetime.strptime(date_of_birth,'%Y-%m-%d')
      return (date - date_of_birth).days//365.25
    except ValueError:
      try:
        #arrive there if the 29 february is not a valid date for this year
        date_of_birth = date_of_birth\
          .replace('-02-29', '-02-28')
        date_of_birth = datetime.strptime(date_of_birth,'%Y-%m-%d')
        return (date - date_of_birth).days//365.25
      except ValueError:
        #unknown exception
        print()
        print('ValueError:')
        print('date_of_birth', date_of_birth)
        print()
    return None

  def get_atribut(new_df_chunk, speaker_atribut, atribut):
    return new_df_chunk.apply(lambda row : speaker_atribut[row['qid']][atribut], axis = 1)
  
  df_chunk['qid'] = df_chunk.apply(get_qid, axis = 1)
  new_df_chunk = df_chunk[df_chunk['qid'].apply(lambda qid : qid != None and qid in speaker_atribut)]
  new_df_chunk['speaker'] = new_df_chunk['qid'].apply(lambda qid : speaker_atribut[qid]['name'])
  new_df_chunk = new_df_chunk.drop(['qids', 'probas', 'urls', 'phase'], axis = 1)

  new_df_chunk['speaker_age'] = new_df_chunk.apply(compute_age, axis = 1)
  new_df_chunk['speaker_gender'] = get_atribut(new_df_chunk, speaker_atribut, 'gender')
  new_df_chunk['speaker_nationality'] = get_atribut(new_df_chunk, speaker_atribut, 'nationality')
  new_df_chunk['speaker_continent'] = get_atribut(new_df_chunk, speaker_atribut, 'continent')
  new_df_chunk['speaker_ethnic_group'] = get_atribut(new_df_chunk, speaker_atribut, 'ethnic_group')
  new_df_chunk['speaker_occupation'] = get_atribut(new_df_chunk, speaker_atribut, 'occupation')
  new_df_chunk['speaker_party'] = get_atribut(new_df_chunk, speaker_atribut, 'party')
  new_df_chunk['speaker_academic_degree'] = get_atribut(new_df_chunk, speaker_atribut, 'academic_degree')
  new_df_chunk['speaker_academic_degree_group'] = get_atribut(new_df_chunk, speaker_atribut, 'academic_degree_group')
  new_df_chunk['speaker_candidacy'] = get_atribut(new_df_chunk, speaker_atribut, 'candidacy')
  new_df_chunk['speaker_religion'] = get_atribut(new_df_chunk, speaker_atribut, 'religion')

  drops += len(df_chunk) - len(new_df_chunk)
  return new_df_chunk, (speaker_atribut, drops)

speaker_atribut = {}