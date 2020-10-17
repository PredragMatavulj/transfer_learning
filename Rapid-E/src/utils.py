# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 03:24:11 2020

@author: Slobodan
"""
import os
import pandas as pd
import logging
import datetime

def open_excel_file_in_pandas(file_path):
    try:
        df = pd.read_excel(file_path, header=0)
    except FileNotFoundError:
        raise error(f'File {file_path} does not exist. Please, provide correct filepath.')
    return df

def save_pandas_to_excel_file(df,file_path):
    df.to_excel(file_path)

def select_pollen(dfH, dfP):
    pollen_codes = list(dfP['CODE'])
    pollen_codes_selected = []
    pollen_codes_skipped = []
    for code in pollen_codes:
        if code in list(dfH.columns):
            pollen_codes_selected.append(code)
        else:
            pollen_codes_skipped.append(code)
    
    if (len(pollen_codes_skipped) > 0):
        message = 'Following pollen types are not found in Hirst data collection: \n'
        #logging.warning()
        for code in pollen_codes_skipped:
            message += '\t' + code +' - ' + dfP[dfP.CODE == code].LATIN.to_string(index=False) +'\n'
        message += 'These pollen types will be skipped.'
        logging.warning(message)
        
    dfH = dfH[['HOUR'] + pollen_codes_selected]
    return dfH

def exclude_calibration_hours(dfH, dfC):
    dfH = dfH[dfH.HOUR.isin(dfC.Time)]
    return dfH

def read_data_dir(dir_path):
    fnames = os.listdir(dirpath)
    dt = [];
    for fname in fnames:
        if os.path.exists(os.path.join(dir_path,fname)):
            data = pickle.load(fp)
            dt.append([datetime.datetime.strptime(fname[-4] + ':00:00', '%Y-%m-%d %H:%M:%S'),fname, len(data[0])])
    df = pd.DataFrame(dt, columns = ['HOUR','FILENAME', 'TOTAL'])
    return df

def join_hirst_rapid_data(dfH,dfR):
    pollen_codes = dfH.columns[1:]
    df = dfH.join(dfR,on='HOUR',how='inner').reindex(columns=['HOUR', 'FILENAME', 'TOTAL'] + pollen_codes)
    return df

def set_time_resolution(df, res='hour'):
    if res == 'hour':
        return df
    elif res == 'day':
        df['DATE'] = list(map(lambda x: x.date(), df['HOUR']))
        colnames = ['TOTAL'] + df.columns[3:]
        gr = df.groupby('DATE').filter(lambda x: len(x['HOUR']) == 24).groupby('DATE')
        df = gr[colnames].sum()
        df['DATE'] = df.index
        return df
    else:
        raise error(f'{res} is not supported aggregation method.')
        







    
    
   #dfH['HOUR'] = list(map(lambda x: x[:-12],list(dfH['HOUR'])))
    

        #dfC = open_excel_file_in_pandas(calib_info_path)
        #dfS = select





#dfH = open_excel_file_in_pandas(hirst_data_path)
#dfC = open_excel_file_in_pandas(pollen_data_path)