#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:10:40 2024

@author: rodrigo
"""

import pandas as pd
import glob


def get_data(database, files=None, time = 'daily'):

    if files == None:
        
        csv_list = glob.glob(f"./database/{database}/{time}/*.csv")
        
    else:
        
        csv_list = [f"./database/{database}/{time}/{file}.csv" for file in files]
        
    csv_list.sort()
    
    dfs = []
    
    for csv in csv_list:
        
        df = pd.read_csv(csv,index_col="Date")
        
        df.sort_index(inplace=True)
        
        df.dropna(inplace=True)
        
        dfs.append(df)
      
    return dfs
 

if __name__ == '__main__':
    
    database ="10_est"
    files=["58880001","58235100"]
    
    # database="bacias"                                        #TODO: Alterar ou remover isso aqui
    # files=["PARAIBA DO SUL ANTA","PARAIBA DO SUL FUNIL"]     #TODO: Alterar ou remover isso aqui

    df=get_data(database,files,"daily")