#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import glob


def get_data(database, file, time = 'daily'):
        
    csv = f"../database/{database}/{time}/{file}.csv" 
    
    try:
        df = pd.read_csv(csv,index_col="Date")
    except:
        df = pd.read_csv(csv,index_col="date")
    
    if df.columns[0] == 'flow_rate':
        df = df.rename(columns={'flow_rate': file})
    
    df.sort_index(inplace=True)
    
    df.dropna(inplace=True)

    if time == "hourly" or time == "daily":

        n = int(len(df)/2)

        df = df[n:]
        
      
    return df
 

if __name__ == '__main__':
    
    database ="stations"
    file="44200000" #"58235100"
    
    # database="bacias"
    # files=["PARAIBA DO SUL ANTA","PARAIBA DO SUL FUNIL"]
    
    df=get_data(database,file,"daily")