import random

import seaborn as sns
import time
import os
from statsmodels.tsa.arima.model import ARIMA
import gc
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
sns.set(rc={"figure.dpi":100, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")
sns.set_theme()
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VARMAX
import matplotlib.ticker as mticker
def format_func(value, tick_number):
    formatted_value = "{:,.0f}".format(value).replace(","," ")
    return formatted_value
import mplcyberpunk

plt.style.use("cyberpunk")
generate_image = False
NOMBRE_ANNEES = 6
AVAIABLE_RANKS_LAST_YEAR = 6
#UTILITAIRES 
def columns_ordering(nombre_annee, nombre_rang, index_name='IDENT'):
    cols = [index_name]
    for annee in range(1,nombre_annee+1):
        for rang in range(1,nombre_rang+1):
            cols.append('CFR-'+str(rang)+'-n-'+str(nombre_annee-annee+1))
            cols.append('PFR-'+str(rang)+'-n-'+str(nombre_annee-annee+1))
            cols.append('CAF-'+str(rang)+'-n-'+str(nombre_annee-annee+1))
            cols.append('CIP-'+str(rang)+'-n-'+str(nombre_annee-annee+1))
    return cols
def series_temporelles(tableau, nombre_series,rem_f = True):#Coupe les données en 3 séries temporelles (CAF,CIP,CPP)
    tableau.pop(0)
    n = len(tableau)
    nombre_iter= int(n/nombre_series)
    listes = [[] for i in range(nombre_series)]
    for i in range(nombre_iter):
        for j in range(nombre_series):
            listes[j].append(tableau[i*nombre_series+j])

    return listes

#RECUPERATION DES DONNÉES CFR_PFR SUR LES ANNEES ANTÉRIEURES
generate_image = False

print("Récupération des données CFR_PFR antérieures\n")
years = [i for i in range(2018,2023)]
data_by_year = [pd.read_csv('donnes_'+str(year)+'_CFR_PFR.csv', sep=';')for year in years]
l = data_by_year[2]['IDENT'].values.tolist()
ident_index = pd.DataFrame([*set(l)])
ident_index.rename(columns={0:'IDENT'},inplace=True)
annee_reference = 2023#Dernière année de donnée 

years = [i for i in range(2018,2023)]#Année des données
data_merged_years = []
for year in years:
    data_merged= data_by_year[-years[0]+year]
    data_merged = data_merged[['IDENT','CFR','PFR','CAF_BRUTE','EXER','RANG']]
    data_merged = data_merged[data_merged['RANG']==1]
    data_merged = data_merged.rename(columns={'CFR':'CFR-1-n-'+str(annee_reference-year),'PFR':'PFR-1-n-'+str(annee_reference-year), 'CAF_BRUTE':'CAF-1-n-'+str(annee_reference-year)})
    data_merged = pd.merge(data_merged,ident_index,on='IDENT',how='right')#
    data_merged.fillna(0,inplace=True)#
    for rang in range(2,13):
        data = data_by_year[-years[0]+year] 
        data = data[data['RANG']==rang]
        data = data[['IDENT','CFR','PFR','CAF_BRUTE']]
        data.rename(columns={'CFR':'CFR-'+str(rang)+'-n-'+str(annee_reference-year), 'PFR':'PFR-'+str(rang)+'-n-'+str(annee_reference-year),'CAF_BRUTE':'CAF-'+str(rang)+'-n-'+str(annee_reference-year)},inplace=True)
        data_merged = pd.merge(data_merged,data, on='IDENT',how='left')
    ###RANG 13
    data = data_by_year[-years[0]+year]
    rang = 13
    if(data[data['RANG']==0].count()['CFR'] !=0):
        data = data[data['RANG']==0]
        data = data[['IDENT','CFR','PFR','CAF_BRUTE']]
        data.rename(columns={'CFR':'CFR-'+str(rang)+'-n-'+str(annee_reference-year), 'PFR':'PFR-'+str(rang)+'-n-'+str(annee_reference-year),'CAF_BRUTE':'CAF-'+str(rang)+'-n-'+str(annee_reference-year)},inplace=True)
        data_merged = pd.merge(data_merged,data, on='IDENT',how='left')
    else:
        data = data_by_year[-years[0]+year]
        data = data[data['RANG']==rang]
        data = data[['IDENT','CFR','PFR','CAF_BRUTE']]
        data.rename(columns={'CFR':'CFR-'+str(rang)+'-n-'+str(annee_reference-year), 'PFR':'PFR-'+str(rang)+'-n-'+str(annee_reference-year),'CAF_BRUTE':'CAF-'+str(rang)+'-n-'+str(annee_reference-year)},inplace=True)
        data_merged = pd.merge(data_merged,data, on='IDENT',how='left')




    data_merged.drop(['RANG','EXER'],axis=1, inplace=True)
    data_merged.fillna(0,inplace=True)
    data_merged_years.append(data_merged)
total = data_merged_years[0].copy()
for i in range(1,len(years)):
    total = pd.merge(total,data_merged_years[i], on='IDENT',how='left')
    total.fillna(0,inplace=True)
###
#RECUPERATION DES DONNÉES CIP SUR LES ANNEES ANTÉRIEURES
print("Récupération des données CIP antérieures\n")
years = [2018,2019,2020,2021,2022]
annee_reference = 2023
data_by_year = [pd.read_csv('CIP_471_'+str(year)+'.csv',sep=';',low_memory=False) for year in years]
data_merged_years = []
for year in years:
    data_merged = data_by_year[-years[0]+year]
    data_merged = data_merged[['ident','solde','rang']]
    data_merged = data_merged[data_merged['rang']==1]
    data_merged = data_merged.rename(columns={'solde':'CIP-1-n-'+str(annee_reference-year)})
    
    
    for rang in range(2,14):
        data = data_by_year[-years[0]+year]
        data = data[data['rang']==rang]
        data = data[['ident','solde']]
        data.rename(columns={'solde':'CIP-'+str(rang)+'-n-'+str(annee_reference-year)},inplace=True)
        data_merged = pd.merge(data_merged,data, on='ident')
    data_merged_years.append(data_merged)

total_CIP = data_merged_years[0].copy()
for i in range(1,len(years)):
    data_merged_years[i].drop(['rang'],axis=1, inplace=True)
    total_CIP = pd.merge(total_CIP,data_merged_years[i],on='ident')
total_CIP.drop(['rang'],axis=1, inplace=True)
total_CIP.rename(columns={'ident':'IDENT'},inplace=True)

#FUSION DES DONNES CIP ET CFR ET MISE EN ORDRE
full_data = pd.merge(total, total_CIP, on='IDENT',how='left')
full_data.fillna(0,inplace=True)
full_data_t = full_data[columns_ordering(5,13)]

#DONNEES SUR LA DERNIERE ANNEE

avaible_ranks = AVAIABLE_RANKS_LAST_YEAR

print("Récupération des données contemporaines\n")
#DONNES MI-ANNEE CRF_PRF
year = 2023
donnes_2022 = pd.read_csv('donnes_'+str(2023)+'_CFR_PFR.csv', sep=';')
data_merged= donnes_2022.copy()
data_merged = data_merged[['IDENT','CFR','PFR','CAF_BRUTE','EXER','RANG']]
data_merged = data_merged[data_merged['RANG']==1]
data_merged = data_merged.rename(columns={'CFR':'CFR-1-n-'+str(annee_reference-year),'PFR':'PFR-1-n-'+str(annee_reference-year), 'CAF_BRUTE':'CAF-1-n-'+str(annee_reference-year)})
data_merged = pd.merge(data_merged,ident_index,on='IDENT',how='right')#
data_merged.fillna(0,inplace=True)#
for rang in range(2,avaible_ranks+1):
        data = donnes_2022
        data = data[data['RANG']==rang]
        data = data[['IDENT','CFR','PFR','CAF_BRUTE']]
        data.rename(columns={'CFR':'CFR-'+str(rang)+'-n-'+str(annee_reference-year), 'PFR':'PFR-'+str(rang)+'-n-'+str(annee_reference-year),'CAF_BRUTE':'CAF-'+str(rang)+'-n-'+str(annee_reference-year)},inplace=True)
        data_merged = pd.merge(data_merged,data, on='IDENT',how='left')
data_merged.drop(['RANG','EXER'],axis=1, inplace=True)
data_merged.fillna(0,inplace=True)
total_2022 = data_merged.copy()
#DONNES MI-ANNEE, CIP
year = 2023

donnes_2022 = pd.read_csv('CIP_471_2023.csv',sep=';')
data_merged = donnes_2022.copy()

data_merged = data_merged[['ident','solde','rang']]
data_merged = data_merged[data_merged['rang']==1]
data_merged = data_merged.rename(columns={'solde':'CIP-1-n-'+str(annee_reference-year)})
   
    
for rang in range(2,avaible_ranks+1):
    data = donnes_2022
    data = data[data['rang']==rang]
    data = data[['ident','solde']]
    data.rename(columns={'solde':'CIP-'+str(rang)+'-n-'+str(annee_reference-year)},inplace=True)
    data_merged = pd.merge(data_merged,data, on='ident')
    data_merged_years.append(data_merged)
data_merged.drop('rang',axis=1,inplace=True) 

data_merged.rename(columns={'ident':'IDENT'},inplace=True)

total_2022 = pd.merge(total_2022, data_merged,on='IDENT',how='left')
total_2022.fillna(0,inplace=True)
total_2022
columns = ['IDENT']
for i in range(1,avaible_ranks+1):
    columns.append('CFR-'+str(i)+'-n-0')
    columns.append('PFR-'+str(i)+'-n-0')
    columns.append('CAF-'+str(i)+'-n-0')
    columns.append('CIP-'+str(i)+'-n-0')
total_2022 = total_2022[columns]

#RÉCUPÉRATION DES MÉTA-DONNÉES (NOM DES COMMUNES, DÉPARTEMENTS)
corresp = pd.read_csv('Corresp.csv',sep=';')
siren_list = corresp['IDENT'].tolist()


def work(siren):
        worked = True
        try:
            
            
            
            print('Traitement de :'+str(siren)+'\n')
            fetch_commune = corresp[corresp['IDENT']==siren]
            nom_commune = np.array(fetch_commune['LCOMM']).tolist()[0]
            departement = np.array(fetch_commune['NDEPT']).tolist()[0]
            test = full_data_t[full_data_t['IDENT']==	siren]
            test_real = total_2022[total_2022['IDENT']==	siren]
            ranks_2023 = 6
            col = ['IDENT']
            for rank in range(1,ranks_2023+1):
                col.append('CFR-'+str(rank)+'-n-0')
                col.append('PFR-'+str(rank)+'-n-0')
                col.append('CAF-'+str(rank)+'-n-0')
                col.append('CIP-'+str(rank)+'-n-0')
            
            to_add_2023 = test_real[col]
            test = pd.merge(test, to_add_2023, on='IDENT',how='left')
            print("ici")
            curves_training = series_temporelles(np.array(test).tolist()[0],4)
            data_real = series_temporelles(np.array(test_real).tolist()[0],4)
            curves_real = [(pd.DataFrame(data_real[i]).transpose().rename(columns={0:65,1:66,2:67,3:68,4:69,5:70,6:71})).transpose() for i in range(4)]
            final = pd.DataFrame()
            data = pd.DataFrame(curves_training).transpose().rename(columns={0:'CFR',1:'PFR',2:'CAF',3:'CIP'})
            
            data.drop('CAF',axis=1,inplace=True)
            data.drop('CIP',axis=1,inplace=True)
            data.drop('CFR',axis=1,inplace=True)
            data.fillna(0.1,inplace=True)
            
            model = ARIMA(data,order=(13,2,13) )
            results = model.fit()#Entrainement du modèle
            result = results.get_forecast(13)
            fcast = result.summary_frame(0)#FCAST = PFR
            final['pfr'] = pd.concat([data['PFR'], fcast['mean']])
            data = pd.DataFrame(curves_training).transpose().rename(columns={0:'CFR',1:'PFR',2:'CAF',3:'CIP'})
            val_CAF = data['CAF'].copy()#ON RECUPERER DATA_CAF
           
            data.drop('CAF',axis=1,inplace=True)
            data.drop('PFR',axis=1,inplace=True)
            data.drop('CIP',axis=1,inplace=True)
            model = ARIMA(data,order=(13,2,13) )
            results = model.fit()#Entrainement du modèle
            result = results.get_forecast(13)
            
            ancien_cfr = data['CFR'].copy()
            fcast = result.summary_frame(0)#FCAST = CFR
            final['cfr'] = pd.concat([data['CFR'], fcast['mean']])
            final['caf'] = final['pfr']-final['cfr']
            final.transpose().to_csv('data_second/donne_'+str(siren)+'.csv')
            script_dir = os.getcwd()
            results_dir = os.path.join(script_dir, 'images_new2/'+str(departement))
            sample_file_name = "/prediction_"+str(nom_commune)+"_"+str(siren)

            del final
            del fcast
            del model
            del data
            del results
            del result
            del curves_real
            del data_real 
            del test
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            
            
            #
           


        except Exception as e:
            print(e)
            worked = False
            f = open('not_working.txt','a+')
            f.write(str(siren)+'\n')

        gc.collect()
      
#A CE POINT, TOUTES LES DONNÉES ONT ÉTÉ RÉCUPÉRÉES 
f = open('done.txt','r+')
nb_work_done = int(f.readline())
siren_list = siren_list[nb_work_done:]
print('Nombre de communes déjà traitées: '+str(nb_work_done))


                 
num_cores = 16

Parallel(n_jobs=num_cores)(delayed(work)(params) for params in siren_list)



