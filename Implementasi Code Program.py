#!/usr/bin/env python
# coding: utf-8

# # Implementasi Transaksi Data Retail Online

# ### Project UAS Data Mining Menggunakan Teknik Asosiasi - Algoritma A Priori

# KELOMPOK 1 
# 1. CHOSMAS MARZUKI_09021182025003
# 2. KURNIA MINARI_09021182025004
# 3. DAMA PUTRA SARPANDA_09021181924016
# 4. YULYA ANITA_09021182025001
# 5. MERI JUWITA_09021182025002

# In[1]:


pip install mlxtend  


# In[2]:


pip install apyori


# ### 1. Import Library

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from apyori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import datetime


# ### 2. Data Selection

# In[6]:


# Import dataset
df = pd.read_excel('C:/Users/User/UAS DAMING/data_retail2.xlsx')


# In[7]:


df.head()


# In[9]:


df.dtypes


# ### 3. Data Pre-processing

# ##### 3.1 Data Cleansing  

# In[25]:


df.head(20)


# In[10]:


# Mengganti kolom PERIODE menjadi tipe data datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# In[11]:


# Membersihkan ruang di deskripsi product, product category dan menghapus beberapa baris yang tidak memiliki nilai yang valid
df['PRODUCT'] = df['PRODUCT'].str.strip()
df['PRODUCT_CATEGORY'] = df['PRODUCT_CATEGORY'].str.strip()

df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)


# In[12]:


# menghapus variabel inoviceNO yang diawali dengan huruf C pada invoice numbernya
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~(df['InvoiceNo'].str[0] == 'C')]


# #### 3.2 Data Transformation  

# In[41]:


# rule/model 1 --> "JAWA TENGAH"
basket = (df[df['PROVINSI'] =="JAWA TENGAH"].groupby(['InvoiceNo', 'PRODUCT_CATEGORY'])['Quantity'].count()                                      .unstack().reset_index().fillna(0)                                      .set_index('InvoiceNo'))
basket


# In[42]:


# Menampilkan subset dari kolom 
basket.iloc[:,[0,1,2,3,4,5,6,7]].head()


# In[43]:


# Melakukan proses encoding -> Mengubah data kebentuk angka, agar sistem atau komputer dapat memahami informasi dari dataset
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.head(5)


# ### 4. Data Exploration

# In[44]:


# Membuat frequent items, rules, dan model dengan menggunakan Algoritma A Priori

frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
frequent_itemsets


# Digunakan data dari basket_sets dengan minimum nilai support 0.1/ 10%.

# In[45]:


rules1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules1 


# In[46]:


result1 = rules1[(rules1['lift'] >= 1) & 
               (rules1['confidence'] >= 0.8)]

apr_result = result1.sort_values(by='confidence', ascending=False)
apr_result


# Melakukan filter untuk nilai lift ratio lebih dari sama dengan 1 dengan tingkat confidence minimal 0.8 (lebih dari sama dengan 80%)

# In[47]:


apr_result.iloc[60]


# ### 5. Interpretation

# Produk-produk yang dibeli secara bersamaan oleh customer di daerah JAWA TENGAH terhadap rule asosiasi pada dataset dengan min_support 0.1 / 10%, min_threshold = 1, dan nilai lift sebesar lebih dari samadengan 1 serta tingkat confidence minimal yang diperhitungkan sebesar 0.8 (80%) adalah:
# - sabun, shampoo, obat-obatan, parfum dengan kosmetik.
# - kosmetik, susu, obat-obatan dengan minuman.
# - kosmetik, alat rumah tangga, minuman dengan sabun dan samphoo.
# 
# Yang mana nilai conviction dari hasil-hasil yang didapat bernilai lebih dari 1, artinya hasil nilai rules yang dibangun dapat di anggap akurat.

# In[23]:


# check barang kedua untuk kombinasi barang pertama
# pairing kombinasi dari pembelian produk pertama yang paling banyak untuk barang kedua adalah
apr_result['consequents'].value_counts()


# Produk atau barang yang menjadi kombinasi produk pertama untuk frekuensi yang paling banyak adalah kosmetik, minuman, sabun dan sampho.

# <br>

# ## RULE/MODEL LAIN - BANTEN

# In[48]:


basket = (df[df['PROVINSI'] =="BANTEN"].groupby(['InvoiceNo', 'PRODUCT_CATEGORY'])['Quantity'].count()                                      .unstack().reset_index().fillna(0)                                      .set_index('InvoiceNo'))
basket.head()


# In[49]:


# Show a subset of columns
basket.iloc[:,[0,1,2,3,4,5,6,7]].head()


# In[50]:


# Melakukan proses encoding -> Mengubah data kebentuk angka, agar sistem atau komputer dapat memahami informasi dari dataset
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.head(5)


# Kemudian melakukan encoding, dimana jika barang kurang dari sama dengan 0 maka keranjang tersebut bernilai 0 dan jika lebih dari 1 maka nilainya adalah 1, sehingga jika sebuah nota membeli barang A sebanyak 10 buah maka hanya akan dihitung 1. Karena analisis yang di gunakan menyaratkan seperti itu.

# In[51]:


# Membuat frequent items, rules, dan model dengan menggunakan Algoritma A Priori

frequent_itemsets = apriori(basket_sets, min_support=0.1, use_colnames=True)
frequent_itemsets.head() # 105 rows × 2 columns


# In[52]:


rules2 = association_rules(frequent_itemsets, metric="lift", min_threshold=2)
rules2.head() # 262 rows × 9 columns


# Jika ditetapkan nilai threshold (min_support) = 2, maka didapat frequent 2-itemset (F2) yaitu: F2 = [Biskuit, Alat Rumah Tangga], [Detergen. Alat Rumah Tangga] dan lainnya [jumlah kombinasi antara barang satu dengan lainnya berjumlah 2]

# In[53]:


result2 = rules2[ (rules2['lift'] >= 1) & 
                (rules2['confidence'] >= 0.85) ]

best_result = result2.sort_values(by='confidence', ascending=False)
best_result.head() # 17 rows × 9 columns


# Produk-produk yang dibeli secara bersamaan oleh customer di daerah Banten terhadap rule asosiasi pada dataset dengan min_support 0.1 / 10%, min_threshold = 2, dan nilai lift sebesar lebih dari samadengan 1 serta tingkat confidence minimal yang diperhitungkan sebesar 0.85 (85%) adalah: Obat-obatan, minuman, sabun, samphoo, alat rumah tangga, parfum, susu, biskuit, snack, parfum, permen, alat rumah tangga dengan kombinasi produk yang didapat sebagian besar adalah pairing dengan kosmetik.

# In[54]:


best_result.head

