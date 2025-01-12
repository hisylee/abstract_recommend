#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import json

# Streamlit 캐시 데코레이터 업데이트
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

@st.cache_data
def load_dataset():
    try:
        df = pd.read_excel('abstract_em.xlsx', header=0)
        df['Abstract_embeddings'] = df['Abstract_embeddings'].apply(json.loads)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

# 모델 및 데이터셋 로드
model = load_model()
df = load_dataset()

# Streamlit UI
st.title('Food Processing Lab - 포스터 초록 추천시스템')
st.write("(찾고 싶은 주제 키워드는 되도록 영어로 입력하시오. 예: buckwheat 관련 포스터는?)")

with st.form('search_form', clear_on_submit=True):
    top_k = st.slider("보여주는 포스터 개수", 5, 10, value=5)
    user_input = st.text_input('검색 키워드: ', '')
    submitted = st.form_submit_button('검색')

if submitted and user_input:
    def get_query_sim_top_k(query, model, df, top_k):
        try:
            query_encode = model.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_encode, df['Abstract_embeddings'].tolist())[0]
            top_results = torch.topk(cos_scores, k=top_k)
            return top_results
        except Exception as e:
            st.error(f"Error calculating similarity: {e}")
            return None

    top_result = get_query_sim_top_k(user_input, model, df, top_k)
    
    if top_result is not None:
        df['Year'] = df['Year'].astype(str)
        result = df.iloc[top_result[1].numpy(), :][['Year', 'Name', 'Title', 'Abstract']]
        st.dataframe(result)
    else:
        st.warning("결과를 불러오는 데 문제가 발생했습니다.")


# In[ ]:




