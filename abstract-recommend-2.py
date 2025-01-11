#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
#from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
#from sklearn.metrics.pairwise import cosine_similarity
import json

# pip list --format=freeze > requirements.txt

#@st.cache(allow_output_mutation=True)
@st.cache_data
def cached_model():
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    #df = pd.read_csv('rheology_dataset_em.csv')
    df = pd.read_excel('abstract_em.xlsx', header = 0)
    df['Abstract_embeddings'] = df['Abstract_embeddings'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('Food Processing Lab - 포스터 초록 추천시스템')
#st.markdown("[세종대학교 식품가공학 연구실](http://home.sejong.ac.kr/~suyonglee/)")
st.write("(찾고 싶은 주제 키워드는 되도록 영어로 입력하시오. 예) buckwheat 관련 포스터는?)")
#st.caption('(아직 학습 중이오니 어색한 답변 이해부탁드려요.)')


with st.form('form', clear_on_submit=True):
    top_k = st.slider(
        "보여주는 포스터 개수",
        5,  # 시작 값
        10,  # 끝 값
        value = 5
    )

    user_input = st.text_input('Questions: ', '')
    submitted = st.form_submit_button('Search')

if submitted and user_input:

    #top_k = 5  # 인코딩 된 것 중 유사도 높은 문서 수

    def get_query_sim_top_k(query, model, df, top_k):
        query_encode = model.encode(query)
        cos_scores = util.pytorch_cos_sim(query_encode, df['Abstract_embeddings'])[0]
        top_results = torch.topk(cos_scores, k=top_k)
        return top_results

    top_result = get_query_sim_top_k(user_input, model, df, top_k)

#    result = df.iloc[top_result[1].numpy(), :][['Name', 'Title', 'Abstract']]

    df['Year']=df['Year'].astype(str)

    result = df.iloc[top_result[1].numpy(), :][['Year', 'Name', 'Title', 'Abstract']]

    st.dataframe(result)


