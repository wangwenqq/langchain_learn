import re

import pandas as pd
import matplotlib.pyplot as plt
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)
from sentence_transformers import SentenceTransformer
import os

from transformers import AutoTokenizer

os.environ["http_proxy"] = "http://127.0.0.1:10809"
os.environ["https_proxy"] = "http://127.0.0.1:10809"
text = ('诗歌的表现手法很多，我国最早流行而至今仍常使用的传统表现手法有"赋、比、兴"。'
        '《毛诗序》说："故诗有六义焉：一曰风，二曰赋，三曰比，四曰兴，五曰雅，六曰颂。'
        '"其间有一个绝句叫："三光日月星，四诗风雅颂"。这"六义"中，"风、雅、颂"'
        '是指《诗经》的诗篇种类，"赋、比、兴"就是诗中的表现手法。')

# 按字符切割
text_splitter = CharacterTextSplitter(
    separator='',
    chunk_size=5,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=False
)
#
# print(text_splitter.split_text(text.txt))

# 递归切割
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=False,
    separators=['\n\n', '\n', ' ', '']
)

chunk_doc = recursive_splitter.create_documents([text])
# print(chunk_doc)

# embedding支持的最大token长度
embedding_name = 'BAAI/bge-large-zh-v1.5'
# print(SentenceTransformer(embedding_name).max_seq_length)

tokenizer = AutoTokenizer.from_pretrained(embedding_name)
length = [len(tokenizer.encode(doc.page_content)) for doc in chunk_doc]
fig = pd.Series(length).hist()
plt.show()
