import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisConfig, RedisVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
import config
REDIS_URL = "redis://localhost:6379"

redis_config = RedisConfig(
    index_name="cord_idx",
    redis_url=REDIS_URL,
)

# 使用 HuggingFace 的 Embedding 模型
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")  # 中文支持优秀
vector_store = RedisVectorStore(embeddings=embedding, config=redis_config)

# 读取元数据
metadata = pd.read_csv("metadata.csv")

# 只保留包含摘要的论文
cord_df = metadata[["title", "abstract", "doi", "url"]].dropna(subset=["abstract"])
random_state = 42
cord_df = cord_df.sample(frac=1, random_state=random_state).reset_index(drop=True)  # 打乱顺序
cord_df = cord_df.head(100)  # 先抽取部分测试


# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = []

for _, row in cord_df.iterrows():
    text = f"{row['title']}\n{row['abstract']}"
    docs.append(text)

ids = vector_store.add_texts(docs)
print(vector_store.similarity_search("SARS", k=2))

os.environ["OPENAI_API_KEY"] = config.API_KEY
os.environ["OPENAI_API_BASE"] = config.API_BASE_URL
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

query = "什么是SARS？"
result = qa_chain.invoke({"query": query})

print("Question:", query)
print("Answer:", result)