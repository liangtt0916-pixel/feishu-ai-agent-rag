import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from config import PRODUCTS_FILE, DATA_DIR, VECTOR_DIR, OPENAI_API_KEY, OPENAI_BASE_URL

# Embedding model (must support embeddings)
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def load_documents():
    docs = []

    df = pd.read_csv(PRODUCTS_FILE)
    for _, row in df.iterrows():
        product_text = f"""
型号：{row['model']}
类别：{row['category']}
品牌：{row['brand']}
价格带：{row['price_range']}
产品特点：{row['features']}
适用场景：{row['scenes']}
说明书文件：{row['manual_file']}
"""
        docs.append(
            Document(
                page_content=product_text.strip(),
                metadata={
                    "type": "product_table",
                    "model": row["model"],
                    "category": row["category"],
                    "brand": row["brand"],
                }
            )
        )

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "manual",
                        "file": filename
                    }
                )
            )

    return docs

def main():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    def _create_embeddings(use_base_url: bool):
        kwargs = {
            "api_key": OPENAI_API_KEY,
            "model": EMBEDDING_MODEL,
        }
        if use_base_url and OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)

    embeddings = _create_embeddings(use_base_url=True)

    try:
        vectorstore = FAISS.from_documents(split_docs, embeddings)
    except Exception as e:
        print("第一次尝试构建向量索引失败：", e)
        print("尝试不使用自定义 BASE URL 重新构建索引...")
        embeddings = _create_embeddings(use_base_url=False)
        vectorstore = FAISS.from_documents(split_docs, embeddings)

    vectorstore.save_local(VECTOR_DIR)
    print("向量索引构建完成。")

if __name__ == "__main__":
    main()