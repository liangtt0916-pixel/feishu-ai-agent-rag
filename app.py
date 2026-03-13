import os
import re
import json
import time
import traceback
from typing import List, Optional

import requests
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    VECTOR_DIR,
    PRODUCTS_FILE,
)

load_dotenv()

app = FastAPI(title="B端采购客服智能体 Demo（OpenClaw 风格 Agent 版）")
print("=== 当前 app.py 已加载（OpenClaw 风格 Agent 版） ===")


# =========================
# 基础配置
# =========================
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
EMBEDDING_MODEL = "BAAI/bge-m3"

# 简单内存去重：message_id -> timestamp
processed_message_ids = {}
DEDUP_TTL_SECONDS = 600  # 10分钟


def clean_processed_cache():
    now = time.time()
    expired_keys = [
        k for k, v in processed_message_ids.items()
        if now - v > DEDUP_TTL_SECONDS
    ]
    for k in expired_keys:
        processed_message_ids.pop(k, None)


# =========================
# 延迟初始化资源，避免启动时直接报错
# =========================
embeddings = None
vectorstore = None
llm = None
product_df = None
startup_errors: List[str] = []


def init_resources():
    global embeddings, vectorstore, llm, product_df, startup_errors

    startup_errors = []

    # 1) 初始化 Embeddings
    try:
        embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=EMBEDDING_MODEL
        )
        print("=== Embeddings 初始化成功 ===")
    except Exception as e:
        embeddings = None
        msg = f"Embeddings 初始化失败: {str(e)}"
        startup_errors.append(msg)
        print("=== " + msg + " ===")

    # 2) 初始化向量库
    try:
        if embeddings is None:
            raise Exception("embeddings 未初始化，无法加载向量库")

        vectorstore = FAISS.load_local(
            VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("=== 向量库加载成功 ===")
    except Exception as e:
        vectorstore = None
        msg = f"向量库加载失败: {str(e)}"
        startup_errors.append(msg)
        print("=== " + msg + " ===")

    # 3) 初始化 LLM
    try:
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.2
        )
        print("=== LLM 初始化成功 ===")
    except Exception as e:
        llm = None
        msg = f"LLM 初始化失败: {str(e)}"
        startup_errors.append(msg)
        print("=== " + msg + " ===")

    # 4) 加载商品表
    try:
        product_df = pd.read_csv(PRODUCTS_FILE)
        print(f"=== 商品数据加载成功，共 {len(product_df)} 条 ===")
    except Exception as e:
        product_df = None
        msg = f"商品数据加载失败: {str(e)}"
        startup_errors.append(msg)
        print("=== " + msg + " ===")


init_resources()


# =========================
# 数据模型
# =========================
class ChatRequest(BaseModel):
    question: str


# =========================
# OpenClaw 风格 Agent
# =========================
class OpenClawStyleAgent:
    """
    这是一个 OpenClaw 风格的 Agent 封装：
    - webhook 层只负责接收消息
    - Agent 层负责任务识别、工具调用、RAG 检索、回复生成
    """

    def __init__(self):
        self.name = "b2b_procurement_agent"
        self.description = "面向B端采购场景的智能客服 Agent"

    def run(self, question: str) -> str:
        question = (question or "").strip()
        if not question:
            return "请输入你的问题。"

        intent = self.detect_intent(question)

        if intent == "compare":
            return self.handle_compare(question)

        if intent == "recommend":
            return self.handle_recommend(question)

        return self.handle_general(question)

    # -------------------------
    # Agent 内部能力
    # -------------------------
    def detect_intent(self, question: str) -> str:
        if any(k in question for k in ["区别", "对比", "差异", "哪个好"]):
            return "compare"
        if any(k in question for k in ["推荐", "适合", "预算", "巡检", "施工", "选型"]):
            return "recommend"
        if any(k in question for k in ["说明书", "手册", "资料", "参数"]):
            return "manual"
        return "general"

    def handle_compare(self, question: str) -> str:
        models = self.find_models_in_question(question)
        if len(models) >= 2:
            return self.compare_products(models[0], models[1])
        return "请提供两个明确的型号，例如：Fluke 17B+ 和 Fluke 18B+ 有什么区别？"

    def handle_recommend(self, question: str) -> str:
        return self.recommend_products(question)

    def handle_general(self, question: str) -> str:
        return self.rag_answer(question)

    def find_models_in_question(self, question: str) -> List[str]:
        if product_df is None or "model" not in product_df.columns:
            return []

        matched = []
        for model in product_df["model"].dropna().astype(str).tolist():
            if model.lower() in question.lower():
                matched.append(model)
        return matched

    def retrieve_context(self, question: str, top_k: int = 4) -> str:
        if vectorstore is None:
            return "当前知识库未成功加载，暂时无法检索资料。"

        try:
            docs = vectorstore.similarity_search(question, k=top_k)
            contents = []
            for doc in docs:
                content = getattr(doc, "page_content", "")
                if content:
                    contents.append(content)
            return "\n\n".join(contents) if contents else "未检索到相关资料。"
        except Exception as e:
            return f"知识检索失败：{str(e)}"

    def compare_products(self, model_a: str, model_b: str) -> str:
        if product_df is None:
            return "商品数据未加载成功，暂时无法进行型号对比。"

        required_cols = {"model", "category", "brand", "price_range", "features", "scenes"}
        if not required_cols.issubset(set(product_df.columns)):
            return "商品数据字段不完整，暂时无法进行型号对比。"

        row_a = product_df[product_df["model"] == model_a]
        row_b = product_df[product_df["model"] == model_b]

        if row_a.empty or row_b.empty:
            return "未找到完整的型号信息，请确认型号名称。"

        if llm is None:
            a = row_a.iloc[0]
            b = row_b.iloc[0]
            return (
                f"已识别到两个型号：{a['model']} 和 {b['model']}。\n"
                f"由于当前大模型不可用，先给出基础信息：\n"
                f"1. 产品A：类别={a['category']}，品牌={a['brand']}，价格带={a['price_range']}\n"
                f"2. 产品B：类别={b['category']}，品牌={b['brand']}，价格带={b['price_range']}\n"
                f"建议你稍后重试，或补充更具体的使用场景。"
            )

        a = row_a.iloc[0]
        b = row_b.iloc[0]

        prompt = f"""
你是一个面向B端采购的工具仪器客服助手。
请根据以下信息，输出专业、简洁的对比结论。

产品A：
型号：{a['model']}
类别：{a['category']}
品牌：{a['brand']}
价格带：{a['price_range']}
特点：{a['features']}
场景：{a['scenes']}

产品B：
型号：{b['model']}
类别：{b['category']}
品牌：{b['brand']}
价格带：{b['price_range']}
特点：{b['features']}
场景：{b['scenes']}

请按以下格式输出：
1. 核心区别
2. 适用场景区别
3. 采购建议
"""
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            return f"产品对比失败：{str(e)}"

    def recommend_products(self, question: str) -> str:
        if product_df is None:
            return "商品数据未加载成功，暂时无法进行产品推荐。"

        required_cols = {"model", "category", "price_range", "features", "scenes"}
        if not required_cols.issubset(set(product_df.columns)):
            return "商品数据字段不完整，暂时无法进行产品推荐。"

        df = product_df.copy()

        category = None
        if "万用表" in question:
            category = "万用表"
        elif "电钻" in question:
            category = "电钻"

        if category:
            df = df[df["category"] == category]

        budget_match = re.search(r"预算\s*(\d+)", question)
        if budget_match:
            budget = int(budget_match.group(1))

            def in_budget(price_range):
                try:
                    low, high = str(price_range).split("-")
                    low, high = int(low), int(high)
                    return low <= budget <= high or budget >= low
                except Exception:
                    return True

            df = df[df["price_range"].apply(in_budget)]

        if df.empty:
            df = product_df.head(3)

        candidates = []
        for _, row in df.head(3).iterrows():
            candidates.append(
                f"型号：{row['model']}，类别：{row['category']}，价格带：{row['price_range']}，"
                f"特点：{row['features']}，场景：{row['scenes']}"
            )

        if llm is None:
            return "当前大模型不可用，暂时无法生成智能推荐结果。"

        prompt = f"""
你是一个B端采购选型助手。
用户问题：{question}

候选产品：
{chr(10).join(candidates)}

请输出：
1. 推荐型号（1-3个）
2. 每个型号推荐理由
3. 最后给出一句采购建议

回答要专业、简洁、像售前顾问。
"""
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            return f"产品推荐失败：{str(e)}"

    def rag_answer(self, question: str) -> str:
        context = self.retrieve_context(question)

        if llm is None:
            return f"当前大模型不可用，但我检索到以下参考资料：\n{context}"

        prompt = f"""
你是一个高端工具仪器一站式采购平台的智能客服，客户偏B端。
请严格基于提供的资料回答，不要编造。
如果资料不足，请明确说明“现有资料不足以判断”。

用户问题：
{question}

参考资料：
{context}

请输出简洁、专业、适合业务场景的答案。
"""
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            return f"RAG回答失败：{str(e)}"


agent = OpenClawStyleAgent()


# =========================
# 保留兼容函数，减少你原代码改动
# =========================
def detect_intent(question: str) -> str:
    return agent.detect_intent(question)


def answer_by_existing_logic(question: str) -> str:
    return agent.run(question)


# =========================
# 飞书 API 辅助函数
# =========================
def get_feishu_tenant_access_token() -> str:
    if not FEISHU_APP_ID or not FEISHU_APP_SECRET:
        raise Exception("未配置 FEISHU_APP_ID 或 FEISHU_APP_SECRET")

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    payload = {
        "app_id": FEISHU_APP_ID,
        "app_secret": FEISHU_APP_SECRET
    }

    resp = requests.post(url, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 0:
        raise Exception(f"获取 tenant_access_token 失败: {data}")

    return data["tenant_access_token"]


def send_feishu_message(chat_id: str, text: str):
    token = get_feishu_tenant_access_token()
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "receive_id": chat_id,
        "msg_type": "text",
        "content": json.dumps({"text": text}, ensure_ascii=False)
    }
    params = {"receive_id_type": "chat_id"}

    resp = requests.post(url, headers=headers, params=params, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()


# =========================
# 普通接口
# =========================
@app.get("/")
def home():
    return {
        "message": "客服智能体 Demo 已启动（OpenClaw 风格 Agent 版）",
        "startup_ok": len(startup_errors) == 0,
        "startup_errors": startup_errors,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_ready": llm is not None,
        "vectorstore_ready": vectorstore is not None,
        "product_data_ready": product_df is not None,
        "startup_errors": startup_errors,
    }


@app.get("/feishu/webhook")
def feishu_webhook_check():
    print("=== GET /feishu/webhook 被调用 ===")
    return {"ok": True, "msg": "feishu webhook route exists"}


@app.post("/chat")
def chat(req: ChatRequest):
    question = (req.question or "").strip()
    answer = answer_by_existing_logic(question)
    return {
        "question": question,
        "intent": detect_intent(question),
        "answer": answer
    }


# =========================
# 飞书 webhook
# =========================
@app.post("/feishu/webhook")
async def feishu_webhook(request: Request):
    print("=== 飞书 webhook 被调用 ===")

    try:
        data = await request.json()
    except Exception as e:
        print("=== 解析 JSON 失败 ===", str(e))
        return {"status": "bad_request", "detail": str(e)}

    print("=== 收到飞书事件 ===", data)

    # 飞书验证
    if data.get("type") == "url_verification":
        return {"challenge": data.get("challenge")}

    header = data.get("header", {})
    event = data.get("event", {})
    event_type = header.get("event_type", "")

    if event_type != "im.message.receive_v1":
        return {"status": "ignored", "event_type": event_type}

    message = event.get("message", {})
    sender = event.get("sender", {})
    sender_type = sender.get("sender_type", "")
    chat_id = message.get("chat_id", "")
    message_type = message.get("message_type", "")
    content_str = message.get("content", "{}")
    mentions = message.get("mentions", [])

    # 幂等去重
    message_id = message.get("message_id", "")
    clean_processed_cache()

    if message_id and message_id in processed_message_ids:
        print(f"=== 重复消息，跳过 === {message_id}")
        return {"status": "duplicate"}

    # 忽略机器人自己发出的消息
    if sender_type == "app":
        print("=== 忽略机器人自身消息 ===")
        return {"status": "skip_self"}

    if not chat_id:
        print("=== chat_id 为空 ===")
        return {"status": "no_chat_id"}

    if message_type != "text":
        print("=== 非文本消息，跳过 ===")
        return {"status": "skip_non_text"}

    try:
        content = json.loads(content_str)
        user_text = (content.get("text") or "").strip()
    except Exception as e:
        print("=== 解析消息内容失败 ===", str(e))
        user_text = ""

    # 群聊里只响应被 @ 的消息
    if isinstance(mentions, list) and len(mentions) == 0:
        print("=== 未@机器人，跳过 ===")
        return {"status": "not_mentioned"}

    if not user_text:
        print("=== 用户文本为空 ===")
        return {"status": "empty_text"}

    print("=== 用户问题 ===", user_text)

    # 在调用 Agent 前先登记，防止飞书重试重复触发
    if message_id:
        processed_message_ids[message_id] = time.time()

    try:
        answer = agent.run(user_text)
    except Exception as e:
        answer = f"处理失败：{str(e)}"
        print("=== Agent 处理失败 ===", str(e))
        traceback.print_exc()

    print("=== Agent 回答 ===", answer)

    try:
        send_feishu_message(chat_id, answer)
    except Exception as e:
        print("=== 发送飞书消息失败 ===", str(e))
        traceback.print_exc()
        return {"status": "send_failed", "detail": str(e)}

    return {"status": "ok"}