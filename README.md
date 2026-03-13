Feishu AI Agent with RAG（飞书智能客服 Agent）

一个基于 FastAPI + RAG + LLM 构建的 AI Agent 系统，通过 飞书机器人 提供智能客服与采购辅助能力，支持知识问答、产品推荐和产品对比等功能。

该项目实现了一个 OpenClaw 风格的 Agent 架构，用于模拟企业级 AI 客服系统。

项目简介

本项目实现了一个 B端采购场景的 AI Agent，用户可以通过飞书机器人直接进行咨询。

系统能够：

回答产品相关问题

推荐合适的产品型号

对比不同产品型号

根据知识库生成专业回答

项目核心能力：

AI Agent任务调度

RAG知识库检索

LLM推理生成

飞书机器人交互

系统架构

系统整体架构如下：

用户（飞书）
     │
     ▼
飞书机器人
     │
     ▼
Webhook 接口（FastAPI）
     │
     ▼
AI Agent 层
     │
     ├── 意图识别
     ├── 产品推荐工具
     ├── 产品对比工具
     └── RAG知识库检索
     │
     ▼
LLM推理
     │
     ▼
生成回复 → 返回飞书
功能介绍
飞书机器人交互

用户可以在飞书群中 @机器人提问，机器人会自动回复。

例如：

@机器人 推荐一款适合电工巡检的万用表
AI Agent架构

系统采用 Agent模式设计：

Agent.run() 作为任务执行入口

根据用户问题进行 意图识别

调用不同工具（Tool）

由 LLM生成最终回复

RAG知识库问答

通过 FAISS向量数据库 实现知识库检索。

知识来源包括：

产品说明书

技术文档

产品参数信息

RAG可以减少 大模型幻觉问题。

产品推荐

用户可以通过自然语言请求推荐产品，例如：

@机器人 推荐一款适合电工巡检的万用表

系统会：

1️⃣ 识别用户意图
2️⃣ 筛选候选产品
3️⃣ 结合 LLM 生成推荐理由

产品对比

用户可以对比不同型号产品。

例如：

@机器人 Fluke 17B+ 和 Fluke 18B+ 有什么区别

系统会：

查询产品数据库

提取结构化信息

使用 LLM生成对比结论

技术栈

后端：

FastAPI

Python

AI / 大模型：

OpenAI Compatible LLM

LangChain

FAISS 向量数据库

数据：

产品数据库（CSV）

知识库文档

平台集成：

飞书开放平台

Webhook事件订阅

项目结构
feishu-ai-agent-rag
│
├── app.py                # FastAPI服务 & Agent逻辑
├── config.py             # API配置
├── requirements.txt      # 依赖列表
│
├── data
│   └── products.csv      # 产品数据库
│
├── vector_store          # FAISS向量库
│
└── README.md
安装步骤

克隆项目：

git clone https://github.com/liangtt0916-pixel/feishu-ai-agent-rag.git
cd feishu-ai-agent-rag

创建虚拟环境：

python -m venv venv

激活虚拟环境：

Windows：

venv\Scripts\activate

Mac / Linux：

source venv/bin/activate

安装依赖：

pip install -r requirements.txt
配置

修改 config.py：

OPENAI_API_KEY
OPENAI_BASE_URL
OPENAI_MODEL
VECTOR_DIR
PRODUCTS_FILE

配置飞书环境变量：

FEISHU_APP_ID
FEISHU_APP_SECRET
启动服务

启动 FastAPI：

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

访问接口文档：

http://127.0.0.1:8000/docs
使用 ngrok 暴露服务
ngrok http 8000

将飞书 webhook 配置为：

https://你的ngrok地址/feishu/webhook
示例对话

产品推荐：

@机器人 推荐一款适合电工巡检的万用表

产品对比：

@机器人 Fluke 17B+ 和 Fluke 18B+ 有什么区别

知识问答：

@机器人 Fluke 17B+ 适合什么场景
后续优化

未来可以继续扩展：

多 Agent 协作

Tool注册机制

Agent记忆模块

企业商品数据库接入

采购流程自动化

许可证

MIT License

作者
梁江，2821391038@qq.com
AI Agent Demo 项目，用于展示 Agent + RAG + 飞书机器人集成能力。
