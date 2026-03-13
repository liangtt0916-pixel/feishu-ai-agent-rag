# Tool CS Demo

一个基于向量搜索的工具客服演示项目，使用产品手册数据为用户提供智能问答服务。

## 项目简介

本项目演示了如何使用 LangChain 和 FAISS 构建一个智能客服系统。通过对工具产品手册进行向量化处理，用户可以提出问题并获得基于手册内容的准确回答。

## 功能特性

- 📚 产品手册向量化存储
- 🔍 基于向量相似度的智能搜索
- 💬 聊天界面支持
- 🚀 快速构建和部署

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/your-username/tool-cs-demo.git
cd tool-cs-demo
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 构建向量索引

```bash
python build_index.py
```

## 使用方法

### 启动应用

```bash
python app.py
```

### 访问聊天界面

打开浏览器访问 `http://localhost:8000` 或直接打开 `chat.html` 文件。

### 示例问题

- "推荐一款适合电工巡检的万用表"
- "如何使用Fluke 17B+万用表测量电压"
- "Bosch GSB600电钻有哪些安全注意事项"

## 项目结构

```
tool-cs-demo/
├── app.py              # 主应用文件
├── build_index.py      # 索引构建脚本
├── chat.html          # 聊天界面
├── config.py          # 配置文件
├── requirements.txt   # Python依赖
├── data/              # 产品手册数据
│   ├── products.csv
│   └── *.txt          # 各产品手册
└── vector_store/      # 向量存储文件
    └── index.faiss
```

## 依赖库

主要依赖包括：
- LangChain: 语言链框架
- FAISS: 向量搜索库
- FastAPI: Web框架
- OpenAI: AI模型接口
- 其他工具库

## 配置说明

编辑 `config.py` 文件来配置：
- OpenAI API密钥
- 向量存储参数
- 模型选择

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 作者

[Liang Jiang] - [2821391038@qq.com]