<div align="center">

  # Javis Memory

  ### **带 RAG 记忆的个人 AI 助手系统**

  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)](https://fastapi.tiangolo.com)
  [![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

  一个基于 FastAPI 的 AI 助手，通过 **RAG 向量检索记忆系统** 自动管理对话上下文，让 AI "记住"之前的对话并提供个性化回复。

</div>

---

## ✨ 核心特性

| 特性 | 说明 |
| :--- | :--- |
| **🧠 RAG 记忆系统** | 向量检索 + 关键词搜索的混合记忆，智能关联历史对话 |
| **🎯 三层记忆架构** | 工作记忆 / 长期记忆 / 档案记忆，智能分层管理 |
| **📊 重要性评分** | 多维度评估记忆价值：新颖性、情感、反馈、访问、密度 |
| **🔄 智能合并** | 自动检测并合并相似记忆，消除冗余 |
| **🔄 LLM 代理** | 多服务器负载均衡，支持 OpenAI / DeepSeek / 豆包 / 通义千问 / 智谱 / Ollama |
| **📂 自动同步** | 文件监控自动更新记忆索引，实时生效 |
| **⚡ 流式传输** | SSE 实时流式响应，低延迟体验 |
| **🎨 可视化面板** | Web 仪表板展示记忆统计和分布 |
| **🔑 API Key 认证** | 可选的用户认证系统，支持速率限制 |
| **🛡️ 优雅降级** | 数据库异常时自动降级，不影响核心功能 |

---

## 🚀 快速开始

### 环境要求

- **Python** 3.10 或更高版本

### 方式一：自动化配置（推荐）

一键完成依赖安装、配置文件生成和数据库初始化：

```bash
python setup.py
```

交互式向导会引导你：
- 选择 LLM 服务商（OpenAI / DeepSeek / 豆包 / 通义千问 / 智谱 / Ollama / 自定义）
- 配置嵌入服务（用于记忆系统）
- 创建默认用户和 API Key

完成后启动服务：

```bash
python run.py
```

### 方式二：手动配置

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 复制配置文件
cp config/servers.example.yaml config/servers.yaml

# 3. 编辑 config/servers.yaml，填入你的 API Key

# 4. 初始化数据库
python init_db.py

# 5. 启动服务
python run.py
```

### 访问服务

服务启动后：

| 服务 | 地址 |
| :--- | :--- |
| **API 服务** | `http://localhost:8000` |
| **交互式文档** | `http://localhost:8000/docs` |
| **ReDoc 文档** | `http://localhost:8000/redoc` |
| **记忆可视化** | `http://localhost:8000/v1/memory/viz` |

---

## 🏗️ 系统架构

### 整体架构

<div align="center">

  ![System Architecture](docs/architecture.svg)

</div>

### 记忆处理流程

<div align="center">

  ![Memory Flow](docs/memory-flow.svg)

</div>

### 三层记忆架构

<div align="center">

  ![Three-tier Memory](docs/three-tier-memory.svg)

</div>

系统自动将记忆分配到三个层级，根据重要性评分和访问模式动态调整：

| 层级 | 重要性阈值 | 说明 | 存储策略 |
| :--- | :--- | :--- | :--- |
| **工作记忆** (Working) | 0.3 - 0.6 | 最近创建的中等重要性记忆 | 定期清理过期数据（7天） |
| **长期记忆** (Long-term) | ≥ 0.6 | 高重要性记忆，频繁访问 | 完整保留，优先检索 |
| **档案记忆** (Archive) | < 0.3 | 低重要性或过期记忆 | 压缩存储，可选检索 |

### 重要性评分系统

记忆通过多维度评分系统自动评估价值（综合分数 0-1）：

| 维度 | 权重 | 计算方式 |
| :--- | :--- | :--- |
| **新颖性** (Novelty) | 25% | 1 - 与现有记忆的最大相似度 |
| **情感强度** (Sentiment) | 15% | 强情感关键词密度分析 |
| **用户反馈** (Feedback) | 30% | 访问次数 + 时间衰减 |
| **访问频率** (Access) | 20% | 被检索的频率（隐式反馈） |
| **信息密度** (Density) | 10% | 实体数量、代码块、标题密度 |

### 自动生命周期管理

后台任务每天凌晨 2 点自动执行：

| 任务 | 触发条件 | 动作 |
| :--- | :--- | :--- |
| **清理工作记忆** | 每天凌晨 2 点 | 超过 7 天的工作记忆 → 档案 |
| **提升记忆层级** | 每天凌晨 2 点 | 访问 ≥ 5 次的工作记忆 → 长期记忆 |
| **归档旧记忆** | 每天凌晨 2 点 | 超过 90 天且低访问的长期记忆 → 档案 |
| **合并相似记忆** | 每天凌晨 2 点 | 相似度 ≥ 85% 的记忆簇自动合并 |

### 混合检索算法

```python
# 伪代码示例
def hybrid_search(query: str, top_k: int = 6) -> List[Document]:
    # 1. 向量搜索：计算余弦相似度
    query_embedding = embed(query)
    vector_scores = cosine_similarity(query_embedding, doc_embeddings)

    # 2. 关键词搜索：文本匹配
    keyword_scores = text_match(query, doc_texts)

    # 3. 结果融合
    results = []
    for doc in all_candidates:
        score = (
            0.7 * normalize(vector_scores[doc]) +
            0.3 * normalize(keyword_scores[doc])
        )
        if score >= 0.35:  # 最小相关度阈值
            results.append((doc, score))

    return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
```

---

## 📖 API 文档

### 对话 API

#### 聊天完成（启用记忆）

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [
      {"role": "user", "content": "你好，还记得我上次说什么吗？"}
    ],
    "enable_profile": true,
    "stream": false
  }'
```

**响应示例：**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "你好！根据我们的对话记录，你之前提到了..."
    }
  }]
}
```

### 记忆搜索 API

#### 搜索记忆

```bash
curl -X POST "http://localhost:8000/v1/memory/search?user_id=1" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "之前讨论过什么话题",
    "max_results": 5
  }'
```

#### 读取记忆文件

```bash
curl "http://localhost:8000/v1/memory/file?user_id=1&path=user_profile.md"
```

#### 手动同步记忆

```bash
curl -X POST "http://localhost:8000/v1/memory/sync?user_id=1"
```

#### 查看系统状态

```bash
curl "http://localhost:8000/v1/memory/status?user_id=1"
```

### 记忆管理 API

#### 获取记忆统计

```bash
curl "http://localhost:8000/v1/memory/stats?user_id=1"
```

**响应示例：**
```json
{
  "user_id": 1,
  "tiers": {
    "working": 5,
    "long_term": 12,
    "archive": 8
  },
  "total_chunks": 25,
  "avg_importance": 0.52,
  "total_accesses": 47
}
```

#### 手动触发记忆合并

```bash
curl -X POST "http://localhost:8000/v1/memory/consolidate?user_id=1"
```

**响应示例：**
```json
{
  "clusters_found": 3,
  "clusters_consolidated": 3,
  "memories_consolidated": 12
}
```

#### 清理过期工作记忆

```bash
curl -X POST "http://localhost:8000/v1/memory/cleanup?user_id=1"
```

**响应示例：**
```json
{
  "cleaned": 5
}
```

#### 提升记忆层级

```bash
curl -X POST "http://localhost:8000/v1/memory/promote?user_id=1"
```

**响应示例：**
```json
{
  "promoted": 2
}
```

#### 获取用户列表

```bash
curl "http://localhost:8000/v1/memory/users"
```

**响应示例：**
```json
{
  "users": [
    {"id": 1, "name": "default", "email": null}
  ]
}
```

### Web 可视化仪表板

访问 `http://localhost:8000/v1/memory/viz` 查看记忆系统可视化面板：

- 层级分布饼图
- 重要性分布柱状图
- Top 重要记忆列表
- 最近访问记录

> 用户列表从数据库动态获取，自动显示所有实际存在的用户。

---

## ⚙️ 配置说明

主要配置位于 `config/servers.yaml`：

### LLM 服务器配置

```yaml
servers:
  deepseek:
    base_url: https://api.deepseek.com/v1
    api_key: sk-your-api-key
    models:
      - deepseek-chat
      - deepseek-reasoner
```

### 记忆系统基础配置

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `memory_search.enabled` | `true` | 是否启用记忆系统 |
| `memory_search.provider` | `auto` | 嵌入提供者：`openai` / `gemini` / `local` / `auto` |
| `memory_search.chunking.tokens` | `400` | 分块 token 数量 |
| `memory_search.chunking.overlap` | `80` | 分块重叠 token 数 |
| `memory_search.query.max_results` | `6` | 搜索最大返回结果数 |
| `memory_search.query.min_score` | `0.35` | 最小相关度分数阈值 |
| `memory_search.query.hybrid.vector_weight` | `0.7` | 向量搜索权重 |
| `memory_search.query.hybrid.text_weight` | `0.3` | 关键词搜索权重 |
| `memory_search.sync.watch` | `true` | 是否启用文件监控自动同步 |
| `memory_search.sync.watch_debounce_ms` | `1500` | 文件变化防抖延迟（毫秒） |

### 生命周期配置

```yaml
memory_search:
  lifecycle:
    enabled: true                      # 是否启用生命周期管理
    working_memory_days: 7             # 工作记忆保留天数
    long_term_threshold: 0.6           # 长期记忆重要性阈值
    archive_threshold: 0.3             # 档案记忆重要性阈值
    archive_after_days: 90             # 归档旧长期记忆天数
    promotion_access_count: 5          # 提升为长期记忆的访问次数

    importance:
      enabled: true
      weights:
        novelty: 0.25                  # 新颖性权重
        sentiment: 0.15                # 情感强度权重
        feedback: 0.30                 # 用户反馈权重
        access: 0.20                   # 访问频率权重
        density: 0.10                  # 信息密度权重
```

### 合并配置

```yaml
memory_search:
  consolidation:
    enabled: true                      # 是否启用记忆合并
    similarity_threshold: 0.85         # 相似度阈值（0-1）
    min_cluster_size: 3                # 最小簇大小
    time_window_days: 7                # 时间窗口（天）
    schedule: "0 2 * * *"              # Cron 表达式
```

---

## 📁 项目结构

```
javis/
├── api/                        # API 层
│   ├── routes/
│   │   ├── chat.py            # 聊天完成接口（集成 RAG）
│   │   ├── memory.py          # 记忆搜索/统计/合并接口
│   │   └── api_keys.py        # API Key 管理
│   ├── main.py                # FastAPI 应用入口
│   └── middleware.py          # 日志中间件
│
├── services/                   # 业务服务层
│   ├── llm_forwarder.py       # LLM 请求转发（负载均衡）
│   ├── memory_index.py        # 记忆索引管理器
│   ├── search_engine.py       # 混合搜索引擎
│   ├── memory_importance.py   # 重要性评分服务
│   ├── memory_lifecycle.py    # 三层记忆生命周期管理
│   ├── memory_consolidation.py # 记忆合并去重服务
│   ├── memory_visualization.py # Web 可视化数据服务
│   ├── entity_extractor.py    # 实体提取服务
│   ├── scheduler.py           # 后台任务调度器
│   ├── chunker.py             # 文本分块服务
│   ├── file_watcher.py        # 文件监控服务
│   ├── conversation.py        # 对话管理
│   ├── auth.py                # 认证服务
│   └── embedding/             # 嵌入服务
│       ├── factory.py         # 提供者工厂
│       ├── openai_client.py   # OpenAI 兼容 API
│       ├── gemini_client.py   # Gemini API
│       └── local_client.py    # 本地 sentence-transformers
│
├── database/                   # 数据持久层
│   ├── models.py              # 用户/对话/消息模型
│   ├── memory_models.py       # RAG 记忆模型 + 元数据表
│   ├── session.py             # 异步会话管理
│   └── repository.py          # 数据访问层
│
├── config/
│   ├── settings.py            # Pydantic 配置模型
│   └── servers.yaml           # 主配置文件
│
├── templates/                  # Web 模板
│   └── memory_viz.html        # 记忆可视化仪表板
│
├── schemas/                    # API 数据模型
│   ├── chat.py                # 聊天相关模型
│   └── memory_search.py       # 记忆搜索模型
│
├── run.py                      # 启动脚本
├── setup.py                    # 自动化配置向导
└── init_db.py                  # 数据库初始化
```

---

## 🔧 技术栈

| 组件 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **Web 框架** | FastAPI + uvicorn | 现代异步 Python Web 框架 |
| **数据库 ORM** | SQLAlchemy 2.0 + aiosqlite | 异步 ORM，SQLite 存储 |
| **向量搜索** | sqlite-vec | 归一化向量 + 余弦相似度 |
| **全文搜索** | SQLite LIKE / FTS5 | 可选 FTS5 全文索引 |
| **HTTP 客户端** | httpx | 支持 HTTP/2、连接池、重试 |
| **文件监控** | watchdog | 跨平台文件系统事件 |
| **配置管理** | YAML + Pydantic | 类型安全的配置解析 |
| **嵌入模型** | OpenAI / Gemini / Local | 支持多种嵌入服务 |
| **任务调度** | asyncio | 后台定时任务执行 |
| **可视化** | Chart.js + CSS Grid | 响应式 Web 仪表板 |

---

## 📝 许可证

MIT License

---

<div align="center">

  **Made with ❤️ by [Your Name]**

  [⭐ Star this repo](../../stargazers) · [🐛 Report a bug](../../issues) · [📖 Request a feature](../../issues)

</div>
