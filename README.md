# 智能投标助手 (Intelligent Bidding Assistant)

一个基于Langchain和Langgraph的AI智能体，用于自动分析招投标文件并生成结构化的关注项摘要报告。

## 功能特性

- 📄 **多格式支持**: 支持PDF、DOCX等常见招标文件格式
- 🤖 **多模型支持**: 支持OpenAI和阿里云通义千问系列模型
- 🔍 **智能提取**: 基于RAG技术的精准信息提取
- 📊 **结构化输出**: 生成Markdown格式的详细分析报告
- 🔄 **多智能体协作**: 使用Langgraph构建的多节点工作流
- 📍 **来源追溯**: 所有提取信息都标注原文来源

## 提取信息模块

### 1. 基础信息模块

- 项目名称、招标编号、预算金额
- 投标截止时间、开标时间
- 投标保证金信息
- 采购人和代理机构信息
- 资格审查硬性条件

### 2. 评分标准分析模块

- 初步评审标准
- 详细评审方法
- 分值构成（技术分、商务分、价格分）
- 详细评分细则表
- 加分项和否决项条款

### 3. 其他重要信息模块

- 合同主要条款
- 付款方式与周期
- 交付要求和期限
- 知识产权归属
- 潜在风险点识别

## 🚀 快速开始

### 方式一：传统 Python 环境

#### 1. 环境要求
- Python 3.12+ (推荐)
- 支持的操作系统: Windows, macOS, Linux

#### 2. 安装依赖
```bash
pip install -r requirements.txt
```

#### 3. 配置 API 密钥
复制环境变量配置文件：
```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的API密钥：

**使用阿里云通义千问（推荐）:**
```env
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

**使用 OpenAI:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 方式二：Docker 部署（推荐新手）

#### 🐳 独立 Docker 部署

**1. 构建 Docker 镜像**
```bash
# 在 backend 目录下执行
docker build -t bidbot-backend .
```

**2. 运行容器**
```bash
# 基础运行
docker run -p 8000:8000 \
  -e DASHSCOPE_API_KEY=your_api_key \
  bidbot-backend

# 带数据持久化运行
docker run -p 8000:8000 \
  -e DASHSCOPE_API_KEY=your_api_key \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/vector_store:/app/vector_store \
  bidbot-backend
```

**3. 验证服务**
```bash
# 健康检查
curl http://localhost:8000/health

# 查看 API 文档
open http://localhost:8000/docs
```

#### 🐳 与前端一起部署

**1. 返回项目根目录**
```bash
cd ..  # 回到 BidBot3 根目录
```

**2. 配置环境变量**
```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件，填入 API 密钥
nano .env
```

**3. 一键部署（开发环境）**
```bash
# 使用部署脚本
./deploy.sh

# 或手动部署
docker-compose up -d --build
```

**4. 一键部署（生产环境）**
```bash
# 需要 root 权限
sudo ./deploy-prod.sh
```

## 🐳 Docker 配置详解

### Dockerfile 说明
```dockerfile
# 基于 Python 3.12 官方镜像
FROM python:3.12-slim

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 安装系统依赖（包含 curl 用于健康检查）
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建必要目录
RUN mkdir -p uploads temp logs output vector_store

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动应用
CMD ["python", "start_api.py"]
```

### 环境变量配置

| 变量名 | 说明 | 默认值 | 必填 |
|--------|------|--------|------|
| `LLM_PROVIDER` | LLM 提供商 | `dashscope` | ✅ |
| `DASHSCOPE_API_KEY` | 阿里云 API 密钥 | - | ✅* |
| `OPENAI_API_KEY` | OpenAI API 密钥 | - | ✅* |
| `OPENAI_BASE_URL` | OpenAI API 基础 URL | - | ❌ |
| `LOG_LEVEL` | 日志级别 | `INFO` | ❌ |
| `VECTOR_STORE_PATH` | 向量存储路径 | `/app/vector_store` | ❌ |
| `OUTPUT_DIR` | 输出目录 | `/app/output` | ❌ |
| `CHUNK_SIZE` | 文本分块大小 | `1000` | ❌ |
| `CLEAR_VECTOR_STORE_ON_NEW_DOCUMENT` | 向量库隔离 | `true` | ❌ |

*根据选择的 LLM 提供商填写对应的 API 密钥

### 数据持久化

Docker 部署时，以下目录需要持久化：

```bash
# 开发环境（使用 Docker 卷）
volumes:
  - backend_uploads:/app/uploads          # 上传文件
  - backend_output:/app/output            # 分析报告
  - backend_vector_store:/app/vector_store # 向量数据
  - backend_logs:/app/logs                # 日志文件
  - backend_temp:/app/temp                # 临时文件

# 生产环境（挂载到系统目录）
volumes:
  - /var/lib/bidbot/uploads:/app/uploads
  - /var/lib/bidbot/output:/app/output
  - /var/lib/bidbot/vector_store:/app/vector_store
  - /var/log/bidbot:/app/logs
  - /tmp/bidbot:/app/temp
```

## 📖 使用方法

### 命令行使用（传统方式）

#### 基本用法
```bash
python main.py path/to/your/bidding_document.pdf
```

#### 指定 LLM 提供商
```bash
python main.py document.pdf --provider openai
python main.py document.pdf --provider dashscope
```

#### 测试连接
```bash
python main.py document.pdf --test-connection
```

#### 显示工作流图
```bash
python main.py document.pdf --show-graph
```

#### 详细输出模式
```bash
python main.py document.pdf --verbose
```

### API 服务使用（推荐）

#### 启动 API 服务
```bash
# 传统方式
python start_api.py

# Docker 方式
docker run -p 8000:8000 bidbot-backend
```

#### API 接口说明

**1. 健康检查**
```bash
GET /health
curl http://localhost:8000/health
```

**2. 文件上传**
```bash
POST /api/upload
curl -X POST -F "file=@document.pdf" http://localhost:8000/api/upload
```

**3. 启动分析**
```bash
POST /api/analyze
curl -X POST -H "Content-Type: application/json" \
  -d '{"file_id": "your-file-id"}' \
  http://localhost:8000/api/analyze
```

**4. 查询分析状态**
```bash
GET /api/analysis/{task_id}
curl http://localhost:8000/api/analysis/your-task-id
```

**5. 下载分析报告**
```bash
GET /api/download-report/{task_id}
curl -O http://localhost:8000/api/download-report/your-task-id?format=md
```

**6. API 文档**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📁 项目结构

```
backend/
├── api/                     # FastAPI 应用
│   ├── main.py                 # FastAPI 主应用
│   ├── middleware/             # 中间件
│   │   └── session.py             # 会话管理中间件
│   ├── models/                 # API 数据模型
│   │   └── api_models.py          # Pydantic 模型定义
│   ├── routers/                # API 路由
│   │   ├── upload.py              # 文件上传路由
│   │   ├── analysis.py            # 分析任务路由
│   │   └── files.py               # 文件管理路由
│   ├── services/               # 业务服务
│   │   ├── file_service.py        # 文件服务
│   │   └── task_service.py        # 任务服务
│   └── tasks/                  # 后台任务
│       ├── cleanup.py             # 清理任务
│       └── simple_cleanup.py      # 简化清理任务
├── config/                  # 配置模块
│   ├── settings.py             # 项目配置
│   └── logging_config.py       # 日志配置
├── src/                     # 核心业务逻辑
│   ├── models/                 # 数据模型
│   │   └── data_models.py         # 业务数据模型
│   ├── utils/                  # 工具类
│   │   ├── document_loader.py     # 文档加载器
│   │   ├── llm_factory.py         # LLM 工厂类
│   │   └── vector_store.py        # 向量存储管理
│   ├── agents/                 # AI 智能体
│   │   ├── document_processor.py     # 文档预处理节点
│   │   ├── basic_info_extractor.py   # 基础信息提取节点
│   │   ├── scoring_analyzer.py       # 评分标准分析节点
│   │   ├── other_info_extractor.py   # 其他信息提取节点
│   │   └── output_formatter.py       # 结果格式化节点
│   └── graph/                  # LangGraph 工作流
│       ├── bidding_graph.py          # 主工作流定义
│       └── parallel_aggregator.py    # 并行聚合器
├── uploads/                 # 上传文件目录
├── output/                  # 分析报告输出目录
├── vector_store/            # 向量数据库存储
├── temp/                    # 临时文件目录
├── logs/                    # 日志文件目录
├── main.py                  # 命令行入口
├── start_api.py             # API 服务启动脚本
├── requirements.txt         # Python 依赖
├── Dockerfile              # Docker 镜像构建文件
├── .dockerignore           # Docker 构建忽略文件
├── .env.example            # 环境变量配置模板
└── README.md               # 项目说明
```

## 工作流程

1. **文档预处理**: 加载文档、文本分割、向量化存储
2. **基础信息提取**: 提取项目基本信息和资格审查条件
3. **评分标准分析**: 分析评分方法、分值构成和评分细则
4. **其他信息提取**: 提取合同条款、风险点等重要信息
5. **结果格式化**: 生成Markdown格式的分析报告

## 输出示例

分析完成后，会在 `output/` 目录下生成详细的Markdown报告，包含：

- 📋 基础信息表格
- 📊 评分标准分析
- ⚠️ 否决项条款提醒
- 🚨 潜在风险点识别
- 📍 所有信息的原文来源

## 技术架构

- **核心框架**: Langchain + Langgraph
- **文档处理**: PyPDF2, python-docx
- **向量数据库**: ChromaDB
- **LLM支持**: OpenAI GPT-4o-mini, 阿里云通义千问
- **嵌入模型**: OpenAI text-embedding-3-large, 通义 text-embedding-v3

## 注意事项

1. 确保API密钥有足够的配额
2. 大型文档可能需要较长的处理时间
3. 建议使用高质量的文本型PDF文件
4. 扫描件PDF可能影响提取质量

## 向量库隔离机制

为了确保每次处理新文档时不受历史文档数据的影响，系统提供了向量库隔离功能：

### 🔒 隔离模式（推荐）

- **默认开启**：`CLEAR_VECTOR_STORE_ON_NEW_DOCUMENT=true`
- **工作原理**：每次处理新文档时自动清空历史向量数据
- **优势**：确保信息提取的准确性，避免历史文档的交叉污染
- **适用场景**：大多数使用场景，特别是需要准确分析单个文档时

### 📚 累积模式

- **配置方式**：`CLEAR_VECTOR_STORE_ON_NEW_DOCUMENT=false`
- **工作原理**：保留历史文档的向量数据，可能存在交叉检索
- **风险**：新文档的信息提取可能受到历史文档内容的影响
- **适用场景**：需要跨文档检索或对比分析的特殊场景

### 🧪 测试隔离效果

```bash
# 运行隔离效果测试
python test_vector_isolation.py
```

## 🚨 故障排除

### Docker 相关问题

**Q: Docker 容器启动失败**
```bash
# 查看容器日志
docker logs bidbot-backend

# 检查容器状态
docker ps -a

# 重新构建镜像
docker build --no-cache -t bidbot-backend .
```

**Q: 健康检查失败**
```bash
# 手动测试健康检查
curl http://localhost:8000/health

# 检查容器内部
docker exec -it bidbot-backend bash
curl localhost:8000/health
```

**Q: 数据持久化问题**
```bash
# 检查数据卷
docker volume ls
docker volume inspect bidbot3_backend_uploads

# 检查挂载点权限
docker exec -it bidbot-backend ls -la /app/uploads
```

### API 相关问题

**Q: API 调用失败**
```bash
# 检查 API 服务状态
curl http://localhost:8000/health

# 查看 API 文档
open http://localhost:8000/docs

# 检查请求格式
curl -X POST -H "Content-Type: application/json" \
  -d '{"test": "data"}' http://localhost:8000/api/test
```

**Q: 文件上传失败**
- 检查文件格式（支持 PDF、DOC、DOCX）
- 检查文件大小（默认限制 100MB）
- 检查磁盘空间是否充足

**Q: 分析任务卡住**
```bash
# 查看任务状态
curl http://localhost:8000/api/analysis/your-task-id

# 重启服务
docker restart bidbot-backend

# 清理向量数据库
docker exec -it bidbot-backend rm -rf /app/vector_store/*
```

### 传统部署问题

**Q: 提示"文档内容为空"**
A: 可能是扫描件PDF，尝试使用OCR工具转换为文本型PDF

**Q: LLM API 调用失败**
A: 检查API密钥是否正确，网络连接是否正常

**Q: 内存不足**
A: 减小chunk_size参数或处理较小的文档

**Q: 依赖安装失败**
```bash
# 升级 pip
pip install --upgrade pip

# 清理缓存重新安装
pip cache purge
pip install -r requirements.txt

# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 日志查看

**Docker 环境**
```bash
# 查看容器日志
docker logs -f bidbot-backend

# 查看特定时间段日志
docker logs --since="2024-01-01T00:00:00" bidbot-backend

# 进入容器查看日志文件
docker exec -it bidbot-backend tail -f /app/logs/bidding_assistant.log
```

**传统环境**
```bash
# 日志文件位置
tail -f logs/bidding_assistant.log

# 按级别查看日志
grep "ERROR" logs/bidding_assistant.log
grep "WARNING" logs/bidding_assistant.log
```

### 性能优化

**内存优化**
```bash
# 限制 Docker 容器内存
docker run -m 2g bidbot-backend

# 调整文本分块大小
export CHUNK_SIZE=500
```

**并发优化**
```bash
# 调整工作进程数
export MAX_WORKERS=2

# 设置超时时间
export REQUEST_TIMEOUT=300
```

## 🔧 开发指南

### 本地开发环境搭建

**1. 克隆项目**
```bash
git clone <repository-url>
cd BidBot3/backend
```

**2. 创建虚拟环境**
```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 使用 conda
conda create -n bidbot python=3.12
conda activate bidbot
```

**3. 安装开发依赖**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 如果有开发依赖
```

**4. 配置开发环境**
```bash
cp .env.example .env
# 编辑 .env 文件，填入开发用的 API 密钥
```

**5. 启动开发服务器**
```bash
# API 服务
python start_api.py

# 或命令行模式
python main.py test_documents/sample.pdf --verbose
```

### Docker 开发环境

**1. 构建开发镜像**
```bash
# 构建镜像
docker build -t bidbot-backend-dev .

# 或使用 docker-compose
docker-compose -f docker-compose.dev.yml up --build
```

**2. 开发时挂载代码**
```bash
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd):/app \
  -v $(pwd)/uploads:/app/uploads \
  -e DASHSCOPE_API_KEY=your_key \
  bidbot-backend-dev
```

### 代码规范

**目录结构规范**
- `api/`: FastAPI 相关代码
- `src/`: 核心业务逻辑
- `config/`: 配置文件
- `tests/`: 测试代码（待添加）

**代码风格**
- 使用 Black 进行代码格式化
- 使用 isort 进行导入排序
- 使用 mypy 进行类型检查
- 遵循 PEP 8 编码规范

**提交规范**
- 使用语义化提交信息
- 每个提交应该是一个完整的功能点
- 提交前运行测试和代码检查

## 🧪 测试

### 单元测试
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_document_loader.py

# 生成覆盖率报告
pytest --cov=src tests/
```

### API 测试
```bash
# 使用 curl 测试
curl -X POST -F "file=@test.pdf" http://localhost:8000/api/upload

# 使用 httpie 测试
http POST localhost:8000/api/upload file@test.pdf

# 使用 Postman 或 Insomnia 测试
# 导入 API 文档: http://localhost:8000/docs
```

### 集成测试
```bash
# 端到端测试
python tests/integration/test_full_workflow.py

# Docker 环境测试
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 📊 监控和维护

### 健康检查
```bash
# 基础健康检查
curl http://localhost:8000/health

# 详细健康检查
curl http://localhost:8000/health?detailed=true
```

### 日志监控
```bash
# 实时日志
tail -f logs/bidding_assistant.log

# 错误日志
grep "ERROR" logs/bidding_assistant.log | tail -20

# 性能日志
grep "PERFORMANCE" logs/bidding_assistant.log
```

### 性能监控
```bash
# 内存使用
docker stats bidbot-backend

# 磁盘使用
du -sh uploads/ output/ vector_store/

# API 响应时间
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
```

### 数据备份
```bash
# 备份上传文件
tar -czf backup-uploads-$(date +%Y%m%d).tar.gz uploads/

# 备份分析报告
tar -czf backup-output-$(date +%Y%m%d).tar.gz output/

# 备份向量数据库
tar -czf backup-vector-$(date +%Y%m%d).tar.gz vector_store/
```

## 🚀 部署指南

### 开发环境部署
```bash
# 使用项目根目录的部署脚本
cd ..
./deploy.sh
```

### 生产环境部署
```bash
# 使用生产环境部署脚本
cd ..
sudo ./deploy-prod.sh
```

### 手动部署
```bash
# 构建镜像
docker build -t bidbot-backend:latest .

# 运行容器
docker run -d \
  --name bidbot-backend \
  -p 8000:8000 \
  -e DASHSCOPE_API_KEY=your_key \
  -v /var/lib/bidbot:/app/data \
  --restart unless-stopped \
  bidbot-backend:latest
```

## 🔮 开发计划

### 已完成功能
- ✅ 多格式文档支持 (PDF, DOC, DOCX)
- ✅ 多 LLM 提供商支持 (OpenAI, 阿里云通义千问)
- ✅ RESTful API 接口
- ✅ Docker 容器化部署
- ✅ 会话管理和数据隔离
- ✅ 并行智能体处理
- ✅ 向量数据库隔离机制
- ✅ 健康检查和监控

### 开发中功能
- 🔄 单元测试覆盖
- 🔄 性能优化
- 🔄 错误处理增强

### 计划功能
- [ ] 支持更多文档格式 (TXT, RTF等)
- [ ] 增加 OCR 功能处理扫描件
- [ ] 支持批量文档处理
- [ ] 增加人工审核环节 (Human-in-the-loop)
- [ ] 优化提取准确性
- [ ] 添加缓存机制
- [ ] 支持分布式部署
- [ ] 增加用户认证和权限管理

## 📚 相关文档

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [Docker 部署指南](../DOCKER_DEPLOYMENT.md)
- [前端项目文档](../frontend/README.md)

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。
