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

## 安装和配置

### 1. 环境要求

- Python 3.8+
- 支持的操作系统: Windows, macOS, Linux

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置API密钥

复制环境变量配置文件：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的API密钥：

**使用OpenAI:**

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
```

**使用阿里云通义千问:**

```env
LLM_PROVIDER=dashscope
DASHSCOPE_API_KEY=your_dashscope_api_key_here
```

## 使用方法

### 基本用法

```bash
python main.py path/to/your/bidding_document.pdf
```

### 指定LLM提供商

```bash
python main.py document.pdf --provider openai
python main.py document.pdf --provider dashscope
```

### 测试连接

```bash
python main.py document.pdf --test-connection
```

### 显示工作流图

```bash
python main.py document.pdf --show-graph
```

### 详细输出模式

```bash
python main.py document.pdf --verbose
```

## 项目结构

```
BiddingAssistant2/
├── config/
│   └── settings.py          # 项目配置
├── src/
│   ├── models/
│   │   └── data_models.py   # 数据模型定义
│   ├── utils/
│   │   ├── document_loader.py    # 文档加载器
│   │   ├── llm_factory.py        # LLM工厂类
│   │   └── vector_store.py       # 向量存储管理
│   ├── agents/
│   │   ├── document_processor.py     # 文档预处理节点
│   │   ├── basic_info_extractor.py   # 基础信息提取节点
│   │   ├── scoring_analyzer.py       # 评分标准分析节点
│   │   ├── other_info_extractor.py   # 其他信息提取节点
│   │   └── output_formatter.py       # 结果格式化节点
│   └── graph/
│       └── bidding_graph.py         # Langgraph工作流定义
├── main.py                  # 主程序入口
├── requirements.txt         # 项目依赖
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

## 故障排除

### 常见问题

**Q: 提示"文档内容为空"**
A: 可能是扫描件PDF，尝试使用OCR工具转换为文本型PDF

**Q: API调用失败**
A: 检查API密钥是否正确，网络连接是否正常

**Q: 内存不足**
A: 减小chunk_size参数或处理较小的文档

### 日志查看

日志文件位于 `logs/bidding_assistant.log`，包含详细的运行信息。

## 开发计划

- [ ] 支持更多文档格式 (DOC, TXT等)
- [ ] 增加OCR功能处理扫描件
- [ ] 添加Web界面
- [ ] 支持批量文档处理
- [ ] 增加人工审核环节 (Human-in-the-loop)
- [ ] 优化提取准确性

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
