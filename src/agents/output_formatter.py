"""
结果格式化输出节点
Output formatting node for the Langgraph workflow
"""

from typing import Dict, Any, List, Optional
from loguru import logger
from src.models.data_models import GraphState, GraphStateModel, ExtractedField, ScoringItem, BidDocumentRequirements, BidEvaluationProcess
from config.settings import settings
import os
from datetime import datetime

class OutputFormatter:
    """结果格式化器"""
    
    def __init__(self):
        """初始化格式化器"""
        pass
    
    def format_output(self, state: GraphState) -> GraphState:
        """
        格式化输出结果

        Args:
            state: 图状态

        Returns:
            GraphState: 更新后的状态
        """
        try:
            logger.info("开始格式化输出结果")

            # 检查状态对象类型
            if isinstance(state, dict):
                logger.warning("状态对象是字典类型，尝试转换为GraphStateModel")
                state = GraphStateModel(**state)

            # 生成Markdown报告
            markdown_content = self._generate_markdown_report(state)

            # 保存到文件
            output_file = self._save_report(markdown_content, state.analysis_result.document_name)

            # 更新状态
            state.current_step = "completed"
            state.analysis_result.processing_notes.append(f"报告已保存到: {output_file}")

            logger.info(f"输出格式化完成，报告保存到: {output_file}")

        except Exception as e:
            error_msg = f"输出格式化失败: {str(e)}"
            logger.error(error_msg)
            if hasattr(state, 'error_messages'):
                state.error_messages.append(error_msg)
            state.current_step = "error"

        return state
    
    def _generate_markdown_report(self, state: GraphState) -> str:
        """生成Markdown格式的报告"""
        result = state.analysis_result
        
        # 报告标题
        markdown = f"""# 智能投标助手分析报告

## 文档信息
- **文档名称**: {result.document_name}
- **分析时间**: {result.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、基础信息模块

### 基本项目信息

| 项目 | 内容 | 来源 |
|------|------|------|
"""
        
        # 基础信息表格
        basic_info = result.basic_information
        basic_fields = [
            ("项目名称", basic_info.project_name),
            ("招标编号", basic_info.tender_number),
            ("采购预算金额", basic_info.budget_amount),
            ("投标截止时间", basic_info.bid_deadline),
            ("开标时间", basic_info.bid_opening_time),
            ("投标保证金金额", basic_info.bid_bond_amount),
            ("投标保证金缴纳账户信息", basic_info.bid_bond_account),
            ("采购人名称", basic_info.purchaser_name),
            ("采购人联系方式", basic_info.purchaser_contact),
            ("采购代理机构名称", basic_info.agent_name),
            ("采购代理机构联系人及联系方式", basic_info.agent_contact),
        ]
        
        for field_name, field_value in basic_fields:
            value = self._format_extracted_field(field_value)
            source = self._get_source_info_for_table(field_value.source) if field_value.source else "来源未知"
            # 清理表格内容，避免换行符导致的格式问题
            value_clean = self._clean_table_content(value)
            source_clean = self._clean_table_content(source)
            markdown += f"| {field_name} | {value_clean} | {source_clean} |\n"
        
        # 投标人资格要求
        markdown += "\n### 投标人资格要求\n\n"

        qualification = basic_info.qualification_criteria

        # 收集所有资格要求项目
        all_qualification_items = []
        if qualification.company_certifications:
            all_qualification_items.extend(qualification.company_certifications)
        if qualification.project_experience:
            all_qualification_items.extend(qualification.project_experience)
        if qualification.team_requirements:
            all_qualification_items.extend(qualification.team_requirements)
        if qualification.other_requirements:
            all_qualification_items.extend(qualification.other_requirements)

        # 动态生成资格要求内容，不使用预定义分类
        if all_qualification_items:
            formatted_content = self._format_qualification_requirements(all_qualification_items)
            markdown += formatted_content

        # 投标文件要求
        markdown += "### 投标文件要求\n\n"

        bid_doc_requirements = basic_info.bid_document_requirements

        if bid_doc_requirements.composition_and_format:
            markdown += "#### 组成与编制规范\n"
            for i, item in enumerate(bid_doc_requirements.composition_and_format, 1):
                value = self._format_extracted_field(item)
                source = self._get_source_info(item)
                # 对于组成与编制规范，保持原文结构
                formatted_value = self._format_composition_content(value)
                markdown += f"{formatted_value}\n\n{source}\n\n"
            markdown += "\n"

        if bid_doc_requirements.binding_and_sealing:
            markdown += "#### 装订与密封要求\n"
            for i, item in enumerate(bid_doc_requirements.binding_and_sealing, 1):
                value = self._format_extracted_field(item)
                source = self._get_source_info(item)
                markdown += f"{value}\n\n{source}\n\n"
            markdown += "\n"

        if bid_doc_requirements.signature_and_seal:
            markdown += "#### 签字盖章要求\n"
            for i, item in enumerate(bid_doc_requirements.signature_and_seal, 1):
                value = self._format_extracted_field(item)
                source = self._get_source_info(item)
                markdown += f"{value}\n\n{source}\n\n"
            markdown += "\n"

        if bid_doc_requirements.document_structure:
            markdown += "#### 投标文件章节框架（目录）\n"
            # 检查是否有多个目录结构
            if len(bid_doc_requirements.document_structure) > 1:
                # 多个目录结构，分别显示
                for i, item in enumerate(bid_doc_requirements.document_structure, 1):
                    value = self._format_extracted_field(item)
                    source = self._get_source_info(item)
                    # 尝试识别目录类型
                    directory_type = self._identify_directory_type(value)
                    markdown += f"##### {directory_type}\n"
                    formatted_structure = self._format_document_structure(value)
                    markdown += f"{formatted_structure}\n\n{source}\n\n"
            else:
                # 单个目录结构
                for i, item in enumerate(bid_doc_requirements.document_structure, 1):
                    value = self._format_extracted_field(item)
                    source = self._get_source_info(item)
                    formatted_structure = self._format_document_structure(value)
                    markdown += f"{formatted_structure}\n\n{source}\n\n"
            markdown += "\n"

        # 开评定标流程
        markdown += "### 开评定标流程\n\n"

        bid_evaluation_process = basic_info.bid_evaluation_process

        if bid_evaluation_process.bid_opening:
            markdown += "#### 开标环节（时间、地点、程序）\n"
            for i, item in enumerate(bid_evaluation_process.bid_opening, 1):
                value = self._format_extracted_field(item)
                source = self._get_source_info(item)
                markdown += f"{value}\n\n{source}\n\n"
            markdown += "\n"

        if bid_evaluation_process.evaluation:
            markdown += "#### 评标环节（评委会、评审方法/标准、主要流程）\n"
            for i, item in enumerate(bid_evaluation_process.evaluation, 1):
                value = self._format_extracted_field(item)
                source = self._get_source_info(item)
                markdown += f"{value}\n\n{source}\n\n"
            markdown += "\n"

        if bid_evaluation_process.award_decision:
            markdown += "#### 定标环节（定标原则、中标通知）\n"
            for i, item in enumerate(bid_evaluation_process.award_decision, 1):
                value = self._format_extracted_field(item)
                source = self._get_source_info(item)
                markdown += f"{value}\n\n{source}\n\n"
            markdown += "\n"

        # 评分标准分析模块
        markdown += "---\n\n## 二、评分标准分析模块\n\n"
        
        scoring = result.scoring_criteria
        
        # 初步评审标准
        if scoring.preliminary_review:
            markdown += "### 初步评审标准\n"
            for i, review in enumerate(scoring.preliminary_review, 1):
                value = self._format_extracted_field(review)
                source = self._get_source_info(review)
                markdown += f"{value}\n\n{source}\n\n"
            markdown += "\n"

        # 详细评审方法
        if scoring.evaluation_method.value:
            markdown += "### 详细评审方法\n"
            value = self._format_extracted_field(scoring.evaluation_method)
            source = self._get_source_info(scoring.evaluation_method)
            markdown += f"{value}\n\n{source}\n\n"
        
        # 分值构成
        markdown += "### 分值构成\n\n"
        markdown += "| 评分类别 | 占比/分值 | 来源 |\n"
        markdown += "|----------|-----------|------|\n"
        
        score_comp = scoring.score_composition
        comp_fields = [
            ("技术分", score_comp.technical_score),
            ("商务分", score_comp.commercial_score),
            ("价格分", score_comp.price_score),
        ]
        
        for field_name, field_value in comp_fields:
            if field_value.value:
                value = self._format_extracted_field(field_value)
                source = self._get_source_info_for_table(field_value.source) if field_value.source else "来源未知"
                value_clean = self._clean_table_content(value)
                source_clean = self._clean_table_content(source)
                markdown += f"| {field_name} | {value_clean} | {source_clean} |\n"

        for other_score in score_comp.other_scores:
            if other_score.value:
                value = self._format_extracted_field(other_score)
                source = self._get_source_info_for_table(other_score.source) if other_score.source else "来源未知"
                value_clean = self._clean_table_content(value)
                source_clean = self._clean_table_content(source)
                markdown += f"| 其他 | {value_clean} | {source_clean} |\n"
        
        markdown += "\n"
        
        # 详细评分细则表
        if scoring.detailed_scoring:
            markdown += "### 详细评分细则表\n\n"
            markdown += "| 评分类别 | 评分项 | 最高分值 | 评分标准 | 来源 |\n"
            markdown += "|----------|--------|----------|----------|------|\n"

            for item in scoring.detailed_scoring:
                category = self._clean_table_content(item.category or "未分类")
                item_name = self._clean_table_content(item.item_name or "未命名")
                max_score = self._clean_table_content(str(item.max_score) if item.max_score is not None else "未指定")
                criteria = self._clean_table_content(item.criteria or "未指定")
                # 添加来源信息
                source = self._get_source_info_for_table(item.source) if item.source else "来源未知"
                source_clean = self._clean_table_content(source)
                markdown += f"| {category} | {item_name} | {max_score} | {criteria} | {source_clean} |\n"

            markdown += "\n"
        
        # 加分项明细
        if scoring.bonus_points:
            markdown += "### 加分项明细\n"
            for i, bonus in enumerate(scoring.bonus_points, 1):
                value = self._format_extracted_field(bonus)
                source = self._get_source_info(bonus)
                markdown += f"{value}\n\n{source}\n\n"
            markdown += "\n"

        # 否决项条款
        if scoring.disqualification_clauses:
            markdown += "### ⚠️ 否决项条款（重要）\n"
            for i, clause in enumerate(scoring.disqualification_clauses, 1):
                value = self._format_extracted_field(clause)
                source = self._get_source_info(clause)
                markdown += f"**{value}**\n\n{source}\n\n"
            markdown += "\n"
        
        # 合同信息模块
        markdown += "---\n\n## 三、合同信息模块\n\n"

        contract_info = result.contract_information

        # 违约责任
        if contract_info.breach_liability:
            markdown += "### 违约责任\n"
            for i, liability in enumerate(contract_info.breach_liability, 1):
                value = self._format_extracted_field(liability)
                source = self._get_source_info(liability)
                markdown += f"{i}. {value}\n\n{source}\n\n"
            markdown += "\n"

        # 合同主要条款
        if contract_info.contract_terms:
            markdown += "### 合同主要条款/特殊约定\n"
            for i, term in enumerate(contract_info.contract_terms, 1):
                value = self._format_extracted_field(term)
                source = self._get_source_info(term)
                markdown += f"{i}. {value}\n\n{source}\n\n"
            markdown += "\n"
        
        # 合同单项信息
        contract_fields = [
            ("付款方式与周期", contract_info.payment_terms),
            ("项目完成期限/交付要求", contract_info.delivery_requirements),
            ("投标有效期", contract_info.bid_validity),
            ("知识产权归属", contract_info.intellectual_property),
            ("保密协议要求", contract_info.confidentiality),
        ]
        
        for field_name, field_value in contract_fields:
            if field_value.value:
                markdown += f"### {field_name}\n"
                value = self._format_extracted_field(field_value)
                source = self._get_source_info(field_value)
                markdown += f"{value}\n\n{source}\n\n"

        # 潜在风险点提示
        if contract_info.risk_warnings:
            markdown += "### 🚨 潜在风险点提示\n"
            for i, risk in enumerate(contract_info.risk_warnings, 1):
                value = self._format_extracted_field(risk)
                source = self._get_source_info(risk)
                notes = risk.notes if risk.notes else ""
                markdown += f"{i}. **{value}**\n\n{source}"
                if notes:
                    markdown += f"\n   - 风险分析：{notes}"
                markdown += "\n\n"
            markdown += "\n"
        
        # 处理说明
        markdown += "---\n\n## 处理说明\n\n"

        # 添加页码说明
        markdown += "- **页码说明**：第-1页表示该信息的具体页码无法确定，可能是由于文档处理过程中页码信息丢失或LLM提取时未包含页码标记\n"

        # 添加其他处理说明
        if result.processing_notes:
            for note in result.processing_notes:
                markdown += f"- {note}\n"
        
        # 错误信息
        if state.error_messages:
            markdown += "\n## 错误信息\n\n"
            for error in state.error_messages:
                markdown += f"- ❌ {error}\n"
        
        return markdown
    
    def _format_extracted_field(self, field: ExtractedField) -> str:
        """格式化提取的字段"""
        if not field or not field.value:
            return "招标文件中未提及"
        return field.value
    
    def _get_source_info(self, field: ExtractedField) -> str:
        """获取来源信息（用于段落中新起一行显示）"""
        # 如果字段不存在或没有来源信息，返回来源未知
        if not field or not field.source or not field.source.source_text:
            return "来源：来源未知"

        # 如果字段值为"招标文件中未提及"，返回来源未知
        if field.value and "招标文件中未提及" in field.value:
            return "来源：来源未知"

        # 尝试解析多个来源
        multiple_sources = self._parse_multiple_sources(field.source.source_text)

        if len(multiple_sources) > 1:
            # 多个来源，分别列出
            source_lines = []
            for source_info in multiple_sources:
                page_num = source_info['page']
                source_text = source_info['text'][:50]
                if len(source_info['text']) > 50:
                    source_text += "..."
                source_lines.append(f"来源：第{page_num}页 | 原文：{source_text}")
            return '\n'.join(source_lines)
        else:
            # 单个来源，使用原有逻辑
            # 强制添加页码信息
            page_number = field.source.page_number
            if not page_number:
                # 如果没有页码，尝试从来源文本中提取
                page_number = self._extract_page_number_from_source(field.source.source_text)

            # 如果仍然没有页码，设置默认值并记录警告
            if not page_number:
                logger.warning(f"来源信息缺少页码，使用默认值-1，来源文本: {field.source.source_text[:50]}...")
                page_number = -1

            source_parts = [f"第{page_number}页"]

            # 添加章节信息（如果有）
            if field.source.section:
                source_parts.append(f"章节: {field.source.section}")

            # 截取来源文本的前50个字符
            source_text = field.source.source_text[:50]
            if len(field.source.source_text) > 50:
                source_text += "..."

            # 组合来源信息，确保页码信息始终存在
            location_info = " ｜ ".join(source_parts)
            return f"来源：{location_info} ｜ 原文: {source_text}"

    def _extract_page_number_from_source(self, source_text: str) -> Optional[int]:
        """从来源文本中提取页码信息"""
        if not source_text:
            return None

        import re
        # 查找页码标记模式：--- 第X页 ---
        page_pattern = r'--- 第(\d+)页 ---'
        match = re.search(page_pattern, source_text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass

        return None

    def _parse_multiple_sources(self, source_text: str) -> list:
        """解析多个来源信息"""
        if not source_text:
            return []

        import re

        # 检查是否包含多个页码的模式
        # 例如："文档片段第58页、第72页" 或 "文档片段第17页、第58页"
        multi_page_pattern = r'文档片段第(\d+)页(?:、第(\d+)页)*'
        match = re.search(multi_page_pattern, source_text)

        if match:
            # 提取所有页码
            page_numbers = []
            # 获取第一个页码
            page_numbers.append(int(match.group(1)))

            # 查找所有其他页码
            all_pages_pattern = r'第(\d+)页'
            all_matches = re.findall(all_pages_pattern, source_text)
            for page_str in all_matches:
                page_num = int(page_str)
                if page_num not in page_numbers:
                    page_numbers.append(page_num)

            # 为每个页码创建来源信息
            sources = []
            for page_num in page_numbers:
                # 尝试为每个页码找到对应的文本片段
                # 这里简化处理，使用相同的源文本
                sources.append({
                    'page': page_num,
                    'text': source_text
                })

            return sources

        # 如果没有找到多页码模式，检查是否有单个页码
        single_page_pattern = r'--- 第(\d+)页 ---'
        match = re.search(single_page_pattern, source_text)
        if match:
            return [{
                'page': int(match.group(1)),
                'text': source_text
            }]

        # 如果都没有找到，返回空列表
        return []

    def _format_structured_content(self, content: str) -> str:
        """格式化结构化内容，如组成与编制规范"""
        if not content:
            return ""

        # 尝试识别列表项并格式化
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否是编号列表项
            if any(line.startswith(prefix) for prefix in ['(', '（', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']):
                formatted_lines.append(f"- {line}")
            elif line.startswith(('第一', '第二', '第三', '第四', '第五', '第六', '第七', '第八', '第九')):
                formatted_lines.append(f"- {line}")
            else:
                # 如果不是明显的列表项，但内容较短，可能是标题
                if len(line) < 50 and ('包括' in line or '应包含' in line or '组成' in line):
                    formatted_lines.append(f"\n**{line}**")
                else:
                    formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _format_qualification_requirements(self, qualification_items: list) -> str:
        """动态格式化投标人资格要求，不使用预定义分类"""
        if not qualification_items:
            return ""

        formatted_content = ""

        for i, item in enumerate(qualification_items, 1):
            value = self._format_extracted_field(item)
            source = self._get_source_info(item)

            # 尝试从原文中提取标题结构
            if item.source and item.source.source_text:
                # 检查是否包含明显的分类标题
                source_text = item.source.source_text
                if any(keyword in source_text for keyword in ['资质要求', '业绩要求', '人员要求', '信誉要求', '其他要求']):
                    # 尝试提取标题
                    title = self._extract_title_from_source(source_text)
                    if title and title not in formatted_content:
                        formatted_content += f"\n#### {title}\n\n"

            formatted_content += f"{i}. {value}\n\n{source}\n\n"

        return formatted_content

    def _extract_title_from_source(self, source_text: str) -> str:
        """从来源文本中提取标题"""
        if not source_text:
            return ""

        # 查找常见的资格要求标题模式
        import re
        title_patterns = [
            r'([^。]*?资质要求[^。]*?)[:：]',
            r'([^。]*?业绩要求[^。]*?)[:：]',
            r'([^。]*?人员要求[^。]*?)[:：]',
            r'([^。]*?信誉要求[^。]*?)[:：]',
            r'([^。]*?其他要求[^。]*?)[:：]',
            r'(\([^)]*\))\s*[：:]',  # 括号内的标题
        ]

        for pattern in title_patterns:
            match = re.search(pattern, source_text)
            if match:
                title = match.group(1).strip()
                # 清理标题
                title = title.replace('（', '').replace('）', '').replace('(', '').replace(')', '')
                if len(title) < 20:  # 标题不应该太长
                    return title

        return ""

    def _format_composition_content(self, content: str) -> str:
        """格式化组成与编制规范内容，保持原文结构"""
        if not content:
            return ""

        # 对于组成与编制规范，保持原文的结构，不强制转换为列表
        # 只进行基本的格式清理
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _format_document_structure(self, content: str) -> str:
        """格式化投标文件章节框架（目录）为直观的目录格式"""
        if not content:
            return ""

        # 特殊处理包含两个信封目录的情况
        if '第一个信封' in content and '第二个信封' in content:
            return self._format_dual_envelope_structure(content)

        # 处理单个目录内容
        # 首先尝试按分号或句号分割
        if '；' in content:
            items = content.split('；')
        elif ';' in content:
            items = content.split(';')
        elif '。' in content and content.count('。') > 2:
            items = content.split('。')
        else:
            # 如果没有明显的分隔符，按换行分割
            items = content.split('\n')

        formatted_lines = []

        for item in items:
            item = item.strip()
            if not item:
                continue

            # 移除末尾的标点符号
            item = item.rstrip('。；;')

            # 检查是否是目录项
            if any(item.startswith(prefix) for prefix in ['第一', '第二', '第三', '第四', '第五', '第六', '第七', '第八', '第九', '第十']):
                formatted_lines.append(f"**{item}**")
            elif any(item.startswith(prefix) for prefix in ['一、', '二、', '三、', '四、', '五、', '六、', '七、', '八、', '九、', '十、']):
                formatted_lines.append(f"**{item}**")
            elif any(item.startswith(prefix) for prefix in ['(', '（', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']):
                formatted_lines.append(f"- {item}")
            elif any(item.startswith(prefix) for prefix in ['①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨']):
                formatted_lines.append(f"  - {item}")
            elif '信封' in item:
                formatted_lines.append(f"### {item}")
            else:
                # 其他内容，如果较短可能是标题，较长则作为普通项
                if len(item) < 30:
                    formatted_lines.append(f"**{item}**")
                else:
                    formatted_lines.append(f"- {item}")

        return '\n'.join(formatted_lines)

    def _format_dual_envelope_structure(self, content: str) -> str:
        """格式化包含两个信封的目录结构"""
        import re

        # 分离两个信封的内容
        # 查找第一个信封的内容
        first_envelope_pattern = r'第一个信封（商务及技术文件）[^：]*：([^。]*第二个信封)'
        match = re.search(first_envelope_pattern, content)

        if match:
            # 提取第一个信封的内容
            first_part = match.group(1).replace('第二个信封', '').strip()

            # 提取第二个信封的内容
            second_envelope_pattern = r'第二个信封（报价文件）[^：]*：([^。]*)'
            second_match = re.search(second_envelope_pattern, content)
            second_part = second_match.group(1).strip() if second_match else ""

            formatted_lines = []

            # 格式化第一个信封
            formatted_lines.append("##### 第一个信封（商务及技术文件）")
            first_items = self._parse_directory_items(first_part)
            for item in first_items:
                formatted_lines.append(f"**{item}**")

            # 添加分隔
            formatted_lines.append("")

            # 格式化第二个信封
            formatted_lines.append("##### 第二个信封（报价文件）")
            second_items = self._parse_directory_items(second_part)
            for item in second_items:
                formatted_lines.append(f"**{item}**")

            return '\n'.join(formatted_lines)

        # 如果无法解析，回退到原始格式
        return content

    def _parse_directory_items(self, content: str) -> list:
        """解析目录项"""
        if not content:
            return []

        # 按分号、句号或逗号分割
        if '；' in content:
            items = content.split('；')
        elif ';' in content:
            items = content.split(';')
        elif '，' in content:
            items = content.split('，')
        elif ',' in content:
            items = content.split(',')
        else:
            items = [content]

        parsed_items = []
        for item in items:
            item = item.strip().rstrip('。；;，,')
            if item:
                parsed_items.append(item)

        return parsed_items

    def _identify_directory_type(self, content: str) -> str:
        """识别目录类型"""
        if not content:
            return "目录"

        content_lower = content.lower()

        # 检查是否包含商务及技术文件相关内容
        if any(keyword in content_lower for keyword in ['商务', '技术', '实施方案', '授权委托书', '资格审查', '售后服务']):
            return "第一个信封（商务及技术文件）目录"

        # 检查是否包含报价文件相关内容
        if any(keyword in content_lower for keyword in ['报价', '价格', '清单', '报价表']):
            return "第二个信封（报价文件）目录"

        # 检查是否明确提到信封
        if '第一个信封' in content or '商务及技术文件' in content:
            return "第一个信封（商务及技术文件）目录"

        if '第二个信封' in content or '报价文件' in content:
            return "第二个信封（报价文件）目录"

        # 默认返回通用目录
        return "投标文件目录"

    def _get_source_info_for_table(self, source) -> str:
        """获取表格用的来源信息（简化版）"""
        if not source:
            return "来源未知"

        # 处理DocumentSource对象
        if hasattr(source, 'page_number') and hasattr(source, 'source_text'):
            if not source.source_text:
                return "来源未知"

            # 构建简化的来源信息
            page_number = source.page_number if source.page_number else -1
            # 截取来源文本的前30个字符用于表格显示
            source_text = source.source_text[:30]
            if len(source.source_text) > 30:
                source_text += "..."
            return f"第{page_number}页 ｜ 原文: {source_text}"

        return "来源未知"

    def _clean_table_content(self, content: str) -> str:
        """清理表格内容，避免换行符等特殊字符导致的格式问题"""
        if not content:
            return ""

        # 替换换行符为空格
        content = content.replace('\n', ' ').replace('\r', ' ')
        # 替换制表符为空格
        content = content.replace('\t', ' ')
        # 替换管道符，避免破坏表格结构
        content = content.replace('|', '｜')
        # 压缩多个空格为单个空格
        import re
        content = re.sub(r'\s+', ' ', content)
        # 去除首尾空格
        content = content.strip()

        return content
    
    def _save_report(self, content: str, document_name: str) -> str:
        """保存报告到文件"""
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_doc_name = "".join(c for c in document_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"投标分析报告_{safe_doc_name}_{timestamp}.md"
        
        # 完整路径
        output_path = os.path.join(settings.output_dir, filename)
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return output_path

def create_output_formatter_node():
    """创建输出格式化节点函数"""
    formatter = OutputFormatter()
    
    def output_formatter_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """输出格式化节点函数"""
        # 转换为GraphStateModel对象
        graph_state = GraphStateModel(**state)

        # 执行输出格式化
        graph_state = formatter.format_output(graph_state)

        # 转换回字典格式
        return graph_state.model_dump()
    
    return output_formatter_node
