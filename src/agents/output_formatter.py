"""
结果格式化输出节点
Output formatting node for the Langgraph workflow
"""

from typing import Dict, Any
from loguru import logger
from src.models.data_models import GraphStateModel, ExtractedField
from config.settings import settings
import os
from datetime import datetime

class OutputFormatter:
    """结果格式化器"""
    
    def __init__(self):
        """初始化格式化器"""
        pass
    
    def format_output(self, state: GraphStateModel) -> GraphStateModel:
        """
        格式化输出结果

        Args:
            state: 图状态

        Returns:
            GraphStateModel: 更新后的状态
        """
        try:
            logger.info("开始格式化输出结果")
            
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
            state.error_messages.append(error_msg)
            state.current_step = "error"
        
        return state
    
    def _generate_markdown_report(self, state: GraphStateModel) -> str:
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
            source = self._get_source_info(field_value)
            markdown += f"| {field_name} | {value} | {source} |\n"
        
        # 资格审查硬性条件
        markdown += "\n### 资格审查硬性条件\n\n"
        
        qualification = basic_info.qualification_criteria
        
        if qualification.company_certifications:
            markdown += "#### 企业资质要求\n"
            for i, cert in enumerate(qualification.company_certifications, 1):
                value = self._format_extracted_field(cert)
                source = self._get_source_info(cert)
                markdown += f"{i}. {value} *({source})*\n"
            markdown += "\n"
        
        if qualification.project_experience:
            markdown += "#### 类似项目业绩要求\n"
            for i, exp in enumerate(qualification.project_experience, 1):
                value = self._format_extracted_field(exp)
                source = self._get_source_info(exp)
                markdown += f"{i}. {value} *({source})*\n"
            markdown += "\n"
        
        if qualification.team_requirements:
            markdown += "#### 项目团队人员要求\n"
            for i, req in enumerate(qualification.team_requirements, 1):
                value = self._format_extracted_field(req)
                source = self._get_source_info(req)
                markdown += f"{i}. {value} *({source})*\n"
            markdown += "\n"
        
        if qualification.other_requirements:
            markdown += "#### 其他硬性要求\n"
            for i, req in enumerate(qualification.other_requirements, 1):
                value = self._format_extracted_field(req)
                source = self._get_source_info(req)
                markdown += f"{i}. {value} *({source})*\n"
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
                markdown += f"{i}. {value} *({source})*\n"
            markdown += "\n"
        
        # 详细评审方法
        if scoring.evaluation_method.value:
            markdown += "### 详细评审方法\n"
            value = self._format_extracted_field(scoring.evaluation_method)
            source = self._get_source_info(scoring.evaluation_method)
            markdown += f"{value} *({source})*\n\n"
        
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
                source = self._get_source_info(field_value)
                markdown += f"| {field_name} | {value} | {source} |\n"
        
        for other_score in score_comp.other_scores:
            if other_score.value:
                value = self._format_extracted_field(other_score)
                source = self._get_source_info(other_score)
                markdown += f"| 其他 | {value} | {source} |\n"
        
        markdown += "\n"
        
        # 详细评分细则表
        if scoring.detailed_scoring:
            markdown += "### 详细评分细则表\n\n"
            markdown += "| 评分类别 | 评分项 | 最高分值 | 评分标准 |\n"
            markdown += "|----------|--------|----------|----------|\n"
            
            for item in scoring.detailed_scoring:
                category = item.category or "未分类"
                item_name = item.item_name or "未命名"
                max_score = str(item.max_score) if item.max_score is not None else "未指定"
                criteria = item.criteria or "未指定"
                markdown += f"| {category} | {item_name} | {max_score} | {criteria} |\n"
            
            markdown += "\n"
        
        # 加分项明细
        if scoring.bonus_points:
            markdown += "### 加分项明细\n"
            for i, bonus in enumerate(scoring.bonus_points, 1):
                value = self._format_extracted_field(bonus)
                source = self._get_source_info(bonus)
                markdown += f"{i}. {value} *({source})*\n"
            markdown += "\n"
        
        # 否决项条款
        if scoring.disqualification_clauses:
            markdown += "### ⚠️ 否决项条款（重要）\n"
            for i, clause in enumerate(scoring.disqualification_clauses, 1):
                value = self._format_extracted_field(clause)
                source = self._get_source_info(clause)
                markdown += f"{i}. **{value}** *({source})*\n"
            markdown += "\n"
        
        # 其他重要信息模块
        markdown += "---\n\n## 三、其他重要信息模块\n\n"
        
        other_info = result.other_information
        
        # 合同主要条款
        if other_info.contract_terms:
            markdown += "### 合同主要条款/特殊约定\n"
            for i, term in enumerate(other_info.contract_terms, 1):
                value = self._format_extracted_field(term)
                source = self._get_source_info(term)
                markdown += f"{i}. {value} *({source})*\n"
            markdown += "\n"
        
        # 其他单项信息
        other_fields = [
            ("付款方式与周期", other_info.payment_terms),
            ("项目完成期限/交付要求", other_info.delivery_requirements),
            ("投标有效期", other_info.bid_validity),
            ("知识产权归属", other_info.intellectual_property),
            ("保密协议要求", other_info.confidentiality),
        ]
        
        for field_name, field_value in other_fields:
            if field_value.value:
                markdown += f"### {field_name}\n"
                value = self._format_extracted_field(field_value)
                source = self._get_source_info(field_value)
                markdown += f"{value} *({source})*\n\n"
        
        # 潜在风险点提示
        if other_info.risk_warnings:
            markdown += "### 🚨 潜在风险点提示\n"
            for i, risk in enumerate(other_info.risk_warnings, 1):
                value = self._format_extracted_field(risk)
                source = self._get_source_info(risk)
                notes = risk.notes if risk.notes else ""
                markdown += f"{i}. **{value}** *({source})*"
                if notes:
                    markdown += f"\n   - 风险分析：{notes}"
                markdown += "\n"
            markdown += "\n"
        
        # 处理说明
        if result.processing_notes:
            markdown += "---\n\n## 处理说明\n\n"
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
        """获取来源信息"""
        if not field or not field.source or not field.source.source_text:
            return "来源未知"
        
        # 截取来源文本的前50个字符
        source_text = field.source.source_text[:50]
        if len(field.source.source_text) > 50:
            source_text += "..."
        
        return f"原文: {source_text}"
    
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
        return graph_state.dict()
    
    return output_formatter_node
