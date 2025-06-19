"""
数据模型定义
Data models for the Intelligent Bidding Assistant
"""

from typing import List, Optional, Dict, Any, TypedDict, Union, Annotated
from pydantic import BaseModel, Field
from datetime import datetime
from langgraph.graph import add_messages
import operator


def merge_analysis_results(existing, new):
    """
    智能合并分析结果，保留所有智能体的提取结果

    Args:
        existing: 现有的分析结果（可能是dict或BiddingAnalysisResult）
        new: 新的分析结果（可能是dict或BiddingAnalysisResult）

    Returns:
        BiddingAnalysisResult: 合并后的结果
    """
    # 处理None情况
    if existing is None:
        return new
    if new is None:
        return existing

    # 转换为BiddingAnalysisResult对象
    if isinstance(existing, dict):
        existing = BiddingAnalysisResult(**existing)
    if isinstance(new, dict):
        new = BiddingAnalysisResult(**new)

    # 创建合并后的结果
    merged = BiddingAnalysisResult(
        document_name=new.document_name or existing.document_name,
        analysis_time=new.analysis_time or existing.analysis_time
    )

    # 合并基础信息 - 保留有值的字段
    merged.basic_information = merge_basic_information(
        existing.basic_information,
        new.basic_information
    )

    # 合并评分标准 - 保留有值的字段
    merged.scoring_criteria = merge_scoring_criteria(
        existing.scoring_criteria,
        new.scoring_criteria
    )

    # 合并合同信息 - 保留有值的字段
    merged.contract_information = merge_contract_information(
        existing.contract_information,
        new.contract_information
    )

    # 合并处理说明
    merged.processing_notes = list(set(existing.processing_notes + new.processing_notes))

    return merged


def merge_basic_information(existing, new):
    """合并基础信息，保留有值的字段"""
    # 如果new的字段有值，使用new的；否则保留existing的
    merged = BasicInformation()

    # 合并简单字段
    for field_name in ['project_name', 'tender_number', 'budget_amount', 'bid_deadline',
                       'bid_opening_time', 'bid_bond_amount', 'bid_bond_account',
                       'purchaser_name', 'purchaser_contact', 'agent_name', 'agent_contact']:
        existing_field = getattr(existing, field_name, ExtractedField())
        new_field = getattr(new, field_name, ExtractedField())

        # 如果new字段有值，使用new的；否则使用existing的
        if new_field.value and new_field.value.strip():
            setattr(merged, field_name, new_field)
        else:
            setattr(merged, field_name, existing_field)

    # 合并复杂字段
    merged.qualification_criteria = merge_qualification_criteria(
        existing.qualification_criteria, new.qualification_criteria
    )
    merged.bid_document_requirements = merge_bid_document_requirements(
        existing.bid_document_requirements, new.bid_document_requirements
    )
    merged.bid_evaluation_process = merge_bid_evaluation_process(
        existing.bid_evaluation_process, new.bid_evaluation_process
    )

    return merged


def merge_scoring_criteria(existing, new):
    """合并评分标准"""
    merged = ScoringCriteria()

    # 合并列表字段 - 去重合并
    merged.preliminary_review = merge_extracted_field_list(
        existing.preliminary_review, new.preliminary_review
    )
    merged.detailed_scoring = merge_scoring_item_list(
        existing.detailed_scoring, new.detailed_scoring
    )
    merged.bonus_points = merge_extracted_field_list(
        existing.bonus_points, new.bonus_points
    )
    merged.disqualification_clauses = merge_extracted_field_list(
        existing.disqualification_clauses, new.disqualification_clauses
    )

    # 合并单个字段
    merged.evaluation_method = (new.evaluation_method
                               if new.evaluation_method.value and new.evaluation_method.value.strip()
                               else existing.evaluation_method)

    # 合并分值构成
    merged.score_composition = merge_score_composition(
        existing.score_composition, new.score_composition
    )

    return merged


def merge_contract_information(existing, new):
    """合并合同信息"""
    merged = ContractInformation()

    # 合并列表字段
    merged.breach_liability = merge_extracted_field_list(
        existing.breach_liability, new.breach_liability
    )
    merged.contract_terms = merge_extracted_field_list(
        existing.contract_terms, new.contract_terms
    )
    merged.risk_warnings = merge_extracted_field_list(
        existing.risk_warnings, new.risk_warnings
    )

    # 合并单个字段
    for field_name in ['payment_terms', 'delivery_requirements', 'bid_validity',
                       'intellectual_property', 'confidentiality']:
        existing_field = getattr(existing, field_name, ExtractedField())
        new_field = getattr(new, field_name, ExtractedField())

        if new_field.value and new_field.value.strip():
            setattr(merged, field_name, new_field)
        else:
            setattr(merged, field_name, existing_field)

    return merged


def merge_extracted_field_list(existing_list, new_list):
    """合并ExtractedField列表，去重"""
    merged = list(existing_list)

    for new_item in new_list:
        # 检查是否已存在相同内容
        exists = False
        for existing_item in merged:
            if (existing_item.value == new_item.value and
                existing_item.source and new_item.source and
                existing_item.source.source_text == new_item.source.source_text):
                exists = True
                break

        if not exists:
            merged.append(new_item)

    return merged


def merge_scoring_item_list(existing_list, new_list):
    """合并ScoringItem列表，去重"""
    merged = list(existing_list)

    for new_item in new_list:
        # 检查是否已存在相同内容
        exists = False
        for existing_item in merged:
            if (existing_item.category == new_item.category and
                existing_item.item_name == new_item.item_name):
                exists = True
                break

        if not exists:
            merged.append(new_item)

    return merged


def merge_qualification_criteria(existing, new):
    """合并资格审查条件"""
    merged = QualificationCriteria()

    merged.company_certifications = merge_extracted_field_list(
        existing.company_certifications, new.company_certifications
    )
    merged.project_experience = merge_extracted_field_list(
        existing.project_experience, new.project_experience
    )
    merged.team_requirements = merge_extracted_field_list(
        existing.team_requirements, new.team_requirements
    )
    merged.other_requirements = merge_extracted_field_list(
        existing.other_requirements, new.other_requirements
    )

    return merged


def merge_bid_document_requirements(existing, new):
    """合并投标文件要求"""
    merged = BidDocumentRequirements()

    merged.composition_and_format = merge_extracted_field_list(
        existing.composition_and_format, new.composition_and_format
    )
    merged.binding_and_sealing = merge_extracted_field_list(
        existing.binding_and_sealing, new.binding_and_sealing
    )
    merged.signature_and_seal = merge_extracted_field_list(
        existing.signature_and_seal, new.signature_and_seal
    )
    merged.document_structure = merge_extracted_field_list(
        existing.document_structure, new.document_structure
    )

    return merged


def merge_bid_evaluation_process(existing, new):
    """合并开评定标流程"""
    merged = BidEvaluationProcess()

    merged.bid_opening = merge_extracted_field_list(
        existing.bid_opening, new.bid_opening
    )
    merged.evaluation = merge_extracted_field_list(
        existing.evaluation, new.evaluation
    )
    merged.award_decision = merge_extracted_field_list(
        existing.award_decision, new.award_decision
    )

    return merged


def merge_score_composition(existing, new):
    """合并分值构成"""
    merged = ScoreComposition()

    # 合并单个字段
    for field_name in ['technical_score', 'commercial_score', 'price_score']:
        existing_field = getattr(existing, field_name, ExtractedField())
        new_field = getattr(new, field_name, ExtractedField())

        if new_field.value and new_field.value.strip():
            setattr(merged, field_name, new_field)
        else:
            setattr(merged, field_name, existing_field)

    # 合并其他分数列表
    merged.other_scores = merge_extracted_field_list(
        existing.other_scores, new.other_scores
    )

    return merged

class DocumentSource(BaseModel):
    """文档来源信息"""
    page_number: Optional[int] = Field(None, description="页码")
    section: Optional[str] = Field(None, description="章节")
    paragraph: Optional[str] = Field(None, description="段落")
    source_text: Optional[str] = Field(None, description="原文片段")

class ExtractedField(BaseModel):
    """提取的字段信息"""
    value: Optional[str] = Field(None, description="提取的值")
    source: Optional[DocumentSource] = Field(None, description="来源信息")
    confidence: Optional[float] = Field(None, description="置信度")
    notes: Optional[str] = Field(None, description="备注")

class QualificationCriteria(BaseModel):
    """资格审查硬性条件"""
    company_certifications: List[ExtractedField] = Field(default_factory=list, description="企业资质要求")
    project_experience: List[ExtractedField] = Field(default_factory=list, description="类似项目业绩要求")
    team_requirements: List[ExtractedField] = Field(default_factory=list, description="项目团队人员要求")
    other_requirements: List[ExtractedField] = Field(default_factory=list, description="其他硬性要求")

class BidDocumentRequirements(BaseModel):
    """投标文件要求"""
    composition_and_format: List[ExtractedField] = Field(default_factory=list, description="组成与编制规范")
    binding_and_sealing: List[ExtractedField] = Field(default_factory=list, description="装订与密封要求")
    signature_and_seal: List[ExtractedField] = Field(default_factory=list, description="签字盖章要求")
    document_structure: List[ExtractedField] = Field(default_factory=list, description="投标文件章节框架（目录）")

class BidEvaluationProcess(BaseModel):
    """开评定标流程"""
    bid_opening: List[ExtractedField] = Field(default_factory=list, description="开标环节（时间、地点、程序）")
    evaluation: List[ExtractedField] = Field(default_factory=list, description="评标环节（评委会、评审方法/标准、主要流程）")
    award_decision: List[ExtractedField] = Field(default_factory=list, description="定标环节（定标原则、中标通知）")

class BasicInformation(BaseModel):
    """基础信息模块"""
    project_name: ExtractedField = Field(default_factory=ExtractedField, description="项目名称")
    tender_number: ExtractedField = Field(default_factory=ExtractedField, description="招标编号")
    budget_amount: ExtractedField = Field(default_factory=ExtractedField, description="采购预算金额")
    bid_deadline: ExtractedField = Field(default_factory=ExtractedField, description="投标截止时间")
    bid_opening_time: ExtractedField = Field(default_factory=ExtractedField, description="开标时间")
    bid_bond_amount: ExtractedField = Field(default_factory=ExtractedField, description="投标保证金金额")
    bid_bond_account: ExtractedField = Field(default_factory=ExtractedField, description="投标保证金缴纳账户信息")
    purchaser_name: ExtractedField = Field(default_factory=ExtractedField, description="采购人名称")
    purchaser_contact: ExtractedField = Field(default_factory=ExtractedField, description="采购人联系方式")
    agent_name: ExtractedField = Field(default_factory=ExtractedField, description="采购代理机构名称")
    agent_contact: ExtractedField = Field(default_factory=ExtractedField, description="采购代理机构联系人及联系方式")
    qualification_criteria: QualificationCriteria = Field(default_factory=QualificationCriteria, description="资格审查硬性条件")
    bid_document_requirements: BidDocumentRequirements = Field(default_factory=BidDocumentRequirements, description="投标文件要求")
    bid_evaluation_process: BidEvaluationProcess = Field(default_factory=BidEvaluationProcess, description="开评定标流程")

class ScoreComposition(BaseModel):
    """分值构成"""
    technical_score: ExtractedField = Field(default_factory=ExtractedField, description="技术分占比")
    commercial_score: ExtractedField = Field(default_factory=ExtractedField, description="商务分占比")
    price_score: ExtractedField = Field(default_factory=ExtractedField, description="价格分占比")
    other_scores: List[ExtractedField] = Field(default_factory=list, description="其他部分占比")

class ScoringItem(BaseModel):
    """评分项"""
    category: str = Field(description="评分类别")
    item_name: str = Field(description="评分项名称")
    max_score: Optional[Union[float, str]] = Field(None, description="最高分值（数字或说明）")
    criteria: Optional[str] = Field(None, description="评分标准")
    source: Optional[DocumentSource] = Field(None, description="来源信息")

class ScoringCriteria(BaseModel):
    """评分标准分析模块"""
    preliminary_review: List[ExtractedField] = Field(default_factory=list, description="初步评审标准")
    evaluation_method: ExtractedField = Field(default_factory=ExtractedField, description="详细评审方法")
    score_composition: ScoreComposition = Field(default_factory=ScoreComposition, description="分值构成")
    detailed_scoring: List[ScoringItem] = Field(default_factory=list, description="详细评分细则")
    bonus_points: List[ExtractedField] = Field(default_factory=list, description="加分项明细")
    disqualification_clauses: List[ExtractedField] = Field(default_factory=list, description="否决项条款")

class ContractInformation(BaseModel):
    """合同信息模块"""
    breach_liability: List[ExtractedField] = Field(default_factory=list, description="违约责任")
    contract_terms: List[ExtractedField] = Field(default_factory=list, description="合同主要条款/特殊约定")
    payment_terms: ExtractedField = Field(default_factory=ExtractedField, description="付款方式与周期")
    delivery_requirements: ExtractedField = Field(default_factory=ExtractedField, description="项目完成期限/交付要求")
    bid_validity: ExtractedField = Field(default_factory=ExtractedField, description="投标有效期")
    intellectual_property: ExtractedField = Field(default_factory=ExtractedField, description="知识产权归属")
    confidentiality: ExtractedField = Field(default_factory=ExtractedField, description="保密协议要求")
    risk_warnings: List[ExtractedField] = Field(default_factory=list, description="潜在风险点提示")

class BiddingAnalysisResult(BaseModel):
    """投标分析结果"""
    document_name: str = Field(description="文档名称")
    analysis_time: datetime = Field(default_factory=datetime.now, description="分析时间")
    basic_information: BasicInformation = Field(default_factory=BasicInformation, description="基础信息")
    scoring_criteria: ScoringCriteria = Field(default_factory=ScoringCriteria, description="评分标准")
    contract_information: ContractInformation = Field(default_factory=ContractInformation, description="合同信息")
    processing_notes: List[str] = Field(default_factory=list, description="处理说明")

class GraphState(TypedDict):
    """Langgraph状态模型"""
    document_path: Annotated[str, lambda x, y: y]  # 后写入的值覆盖前面的值
    document_content: Annotated[Optional[str], lambda x, y: y]  # 后写入的值覆盖前面的值
    chunks: Annotated[List[str], lambda x, y: y]  # 后写入的值覆盖前面的值
    vector_store: Annotated[Optional[Any], lambda x, y: y]  # 后写入的值覆盖前面的值
    analysis_result: Annotated[BiddingAnalysisResult, merge_analysis_results]  # 智能合并分析结果
    current_step: Annotated[str, lambda x, y: y]  # 后写入的值覆盖前面的值
    error_messages: Annotated[List[str], operator.add]  # 支持并发追加
    retry_count: Annotated[int, lambda x, y: y]  # 后写入的值覆盖前面的值

class GraphStateModel(BaseModel):
    """Pydantic版本的GraphState，用于数据验证和转换"""
    document_path: str = Field(description="文档路径")
    document_content: Optional[str] = Field(None, description="文档内容")
    chunks: List[str] = Field(default_factory=list, description="文档分块")
    vector_store: Optional[Any] = Field(None, description="向量存储")
    analysis_result: BiddingAnalysisResult = Field(default_factory=BiddingAnalysisResult, description="分析结果")
    current_step: str = Field(default="start", description="当前处理步骤")
    error_messages: List[str] = Field(default_factory=list, description="错误信息")
    retry_count: int = Field(default=0, description="重试次数")

    class Config:
        arbitrary_types_allowed = True
