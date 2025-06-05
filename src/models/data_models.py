"""
数据模型定义
Data models for the Intelligent Bidding Assistant
"""

from typing import List, Optional, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

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
    max_score: Optional[float] = Field(None, description="最高分值")
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

class OtherInformation(BaseModel):
    """其他重要信息模块"""
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
    other_information: OtherInformation = Field(default_factory=OtherInformation, description="其他重要信息")
    processing_notes: List[str] = Field(default_factory=list, description="处理说明")

class GraphState(TypedDict):
    """Langgraph状态模型"""
    document_path: str
    document_content: Optional[str]
    chunks: List[str]
    vector_store: Optional[Any]
    analysis_result: BiddingAnalysisResult
    current_step: str
    error_messages: List[str]
    retry_count: int

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
