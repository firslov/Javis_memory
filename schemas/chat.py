"""聊天相关的请求和响应模型

定义聊天完成、对话、消息、API Key 等相关的 Pydantic 模型。
"""
from typing import List, Optional, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="角色: user, assistant, system")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """聊天完成请求"""
    messages: List[ChatMessage] = Field(..., description="消息列表")
    model: str = Field(..., description="模型名称")
    temperature: Optional[float] = Field(0.7, description="温度参数")
    max_tokens: Optional[int] = Field(None, description="最大token数")
    stream: Optional[bool] = Field(False, description="是否流式响应")
    # 扩展字段
    user_id: Optional[int] = Field(None, description="用户ID")
    conversation_id: Optional[int] = Field(None, description="对话ID")
    enable_profile: Optional[bool] = Field(True, description="是否启用用户画像增强")


class ChatCompletionResponse(BaseModel):
    """聊天完成响应"""
    id: str = Field(..., description="响应ID")
    object: str = Field(default="chat.completion", description="对象类型")
    created: int = Field(..., description="创建时间戳")
    model: str = Field(..., description="模型名称")
    choices: List[dict] = Field(..., description="选择列表")
    usage: Optional[dict] = Field(None, description="使用情况")


class ConversationCreate(BaseModel):
    """创建对话请求"""
    user_id: int = Field(..., description="用户ID")
    title: Optional[str] = Field("新对话", description="对话标题")
    model: Optional[str] = Field("gpt-4o-mini", description="模型名称")


class ConversationResponse(BaseModel):
    """对话响应"""
    id: int = Field(..., description="对话ID")
    user_id: int = Field(..., description="用户ID")
    title: str = Field(..., description="对话标题")
    model: str = Field(..., description="模型名称")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")


class MessageResponse(BaseModel):
    """消息响应"""
    id: int = Field(..., description="消息ID")
    conversation_id: int = Field(..., description="对话ID")
    role: str = Field(..., description="角色")
    content: str = Field(..., description="内容")
    tokens: Optional[int] = Field(None, description="token数量")
    created_at: str = Field(..., description="创建时间")


class UserProfileResponse(BaseModel):
    """用户记忆响应（第二层记忆）"""
    name: Optional[str] = Field(None, description="名字")
    nickname: Optional[str] = Field(None, description="称呼")
    profession: Optional[str] = Field(None, description="职业")
    company: Optional[str] = Field(None, description="公司")
    job_title: Optional[str] = Field(None, description="职位")
    research_topics: Optional[str] = Field(None, description="研究方向")
    interests: Optional[str] = Field(None, description="兴趣爱好")
    goals: Optional[str] = Field(None, description="目标")
    communication_style: Optional[str] = Field(None, description="沟通风格")
    tone_preference: Optional[str] = Field(None, description="语气偏好")
    language: Optional[str] = Field(None, description="语言偏好")
    technical_level: Optional[str] = Field(None, description="技术水平")
    preferred_detail_level: Optional[str] = Field(None, description="详细程度偏好")
    custom_preferences: Optional[dict] = Field(None, description="自定义偏好")


class RecentConversationResponse(BaseModel):
    """近期对话响应（第三层记忆）"""
    conversation_id: int = Field(..., description="对话ID")
    title: str = Field(..., description="对话标题")
    conversation_date: str = Field(..., description="对话日期")
    key_points: Optional[str] = Field(None, description="关键信息")
    user_summary: Optional[str] = Field(None, description="用户发言摘要")
    topics: Optional[str] = Field(None, description="主题")
    tags: Optional[str] = Field(None, description="标签")
    message_count: int = Field(..., description="消息数量")
    user_message_count: int = Field(..., description="用户消息数量")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: dict = Field(..., description="错误详情")


class ModelsResponse(BaseModel):
    """模型列表响应"""
    object: str = Field(default="list", description="对象类型")
    data: List[dict] = Field(..., description="模型列表")


# API Key 相关模型
class APIKeyCreate(BaseModel):
    """创建API密钥请求"""
    name: str = Field(..., description="密钥名称", min_length=1, max_length=100)
    rate_limit: Optional[int] = Field(None, description="每分钟请求限制")
    daily_limit: Optional[int] = Field(None, description="每日请求限制")
    expires_days: Optional[int] = Field(None, description="有效期（天）", ge=1)


class APIKeyResponse(BaseModel):
    """API密钥响应"""
    id: int = Field(..., description="密钥ID")
    name: str = Field(..., description="密钥名称")
    key: str = Field(..., description="密钥值（仅在创建时返回完整值）")
    is_active: bool = Field(..., description="是否启用")
    rate_limit: Optional[int] = Field(None, description="每分钟请求限制")
    daily_limit: Optional[int] = Field(None, description="每日请求限制")
    expires_at: Optional[str] = Field(None, description="过期时间")
    last_used_at: Optional[str] = Field(None, description="最后使用时间")
    total_requests: int = Field(..., description="总请求次数")
    created_at: str = Field(..., description="创建时间")


class APIKeyInfo(BaseModel):
    """API密钥信息（不返回完整密钥）"""
    id: int = Field(..., description="密钥ID")
    name: str = Field(..., description="密钥名称")
    key_preview: str = Field(..., description="密钥预览（部分隐藏）")
    is_active: bool = Field(..., description="是否启用")
    rate_limit: Optional[int] = Field(None, description="每分钟请求限制")
    daily_limit: Optional[int] = Field(None, description="每日请求限制")
    expires_at: Optional[str] = Field(None, description="过期时间")
    last_used_at: Optional[str] = Field(None, description="最后使用时间")
    total_requests: int = Field(..., description="总请求次数")
    created_at: str = Field(..., description="创建时间")
