"""LLM 转发核心服务

将聊天请求转发到配置的 LLM 服务器，支持负载均衡、流式响应和重试机制。
"""
import json
from typing import Dict, List, Optional, Any, AsyncIterator
import httpx
from config.settings import get_settings, ServerConfig
from config.logging import get_logger


logger = get_logger(__name__)


class LLMForwarder:
    """LLM请求转发服务"""

    def __init__(self):
        self.http_client: Optional[httpx.AsyncClient] = None
        self._model_server_map: Dict[str, ServerConfig] = {}
        self._counters: Dict[str, int] = {}

    async def initialize(self):
        """初始化HTTP客户端和服务器映射"""
        settings = get_settings()

        # 构建模型到服务器的映射
        for server_name, server_config in settings.servers.items():
            for model in server_config.models:
                if model not in self._model_server_map:
                    self._model_server_map[model] = []
                self._model_server_map[model].append(server_config)
                self._counters[model] = 0

        # 初始化HTTP客户端
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=500,
                max_keepalive_connections=50,
                keepalive_expiry=180,
            ),
            timeout=httpx.Timeout(
                connect=5.0,
                read=300.0,
                write=5.0,
                pool=5.0,
            ),
            transport=httpx.AsyncHTTPTransport(
                retries=2,
                http2=True,
            ),
        )

    def get_server_for_model(self, model: str) -> ServerConfig:
        """根据模型名获取服务器配置（轮询）"""
        if model not in self._model_server_map:
            raise ValueError(f"Unsupported model: {model}")

        servers = self._model_server_map[model]
        counter = self._counters[model] % len(servers)
        self._counters[model] += 1

        return servers[counter]

    def get_auth_headers(self, server_config: ServerConfig) -> Dict[str, str]:
        """生成认证头"""
        return {
            "Authorization": f"Bearer {server_config.api_key}",
            "Content-Type": "application/json",
        }

    async def forward_request(
        self,
        model: str,
        data: Dict[str, Any],
        stream: bool = False
    ) -> str | httpx.Response:
        """
        转发请求到LLM服务器

        Args:
            model: 模型名称
            data: 请求数据
            stream: 是否流式响应

        Returns:
            非流式: 返回响应文本
            流式: 返回Response对象
        """
        server_config = self.get_server_for_model(model)
        headers = self.get_auth_headers(server_config)

        url = f"{server_config.base_url}/chat/completions"

        if stream:
            return await self.http_client.post(
                url,
                json=data,
                headers=headers,
                timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
            )
        else:
            response = await self.http_client.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.text

    async def forward_stream(
        self,
        model: str,
        data: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """
        流式转发请求

        Args:
            model: 模型名称
            data: 请求数据

        Yields:
            SSE格式的数据块
        """
        server_config = self.get_server_for_model(model)
        headers = self.get_auth_headers(server_config)
        url = f"{server_config.base_url}/chat/completions"

        async with self.http_client.stream(
            "POST",
            url,
            json=data,
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_text():
                yield chunk

    def get_available_models(self) -> List[str]:
        """获取所有可用模型列表"""
        return list(self._model_server_map.keys())

    async def close(self):
        """关闭HTTP客户端"""
        if self.http_client:
            await self.http_client.aclose()
