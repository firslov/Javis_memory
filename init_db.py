"""数据库初始化脚本

初始化数据库表结构并创建默认用户和 API Key。

功能:
    - 创建数据库表（用户、对话、消息、API Key、记忆系统）
    - 创建默认用户和 API Key
    - 支持会话记录和记忆文件索引
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from database.session import init_db, get_db_session, close_db
from database.repository import UserRepository
from services.auth import AuthService
from config.settings import get_settings


def print_header(text: str) -> None:
    """打印标题"""
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)


def print_success(text: str) -> None:
    """打印成功消息"""
    print(f"✓ {text}")


def print_error(text: str) -> None:
    """打印错误消息"""
    print(f"✗ {text}")


def check_config_file() -> bool:
    """检查配置文件是否存在"""
    config_path = Path("config/servers.yaml")
    example_path = Path("config/servers.example.yaml")

    if not config_path.exists():
        print_header("配置文件不存在")
        print_error("未找到 config/servers.yaml 配置文件")
        print()
        print("请先创建配置文件：")
        if example_path.exists():
            print_success(f"示例配置文件已存在: {example_path}")
            print()
            print("执行以下命令创建配置文件：")
            print(f"  cp {example_path} {config_path}")
            print()
            print("然后编辑 config/servers.yaml，填入你的 API Key：")
            print("  nano config/servers.yaml")
            print("  # 或")
            print("  vim config/servers.yaml")
        else:
            print_error("示例配置文件也不存在，请手动创建 config/servers.yaml")
        return False

    return True


async def create_default_user():
    """创建默认用户和API Key"""
    # 检查配置文件
    if not check_config_file():
        sys.exit(1)

    try:
        print_header("正在初始化数据库...")
        await init_db()
        print_success("数据库初始化完成")
    except Exception as e:
        print_error(f"数据库初始化失败: {e}")
        sys.exit(1)

    # 尝试加载配置
    try:
        settings = get_settings()
        print_success("配置文件加载成功")
    except Exception as e:
        print_error(f"配置文件加载失败: {e}")
        print()
        print("请检查 config/servers.yaml 文件格式是否正确")
        sys.exit(1)

    # 检查 API Key 配置
    if not settings.servers:
        print_error("未配置任何 LLM 服务器")
        print()
        print("请在 config/servers.yaml 中配置至少一个服务器：")
        print("""
servers:
  openai:
    base_url: https://api.openai.com/v1
    api_key: sk-your-api-key-here
    models:
      - gpt-4o-mini
""")
        sys.exit(1)

    # 创建用户和 API Key
    try:
        # 收集用户名
        print_header("创建默认用户")
        username = input("请输入用户名 (直接回车使用 'default_user'): ").strip()
        if not username:
            username = "default_user"

        async for session in get_db_session():
            # 创建用户
            user = await UserRepository.get_or_create(
                session,
                name=username,
                email=None
            )
            await session.commit()
            print_success(f"用户创建完成: {user.name} (ID: {user.id})")

            # 创建API Key
            api_key = await AuthService.create_api_key(
                session,
                user_id=user.id,
                name="Default API Key",
                daily_limit=10000,
                expires_days=365
            )
            await session.commit()
            print_success("API Key 创建完成")

            # 显示结果
            print_header("初始化完成！")
            print()
            print(f"用户ID:     {user.id}")
            print(f"用户名:     {user.name}")
            print()
            print("API Key (请妥善保管):")
            print(f"  {api_key.key}")
            print()

            # 显示服务器配置
            print("已配置的 LLM 服务器:")
            for name, server in settings.servers.items():
                print(f"  - {name}: {len(server.models)} 个模型")
                for model in server.models[:3]:  # 只显示前3个
                    print(f"      · {model}")
                if len(server.models) > 3:
                    print(f"      · ... 等 {len(server.models)} 个模型")

            # 显示记忆系统配置
            if settings.memory_search.enabled:
                print()
                print("记忆系统: RAG 向量检索")
                print(f"  - 嵌入服务: {settings.memory_search.provider}")
                print(f"  - 嵌入模型: {settings.memory_search.model}")
                print(f"  - 数据源: {', '.join(settings.memory_search.sources)}")
                print(f"  - 检索权重: 向量 {settings.memory_search.query.hybrid.vector_weight*100:.0f}% + "
                      f"关键词 {settings.memory_search.query.hybrid.text_weight*100:.0f}%")
                print()
                print("记忆文件位置:")
                print(f"  - 用户目录: ~/.javis/memory/{user.id}/")
                print("     - conversations.md  (对话记录)")
                print("     - memory.sqlite     (向量数据库)")
                print("  - 会话记录: ~/.ai-agent/sessions/")

            # 使用示例
            print()
            print("=" * 50)
            print("快速开始")
            print("=" * 50)
            print()
            print("1. 启动服务:")
            print("   python run.py")
            print()
            print("2. 测试聊天:")
            print(f'   curl -X POST "http://localhost:8000/v1/chat/completions" \\')
            print(f'     -H "Content-Type: application/json" \\')
            print(f'     -H "Authorization: Bearer {api_key.key}" \\')
            print(f'     -d \'{{"model":"{list(settings.servers.values())[0].models[0]}",')
            print(f'          "messages":[{{"role":"user","content":"你好"}}]}}\'')
            print()
            print("3. 访问 API 文档:")
            print("   http://localhost:8000/docs")
            print()

            break  # 只需要一次

    except Exception as e:
        print_error(f"创建用户失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await close_db()


def main():
    """主函数"""
    try:
        asyncio.run(create_default_user())
    except KeyboardInterrupt:
        print()
        print_error("操作已取消")
        sys.exit(1)
    except Exception as e:
        print()
        print_error(f"发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
