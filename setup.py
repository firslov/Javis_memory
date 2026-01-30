#!/usr/bin/env python3
"""Javis 自动化配置脚本

交互式引导用户完成安装和配置。
"""
import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


# ANSI 颜色代码
class Colors:
    """终端颜色"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    # 禁用颜色（在非终端环境）
    DISABLED = False


def disable_colors():
    """禁用颜色输出"""
    Colors.DISABLED = True


def c(color: str, text: str) -> str:
    """为文本添加颜色"""
    if Colors.DISABLED or not sys.stdout.isatty():
        return text
    return f"{color}{text}{Colors.ENDC}"


def print_banner():
    """打印横幅"""
    print()
    print(c(Colors.OKCYAN, "╔═══════════════════════════════════════════════════════════════════╗"))
    print(c(Colors.OKCYAN, "║                                                                   ║"))
    print(c(Colors.OKCYAN, "║") + c(Colors.BOLD, "              Javis - 带 RAG 记忆的个人 AI 助手系统              ") + c(Colors.OKCYAN, "              ║"))
    print(c(Colors.OKCYAN, "║") + c(Colors.OKGREEN, "                    自动化配置向导                              ") + c(Colors.OKCYAN, "                    ║"))
    print(c(Colors.OKCYAN, "║                                                                   ║"))
    print(c(Colors.OKCYAN, "╚═══════════════════════════════════════════════════════════════════╝"))
    print()


def print_step(num: int, total: int, title: str):
    """打印步骤标题"""
    print()
    print(c(Colors.BOLD, f"┌─ 步骤 {num}/{total}: {title}"))
    print(c(Colors.OKCYAN, "│") + " " + "─" * 60)
    print(c(Colors.OKCYAN, "│"))


def print_step_end():
    """结束步骤显示"""
    print(c(Colors.OKCYAN, "│"))
    print(c(Colors.OKCYAN, "└") + " " + "─" * 62)
    print()


def print_success(text: str):
    """打印成功信息"""
    print(c(Colors.OKCYAN, "│ ") + c(Colors.OKGREEN, "✓") + f" {text}")


def print_error(text: str):
    """打印错误信息"""
    print(c(Colors.OKCYAN, "│ ") + c(Colors.FAIL, "✗") + f" {text}")


def print_info(text: str):
    """打印提示信息"""
    print(c(Colors.OKCYAN, "│ ") + f"  {text}")


def print_input(prompt: str):
    """打印输入提示"""
    return input(c(Colors.OKCYAN, "│ ") + c(Colors.WARNING, "▶") + f" {prompt}")


def print_option(num: int, text: str):
    """打印选项"""
    print(c(Colors.OKCYAN, f"│   {num}. ") + text)


def print_header(text: str):
    """打印小标题"""
    print()
    print(c(Colors.OKCYAN, "│ ") + c(Colors.BOLD, text))


def print_box(title: str, content: list, color=Colors.OKGREEN):
    """打印信息框"""
    width = 60
    print()
    print(c(Colors.OKCYAN, "│") + "┌" + "─" * (width - 2) + "┐")
    print(c(Colors.OKCYAN, "│") + c(color, f"│ {title:^{width - 4}} │"))
    print(c(Colors.OKCYAN, "│") + "├" + "─" * (width - 2) + "┤")
    for line in content:
        print(c(Colors.OKCYAN, "│") + f"│ {line:<{width - 4}} │")
    print(c(Colors.OKCYAN, "│") + "└" + "─" * (width - 2) + "┘")
    print()


def show_progress(text: str, done=False):
    """显示进度指示"""
    if not done:
        print(c(Colors.OKCYAN, "│ ") + c(Colors.OKBLUE, "◐") + f" {text}...", end="\r")
    else:
        print(c(Colors.OKCYAN, "│ ") + c(Colors.OKGREEN, "◉") + f" {text}")


def check_python_version() -> bool:
    """检查 Python 版本"""
    total_steps = 6
    print_step(1, total_steps, "检查 Python 版本")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"当前 Python 版本: {c(Colors.BOLD, version_str)}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_error("Python 版本过低，需要 3.10 或更高版本")
        print_info("请升级 Python 后重试")
        print_step_end()
        return False

    print_success(f"Python {version_str} 符合要求")
    print_step_end()
    return True


def check_config_exists() -> bool:
    """检查配置文件是否已存在"""
    config_path = Path("config/servers.yaml")

    if config_path.exists():
        print()
        print(c(Colors.WARNING, "⚠ 检测到已存在配置文件 config/servers.yaml"))
        response = input(c(Colors.OKCYAN, "│ ") + "是否重新配置？" + c(Colors.WARNING, "(y/N)") + ": ").strip().lower()
        return response == 'y'

    return True


def install_dependencies():
    """安装依赖"""
    total_steps = 6
    print_step(2, total_steps, "安装依赖包")

    show_progress("正在安装依赖包")
    print()

    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print_error("依赖安装失败")
        if result.stderr:
            print(c(Colors.OKCYAN, "│ ") + result.stderr)
        print_step_end()
        return False

    print_success("依赖包安装完成")
    print_step_end()
    return True


def collect_llm_config() -> dict:
    """收集 LLM 配置"""
    total_steps = 6
    print_step(3, total_steps, "配置 LLM 服务器")

    print_header("请选择你的 LLM 服务提供商:")
    print_option("1", "OpenAI (GPT-4, GPT-4o)")
    print_option("2", "DeepSeek (深度求索)")
    print_option("3", "豆包 (字节跳动)")
    print_option("4", "通义千问 (阿里云)")
    print_option("5", "智谱 AI (ChatGLM)")
    print_option("6", "本地 Ollama")
    print_option("7", "自定义")

    choice = print_input("请输入选项 (1-7): ").strip()

    configs = {
        "1": {
            "name": "openai",
            "base_url": "https://api.openai.com/v1",
            "models": ["gpt-4o-mini", "gpt-4o"],
            "api_key_prompt": "请输入 OpenAI API Key (sk-...): ",
            "emoji": "🤖"
        },
        "2": {
            "name": "deepseek",
            "base_url": "https://api.deepseek.com/v1",
            "models": ["deepseek-chat", "deepseek-reasoner"],
            "api_key_prompt": "请输入 DeepSeek API Key (sk-...): ",
            "emoji": "🔍"
        },
        "3": {
            "name": "doubao",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "models": ["doubao-pro-4k", "doubao-pro-32k"],
            "api_key_prompt": "请输入豆包 API Key: ",
            "emoji": "🫘"
        },
        "4": {
            "name": "qwen",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "models": ["qwen-turbo", "qwen-plus"],
            "api_key_prompt": "请输入通义千问 API Key (sk-...): ",
            "emoji": "☁️"
        },
        "5": {
            "name": "zhipu",
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "models": ["glm-4-flash", "glm-4-plus"],
            "api_key_prompt": "请输入智谱 AI API Key: ",
            "emoji": "🧠"
        },
        "6": {
            "name": "ollama",
            "base_url": "http://localhost:11434/v1",
            "models": ["llama3", "qwen2"],
            "api_key_prompt": "Ollama 通常不需要 API Key，直接回车: ",
            "api_key_default": "ollama",
            "emoji": "🦙"
        },
        "7": {
            "name": "custom",
            "base_url": "",
            "models": [],
            "custom": True,
            "emoji": "⚙️"
        }
    }

    config = configs.get(choice, configs["1"])

    # 自定义配置
    if config.get("custom"):
        print_header("请输入自定义配置:")
        base_url = print_input("API Base URL: ").strip()
        api_key = print_input("API Key: ").strip()
        models_input = print_input("模型列表 (用逗号分隔): ").strip()
        models = [m.strip() for m in models_input.split(",")]
        name = "custom"
        emoji = "⚙️"
    else:
        base_url = config["base_url"]
        api_key = print_input(config["api_key_prompt"]).strip()
        if not api_key and config.get("api_key_default"):
            api_key = config["api_key_default"]
        models = config["models"]
        name = config["name"]
        emoji = config["emoji"]

    print()
    print_success(f"{emoji} 已配置: {c(Colors.BOLD, name)}")
    print_info(f"  API 地址: {c(Colors.OKBLUE, base_url)}")
    print_info(f"  模型: {c(Colors.OKBLUE, ', '.join(models))}")

    print_step_end()
    return {
        "name": name,
        "base_url": base_url,
        "api_key": api_key,
        "models": models
    }


def collect_embedding_config(llm_config: dict) -> dict:
    """收集嵌入服务配置"""
    total_steps = 6
    print_step(4, total_steps, "配置记忆系统")

    print_header("记忆系统需要将文本转换为向量，请配置嵌入服务:")
    print_option("1", "使用与 LLM 相同的 API (推荐)")
    print_option("2", "使用不同的 API")
    print_option("3", c(Colors.FAIL, "禁用记忆系统"))

    choice = print_input("请输入选项 (1-3): ").strip()

    if choice == "3":
        print_info(c(Colors.WARNING, "记忆系统已禁用，AI 将无法记住对话内容"))
        print_step_end()
        return {"enabled": False}

    if choice == "1":
        # 使用相同的 API
        provider_map = {
            "openai": "openai",
            "deepseek": "openai",
            "doubao": "openai",
            "qwen": "openai",
            "zhipu": "openai",
            "ollama": "local",
            "custom": "openai"
        }
        provider = provider_map.get(llm_config["name"], "openai")
        api_key = llm_config["api_key"]
        base_url = llm_config["base_url"].replace("/v1", "").replace("/chat", "")

        # 选择模型
        if provider == "openai":
            print_header("选择嵌入模型:")
            print_option("1", "text-embedding-3-small" + c(Colors.OKGREEN, " (推荐，快速)"))
            print_option("2", "text-embedding-3-large" + c(Colors.OKBLUE, " (更高精度)"))
            print_info("或者直接输入自定义模型名称 (如: doubao-embedding)")
            model_choice = print_input("请选择 (1-3) 或输入模型名称: ").strip()

            if model_choice == "2":
                model = "text-embedding-3-large"
            elif model_choice == "3":
                model = print_input("请输入嵌入模型名称: ").strip()
                while not model:
                    print_error("模型名称不能为空")
                    model = print_input("请输入嵌入模型名称: ").strip()
            elif model_choice == "1" or not model_choice:
                model = "text-embedding-3-small"
            else:
                # 用户直接输入了模型名称
                model = model_choice
        else:
            print_info(f"默认使用 LLM 模型作为嵌入模型")
            model = llm_config["models"][0] if llm_config["models"] else "embedding"

            # 也允许自定义
            custom = print_input(f"使用默认模型 [{model}]？直接回车确认，或输入自定义: ").strip()
            if custom:
                model = custom

    else:
        # 使用不同的 API
        print_header("请输入嵌入服务配置:")
        provider = print_input("提供商 (openai/gemini/local): ").strip() or "openai"
        api_key = print_input("API Key: ").strip()
        base_url = print_input("Base URL (可选，回车跳过): ").strip() or ""

        # 选择模型
        print_header("选择嵌入模型:")
        print_option("1", "text-embedding-3-small")
        print_option("2", "text-embedding-ada-002")
        print_option("3", "自定义模型")
        model_choice = print_input("请选择 (1-3): ").strip()

        if model_choice == "2":
            model = "text-embedding-ada-002"
        elif model_choice == "3":
            model = print_input("请输入嵌入模型名称: ").strip()
            while not model:
                print_error("模型名称不能为空")
                model = print_input("请输入嵌入模型名称: ").strip()
        else:
            model = "text-embedding-3-small"

    print()
    print_success("记忆系统配置完成")
    print_info(f"  提供商: {c(Colors.OKBLUE, provider)}")
    print_info(f"  模型: {c(Colors.OKBLUE, model)}")

    print_step_end()
    return {
        "enabled": True,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url
    }


def create_config_file(llm_config: dict, embedding_config: dict):
    """创建配置文件"""
    total_steps = 6
    print_step(5, total_steps, "创建配置文件")

    show_progress("正在生成配置文件")
    print()

    config_content = f"""# ============================================
# Javis 配置文件 - 自动生成
# ============================================

# --- LLM 服务器配置 ---
servers:
  {llm_config['name']}:
    base_url: {llm_config['base_url']}
    api_key: {llm_config['api_key']}
    models:
"""

    for model in llm_config['models']:
        config_content += f"      - {model}\n"

    config_content += f"""
# --- 记忆搜索配置 (RAG 向量检索) ---
memory_search:
  enabled: {str(embedding_config['enabled']).lower()}
  sources:
    - memory
    - sessions
  extra_paths: []
  memory_files_dir: "~/.javis/memory/{{user_id}}"  # 用户记忆文件存储目录

  # Embedding Provider
  provider: {embedding_config.get('provider', 'openai')}
  model: {embedding_config.get('model', 'text-embedding-3-small')}
  fallback: none

  # 远程 API 配置
  remote:
    api_key: {embedding_config.get('api_key', llm_config['api_key'])}
"""

    if embedding_config.get('base_url'):
        config_content += f"    base_url: {embedding_config['base_url']}\n"
    else:
        config_content += f"    base_url: {llm_config['base_url'].replace('/v1', '')}\n"

    config_content += """    gemini_api_key: ""
    batch:
      enabled: true
      wait: true
      concurrency: 2
      timeout_minutes: 60

  # 本地向量化配置
  local:
    model_path: ""
    model_cache_dir: ""
    device: cpu

  # 存储配置
  store:
    path: "~/.javis/memory/{user_id}/memory.sqlite"
    vector:
      enabled: true
      extension_path: ""

  # 分块配置
  chunking:
    tokens: 400
    overlap: 80

  # 同步配置
  sync:
    on_session_start: true
    on_search: false
    watch: true
    watch_debounce_ms: 1500
    interval_minutes: 0

  # 检索配置
  query:
    max_results: 6
    min_score: 0.5
    hybrid:
      enabled: true
      vector_weight: 0.7
      text_weight: 0.3
      candidate_multiplier: 4

  # 缓存配置
  cache:
    enabled: true
    max_entries: 10000

# --- 数据库 ---
database:
  url: sqlite+aiosqlite:///./javis.db

# --- 缓存 ---
cache:
  enabled: true

# --- 日志 ---
logging:
  level: INFO
  slow_request_threshold: 5.0
"""

    config_path = Path("config/servers.yaml")
    config_path.write_text(config_content)

    print_success(f"配置文件已创建: {c(Colors.OKBLUE, config_path)}")
    print_step_end()


async def init_database():
    """初始化数据库"""
    total_steps = 6
    print_step(6, total_steps, "初始化数据库")

    show_progress("正在初始化数据库")
    print()

    try:
        from database.session import init_db, get_db_session, close_db
        from database.repository import UserRepository
        from services.auth import AuthService

        await init_db()
        print_success("数据库初始化完成")

        # 收集用户名
        print_header("创建默认用户")
        username = print_input("请输入用户名 (直接回车使用 'default_user'): ").strip()
        if not username:
            username = "default_user"

        async for session in get_db_session():
            user = await UserRepository.get_or_create(
                session,
                name=username,
                email=None
            )
            await session.commit()

            api_key = await AuthService.create_api_key(
                session,
                user_id=user.id,
                name="Default API Key",
                daily_limit=10000,
                expires_days=365
            )
            await session.commit()

            print_success(f"用户 '{username}' 和 API Key 创建完成")
            break

        await close_db()

        print_step_end()

        # 显示完成信息
        print_box("初始化完成！", [
            f"用户ID: {user.id}",
            f"用户名: {user.name}",
            f"记忆目录: ~/.javis/memory/{user.id}/",
            "",
            "API Key (请妥善保管):",
            f"{c(Colors.BOLD, c(Colors.OKGREEN, api_key.key))}"
        ])

        return api_key.key

    except Exception as e:
        print_error(f"数据库初始化失败: {e}")
        import traceback
        traceback.print_exc()
        print_step_end()
        return None


def print_completion_guide(api_key: str):
    """打印完成后的使用指南"""
    print()
    print(c(Colors.OKCYAN, "╔═══════════════════════════════════════════════════════════════════╗"))
    print(c(Colors.OKCYAN, "║") + c(Colors.BOLD, "                    🎉 配置完成！下一步                              ") + c(Colors.OKCYAN, "                    ║"))
    print(c(Colors.OKCYAN, "╚═══════════════════════════════════════════════════════════════════╝"))
    print()
    print(c(Colors.OKGREEN, "1️⃣  启动服务:"))
    print("   " + c(Colors.OKBLUE, "python run.py"))
    print()
    print(c(Colors.OKGREEN, "2️⃣  测试聊天:"))
    print(f"   curl -X POST http://localhost:8000/v1/chat/completions \\")
    print(f"     -H \"Content-Type: application/json\" \\")
    print(f"     -H \"Authorization: Bearer {c(Colors.OKGREEN, api_key[:20])}...\" \\")
    print(f"     -d '{{\"model\":\"gpt-4o-mini\",\"messages\":[{{\"role\":\"user\",\"content\":\"你好\"}}]}}'")
    print()
    print(c(Colors.OKGREEN, "3️⃣  访问 API 文档:"))
    print("   " + c(Colors.OKBLUE, "http://localhost:8000/docs"))
    print()
    print(c(Colors.OKCYAN, "─" * 69))
    print()


def main():
    """主函数"""
    print_banner()

    # 检测 Windows 环境，禁用颜色
    if sys.platform == "win32":
        try:
            import colorama
            colorama.init()
        except ImportError:
            disable_colors()

    try:
        # 检查 Python 版本
        if not check_python_version():
            sys.exit(1)

        # 检查是否需要重新配置
        if not check_config_exists():
            print("配置已存在，如需重新配置请删除 config/servers.yaml")
            print()
            print("继续使用现有配置进行初始化？(y/N): ", end="")
            if input().strip().lower() != 'y':
                print("配置已取消")
                sys.exit(0)

            # 直接初始化数据库
            api_key = asyncio.run(init_database())
            if api_key:
                print_completion_guide(api_key)
            sys.exit(0)

        # 安装依赖
        if not install_dependencies():
            sys.exit(1)

        # 收集配置
        llm_config = collect_llm_config()
        embedding_config = collect_embedding_config(llm_config)

        # 创建配置文件
        create_config_file(llm_config, embedding_config)

        # 初始化数据库
        api_key = asyncio.run(init_database())

        if api_key:
            print_completion_guide(api_key)

        print(c(Colors.OKGREEN, "✓ 配置完成！现在可以运行 '") + c(Colors.OKBLUE, "python run.py") + c(Colors.OKGREEN, "' 启动服务"))
        print()

    except KeyboardInterrupt:
        print()
        print()
        print(c(Colors.WARNING, "⚠ 配置已取消"))
        sys.exit(1)
    except Exception as e:
        print()
        print(c(Colors.FAIL, f"✗ 配置过程中发生错误: {e}"))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
