"""Javis 启动脚本

启动 FastAPI 开发服务器，支持自动重载。
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn


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
    GRAY = '\033[90m'

    DISABLED = False


def c(color: str, text: str) -> str:
    """为文本添加颜色"""
    if Colors.DISABLED or not sys.stdout.isatty():
        return text
    return f"{color}{text}{Colors.ENDC}"


def print_banner():
    """打印启动横幅"""
    print()
    print(c(Colors.OKCYAN, "╔═══════════════════════════════════════════════════════════════════╗"))
    print(c(Colors.OKCYAN, "║                                                                   ║"))
    print(c(Colors.OKCYAN, "║") + c(Colors.BOLD, "              Javis - 带 RAG 记忆的个人 AI 助手系统              ") + c(Colors.OKCYAN, "              ║"))
    print(c(Colors.OKCYAN, "║") + c(Colors.OKGREEN, "           RAG 向量检索记忆 + LLM API 代理                      ") + c(Colors.OKCYAN, "                    ║"))
    print(c(Colors.OKCYAN, "║                                                                   ║"))
    print(c(Colors.OKCYAN, "╚═══════════════════════════════════════════════════════════════════╝"))
    print()


def print_separator(char="─", length=69):
    """打印分隔线"""
    print(c(Colors.GRAY, char * length))


def check_config() -> bool:
    """检查配置文件是否存在"""
    config_path = Path("config/servers.yaml")

    if not config_path.exists():
        print()
        print(c(Colors.FAIL, "╔═══════════════════════════════════════════════════════════════════╗"))
        print(c(Colors.FAIL, "║                       ⚠ 配置文件不存在                              ║"))
        print(c(Colors.FAIL, "╚═══════════════════════════════════════════════════════════════════╝"))
        print()
        print(c(Colors.WARNING, "未找到 config/servers.yaml 配置文件"))
        print()
        print(c(Colors.OKGREEN, "请先执行以下步骤:"))
        print()
        print(c(Colors.OKBLUE, "  1️⃣  运行自动化配置:"))
        print(c(Colors.GRAY, "     python setup.py"))
        print()
        print(c(Colors.OKBLUE, "  2️⃣  或手动配置:"))
        print(c(Colors.GRAY, "     cp config/servers.example.yaml config/servers.yaml"))
        print(c(Colors.GRAY, "     nano config/servers.yaml  # 填入 API Key"))
        print()
        print(c(Colors.OKBLUE, "  3️⃣  初始化数据库:"))
        print(c(Colors.GRAY, "     python init_db.py"))
        print()
        return False

    return True


def check_database() -> bool:
    """检查数据库是否已初始化"""
    import sqlite3
    db_path = Path("javis.db")

    if not db_path.exists():
        print()
        print(c(Colors.WARNING, "╔═══════════════════════════════════════════════════════════════════╗"))
        print(c(Colors.WARNING, "║                       ⚠ 数据库未初始化                              ║"))
        print(c(Colors.WARNING, "╚═══════════════════════════════════════════════════════════════════╝"))
        print()
        print(c(Colors.WARNING, "未找到 javis.db 数据库文件"))
        print()
        print(c(Colors.OKGREEN, "请先运行数据库初始化:"))
        print()
        print(c(Colors.OKBLUE, "  python init_db.py"))
        print()
        print(c(Colors.GRAY, "或使用自动化配置（包含数据库初始化）:"))
        print(c(Colors.OKBLUE, "  python setup.py"))
        print()
        return False

    # 检查表是否存在
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        required_tables = {"users", "conversations", "messages", "api_keys"}
        missing_tables = required_tables - tables

        if missing_tables:
            print()
            print(c(Colors.WARNING, "╔═══════════════════════════════════════════════════════════════════╗"))
            print(c(Colors.WARNING, "║                    ⚠ 数据库表缺失                                ║"))
            print(c(Colors.WARNING, "╚═══════════════════════════════════════════════════════════════════╝"))
            print()
            print(c(Colors.WARNING, f"数据库缺少以下表: {', '.join(missing_tables)}"))
            print()
            print(c(Colors.OKGREEN, "请重新初始化数据库:"))
            print()
            print(c(Colors.OKBLUE, "  python init_db.py"))
            print()
            return False

    except sqlite3.Error as e:
        print()
        print(c(Colors.WARNING, "╔═══════════════════════════════════════════════════════════════════╗"))
        print(c(Colors.WARNING, "║                    ⚠ 数据库检查失败                                ║"))
        print(c(Colors.WARNING, "╚═══════════════════════════════════════════════════════════════════╝"))
        print()
        print(c(Colors.WARNING, f"错误: {e}"))
        print()
        print(c(Colors.OKGREEN, "请重新初始化数据库:"))
        print()
        print(c(Colors.OKBLUE, "  python init_db.py"))
        print()
        return False

    return True


def print_config_info(settings):
    """打印配置信息"""
    print()
    print(c(Colors.OKGREEN, "✓ 配置加载成功"))
    print()
    print(c(Colors.BOLD, "  服务配置:"))
    print(c(Colors.GRAY, f"    LLM 服务器: {c(Colors.OKBLUE, str(len(settings.servers)))} 个"))

    for name, server in settings.servers.items():
        models_display = ', '.join(server.models[:2])
        if len(server.models) > 2:
            models_display += f" +{len(server.models) - 2} 更多"
        print(c(Colors.GRAY, f"    · {c(Colors.OKCYAN, name)}: {c(Colors.OKBLUE, models_display)}"))

    print()
    if settings.memory_search.enabled:
        print(c(Colors.GRAY, f"    记忆系统: {c(Colors.OKGREEN, '启用 ')}({c(Colors.OKCYAN, settings.memory_search.provider)})"))
        print(c(Colors.GRAY, f"    嵌入模型: {c(Colors.OKBLUE, settings.memory_search.model)}"))
    else:
        print(c(Colors.GRAY, f"    记忆系统: {c(Colors.WARNING, '禁用')}"))
    print()


def print_server_start():
    """打印服务器启动信息"""
    print()
    print(c(Colors.OKGREEN, "◉ 正在启动服务器..."))
    print()
    print(c(Colors.GRAY, "─" * 69))
    print(c(Colors.GRAY, "  " + c(Colors.BOLD, "服务信息")))
    print(c(Colors.GRAY, "─" * 69))
    print()
    print(c(Colors.GRAY, f"  地址: {c(Colors.OKBLUE, 'http://0.0.0.0:8000')}"))
    print(c(Colors.GRAY, f"  本地: {c(Colors.OKBLUE, 'http://localhost:8000')}"))
    print(c(Colors.GRAY, f"  文档: {c(Colors.OKBLUE, 'http://localhost:8000/docs')}"))
    print()
    print(c(Colors.GRAY, "─" * 69))
    print(c(Colors.GRAY, "  " + c(Colors.BOLD, "开发模式")))
    print(c(Colors.GRAY, "─" * 69))
    print()
    print(c(Colors.GRAY, "  • 自动重载: ") + c(Colors.OKGREEN, "启用"))
    log_level_str = c(Colors.GRAY, "  • 日志级别: ") + c(Colors.OKBLUE, "INFO")
    print(log_level_str)
    print()
    print(c(Colors.OKGREEN, "✓ 服务器已启动，按 Ctrl+C 停止"))
    print()
    print(c(Colors.GRAY, "─" * 69))
    print()


def main():
    """主函数"""
    print_banner()

    # 检测 Windows 环境
    if sys.platform == "win32":
        try:
            import colorama
            colorama.init()
        except ImportError:
            Colors.DISABLED = True

    # 检查配置文件
    if not check_config():
        sys.exit(1)

    # 检查数据库
    if not check_database():
        sys.exit(1)

    # 加载配置
    try:
        from config.settings import get_settings
        settings = get_settings()
        print_config_info(settings)
    except Exception as e:
        print()
        print(c(Colors.FAIL, f"✗ 配置加载失败: {e}"))
        print()
        print(c(Colors.WARNING, "请检查 config/servers.yaml 文件格式是否正确"))
        sys.exit(1)

    # 打印启动信息
    print_server_start()

    # 启动服务器
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="warning",  # 降低默认日志级别
    )


if __name__ == "__main__":
    main()
