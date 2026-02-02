"""记忆数据可视化模块

提供CLI终端形式的记忆数据可视化展示。
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from config.logging import get_logger

logger = get_logger(__name__)


# ANSI颜色代码
class Colors:
    """终端颜色"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"

    # 层级颜色
    WORKING = "\033[33m"      # 黄色
    LONG_TERM = "\033[32m"    # 绿色
    ARCHIVE = "\033[90m"      # 灰色


def print_header(title: str, width: int = 80):
    """打印标题头"""
    print("\n" + Colors.BOLD + Colors.CYAN)
    print("═" * width)
    print(f"  {title}")
    print("═" * width)
    print(Colors.RESET)


def print_section(title: str):
    """打印小节标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {title}{Colors.RESET}")


def print_table(headers: List[str], rows: List[List[str]],
                col_widths: Optional[List[int]] = None,
                align: Optional[List[str]] = None):
    """打印表格

    Args:
        headers: 表头列表
        rows: 数据行列表
        col_widths: 列宽列表（可选）
        align: 对齐方式列表 ('left', 'center', 'right')
    """
    if not rows:
        print(f"{Colors.DIM}  (无数据){Colors.RESET}")
        return

    # 计算列宽
    if col_widths is None:
        col_widths = []
        for i, header in enumerate(headers):
            max_len = len(header)
            for row in rows:
                if i < len(row):
                    max_len = max(max_len, len(str(row[i])))
            col_widths.append(max_len + 2)

    # 默认对齐
    if align is None:
        align = ['left'] * len(headers)

    # 打印表头
    header_line = Colors.BOLD + Colors.CYAN
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        if align[i] == 'center':
            header_line += header.center(width)
        elif align[i] == 'right':
            header_line += header.rjust(width)
        else:
            header_line += header.ljust(width)
    header_line += Colors.RESET
    print(header_line)

    # 打印分隔线
    sep_line = Colors.DIM
    for width in col_widths:
        sep_line += "─" * width
    sep_line += Colors.RESET
    print(sep_line)

    # 打印数据行
    for row in rows:
        line = ""
        for i, (cell, width) in enumerate(zip(row, col_widths)):
            cell_str = str(cell)[:width-2]  # 限制长度
            if align[i] == 'center':
                line += cell_str.center(width)
            elif align[i] == 'right':
                line += cell_str.rjust(width)
            else:
                line += cell_str.ljust(width)
        print(line)


def print_progress_bar(label: str, value: float, max_value: float = 1.0,
                       width: int = 30, color: str = Colors.GREEN):
    """打印进度条

    Args:
        label: 标签
        value: 当前值
        max_value: 最大值
        width: 进度条宽度
        color: 颜色
    """
    ratio = min(1.0, value / max_value) if max_value > 0 else 0
    filled = int(ratio * width)
    bar = "█" * filled + "░" * (width - filled)

    percentage = ratio * 100
    print(f"  {label}: {color}{bar}{Colors.RESET} {percentage:.1f}%")


def print_tier_badge(tier: str) -> str:
    """打印层级徽章"""
    colors = {
        "working": Colors.WORKING,
        "long_term": Colors.LONG_TERM,
        "archive": Colors.ARCHIVE,
    }
    labels = {
        "working": "工作记忆",
        "long_term": "长期记忆",
        "archive": "档案",
    }

    color = colors.get(tier, Colors.WHITE)
    label = labels.get(tier, tier)

    return f"{color}[{label}]{Colors.RESET}"


def print_importance_score(score: float) -> str:
    """打印重要性分数（带颜色）"""
    if score >= 0.7:
        return f"{Colors.GREEN}●{Colors.RESET} {score:.2f}"
    elif score >= 0.4:
        return f"{Colors.YELLOW}●{Colors.RESET} {score:.2f}"
    else:
        return f"{Colors.RED}○{Colors.RESET} {score:.2f}"


def print_ascii_chart(data: Dict[str, int], width: int = 50, height: int = 10):
    """打印ASCII柱状图

    Args:
        data: 标签->值的字典
        width: 图表宽度
        height: 图表高度
    """
    if not data:
        print(f"{Colors.DIM}  (无数据){Colors.RESET}")
        return

    max_value = max(data.values()) if data.values() else 1
    if max_value == 0:
        max_value = 1

    labels = list(data.keys())
    values = list(data.values())

    # 打印标题
    print(f"\n  {Colors.BOLD}分布统计{Colors.RESET}\n")

    # 计算每列宽度
    col_width = width // len(data)

    # 打印柱状图
    for level in range(height, 0, -1):
        line = "  "
        threshold = (max_value / height) * level

        for value in values:
            if value >= threshold:
                # 根据相对高度选择字符
                rel_height = value / max_value
                if rel_height > 0.8:
                    line += f"{Colors.GREEN}█{Colors.RESET}"
                elif rel_height > 0.5:
                    line += f"{Colors.YELLOW}▓{Colors.RESET}"
                else:
                    line += f"{Colors.BLUE}░{Colors.RESET}"
            else:
                line += " "

            # 填充到列宽
            line += " " * (col_width - 1)

        print(line)

    # 打印标签
    label_line = "  "
    for label in labels:
        label_short = label[:col_width-1]
        label_line += label_short.center(col_width-1) + " "
    print(label_line)

    # 打印数值
    value_line = "  "
    for value in values:
        value_line += str(value).center(col_width-1) + " "
    print(f"{Colors.DIM}{value_line}{Colors.RESET}")


@dataclass
class MemoryVisualizationData:
    """记忆可视化数据"""
    user_id: int
    tier_stats: Dict[str, int]
    total_chunks: int
    avg_importance: float
    total_accesses: int

    # 详细数据
    top_memories: List[Dict[str, Any]] = None
    recent_accesses: List[Dict[str, Any]] = None
    importance_distribution: Dict[str, int] = None
    tier_distribution: Dict[str, int] = None


class MemoryVisualizer:
    """记忆数据可视化器"""

    def __init__(self, db_conn):
        """初始化可视化器

        Args:
            db_conn: 数据库连接
        """
        self.db = db_conn

    async def collect_data(self, user_id: int) -> MemoryVisualizationData:
        """收集可视化数据

        Args:
            user_id: 用户ID

        Returns:
            MemoryVisualizationData 可视化数据
        """
        # 获取层级统计
        cursor = await self.db.execute("""
            SELECT memory_tier, COUNT(*) as count
            FROM memory_chunks_meta
            WHERE user_id = ?
            GROUP BY memory_tier
        """, (user_id,))
        tier_rows = await cursor.fetchall()
        await cursor.close()

        tier_stats = {row[0]: row[1] for row in tier_rows}

        # 获取总体统计
        cursor = await self.db.execute("""
            SELECT
                COUNT(*) as total,
                AVG(importance_score) as avg_importance,
                SUM(access_count) as total_accesses
            FROM memory_chunks_meta
            WHERE user_id = ?
        """, (user_id,))
        row = await cursor.fetchone()
        await cursor.close()

        total_chunks = row[0] if row else 0
        avg_importance = float(row[1]) if row and row[1] else 0.0
        total_accesses = row[2] if row else 0

        # 获取Top记忆（按重要性）
        cursor = await self.db.execute("""
            SELECT m.chunk_id, m.importance_score, m.memory_tier,
                   m.access_count, c.text
            FROM memory_chunks_meta m
            JOIN memory_chunks c ON m.chunk_id = c.id AND m.user_id = c.user_id
            WHERE m.user_id = ?
            ORDER BY m.importance_score DESC
            LIMIT 10
        """, (user_id,))
        top_rows = await cursor.fetchall()
        await cursor.close()

        top_memories = []
        for r in top_rows:
            top_memories.append({
                "chunk_id": r[0][:12] + "...",
                "importance": r[1],
                "tier": r[2],
                "access_count": r[3],
                "text": r[4][:50] + "..." if len(r[4]) > 50 else r[4],
            })

        # 获取最近访问的记忆
        cursor = await self.db.execute("""
            SELECT m.chunk_id, m.last_accessed_at, m.access_count,
                   m.memory_tier, c.text
            FROM memory_chunks_meta m
            JOIN memory_chunks c ON m.chunk_id = c.id AND m.user_id = c.user_id
            WHERE m.user_id = ? AND m.last_accessed_at IS NOT NULL
            ORDER BY m.last_accessed_at DESC
            LIMIT 10
        """, (user_id,))
        recent_rows = await cursor.fetchall()
        await cursor.close()

        recent_accesses = []
        for r in recent_rows:
            last_accessed = r[1]
            time_ago = ""
            if last_accessed:
                try:
                    accessed_at = datetime.fromisoformat(last_accessed)
                    delta = datetime.now() - accessed_at
                    if delta.days > 0:
                        time_ago = f"{delta.days}天前"
                    elif delta.seconds >= 3600:
                        hours = delta.seconds // 3600
                        time_ago = f"{hours}小时前"
                    else:
                        minutes = delta.seconds // 60
                        time_ago = f"{minutes}分钟前"
                except:
                    time_ago = "未知"

            recent_accesses.append({
                "chunk_id": r[0][:12] + "...",
                "time_ago": time_ago,
                "access_count": r[2],
                "tier": r[3],
                "text": r[4][:40] + "..." if len(r[4]) > 40 else r[4],
            })

        # 重要性分布
        importance_distribution = {
            "高 (≥0.7)": 0,
            "中 (0.4-0.7)": 0,
            "低 (<0.4)": 0,
        }

        cursor = await self.db.execute("""
            SELECT
                SUM(CASE WHEN importance_score >= 0.7 THEN 1 ELSE 0 END) as high,
                SUM(CASE WHEN importance_score >= 0.4 AND importance_score < 0.7 THEN 1 ELSE 0 END) as medium,
                SUM(CASE WHEN importance_score < 0.4 THEN 1 ELSE 0 END) as low
            FROM memory_chunks_meta
            WHERE user_id = ?
        """, (user_id,))
        dist_row = await cursor.fetchone()
        await cursor.close()

        if dist_row:
            importance_distribution["高 (≥0.7)"] = dist_row[0] or 0
            importance_distribution["中 (0.4-0.7)"] = dist_row[1] or 0
            importance_distribution["低 (<0.4)"] = dist_row[2] or 0

        return MemoryVisualizationData(
            user_id=user_id,
            tier_stats=tier_stats,
            total_chunks=total_chunks,
            avg_importance=avg_importance,
            total_accesses=total_accesses,
            top_memories=top_memories,
            recent_accesses=recent_accesses,
            importance_distribution=importance_distribution,
            tier_distribution=tier_stats,
        )

    def display_overview(self, data: MemoryVisualizationData):
        """显示概览

        Args:
            data: 可视化数据
        """
        print_header(f"记忆系统概览 - 用户 {data.user_id}", 70)

        # 关键指标
        print(f"\n  {Colors.BOLD}关键指标{Colors.RESET}\n")
        print(f"    总记忆数: {Colors.CYAN}{data.total_chunks}{Colors.RESET}")
        print(f"    平均重要性: {print_importance_score(data.avg_importance)}")
        print(f"    总访问次数: {Colors.CYAN}{data.total_accesses}{Colors.RESET}")

        # 层级分布
        print(f"\n  {Colors.BOLD}记忆层级分布{Colors.RESET}\n")

        for tier, label in [
            ("working", "工作记忆"),
            ("long_term", "长期记忆"),
            ("archive", "档案"),
        ]:
            count = data.tier_stats.get(tier, 0)
            percentage = (count / data.total_chunks * 100) if data.total_chunks > 0 else 0
            print(f"    {print_tier_badge(tier)}: {Colors.CYAN}{count}{Colors.RESET} ({percentage:.1f}%)")

        # 层级分布图
        if data.tier_distribution:
            print_ascii_chart(data.tier_distribution, width=50, height=8)

    def display_top_memories(self, data: MemoryVisualizationData, limit: int = 10):
        """显示Top记忆

        Args:
            data: 可视化数据
            limit: 显示数量
        """
        print_section(f"Top {limit} 重要记忆")

        headers = ["ID", "重要性", "层级", "访问", "内容预览"]
        rows = []
        for mem in data.top_memories[:limit]:
            rows.append([
                mem["chunk_id"],
                print_importance_score(mem["importance"]),
                print_tier_badge(mem["tier"]),
                str(mem["access_count"]),
                mem["text"],
            ])

        print_table(headers, rows, col_widths=[15, 12, 15, 8, 40])

    def display_recent_accesses(self, data: MemoryVisualizationData, limit: int = 10):
        """显示最近访问

        Args:
            data: 可视化数据
            limit: 显示数量
        """
        print_section(f"最近 {limit} 次访问")

        headers = ["ID", "时间", "次数", "层级", "内容预览"]
        rows = []
        for acc in data.recent_accesses[:limit]:
            rows.append([
                acc["chunk_id"],
                acc["time_ago"],
                str(acc["access_count"]),
                print_tier_badge(acc["tier"]),
                acc["text"],
            ])

        print_table(headers, rows, col_widths=[15, 12, 8, 15, 35])

    def display_importance_distribution(self, data: MemoryVisualizationData):
        """显示重要性分布

        Args:
            data: 可视化数据
        """
        print_section("重要性分布")

        print_ascii_chart(data.importance_distribution, width=50, height=8)

        # 详细统计
        print(f"\n  {Colors.BOLD}详细统计{Colors.RESET}\n")

        for label, count in data.importance_distribution.items():
            percentage = (count / data.total_chunks * 100) if data.total_chunks > 0 else 0
            print(f"    {label}: {Colors.CYAN}{count}{Colors.RESET} ({percentage:.1f}%)")

    def display_full_report(self, data: MemoryVisualizationData):
        """显示完整报告

        Args:
            data: 可视化数据
        """
        self.display_overview(data)
        self.display_importance_distribution(data)
        self.display_top_memories(data)
        self.display_recent_accesses(data)

        print(f"\n{Colors.DIM}{'═' * 70}{Colors.RESET}\n")


# 便捷函数
async def show_memory_dashboard(user_id: int, db_conn):
    """显示记忆仪表板

    Args:
        user_id: 用户ID
        db_conn: 数据库连接
    """
    visualizer = MemoryVisualizer(db_conn)
    data = await visualizer.collect_data(user_id)
    visualizer.display_full_report(data)


async def show_memory_stats(user_id: int, db_conn):
    """显示记忆统计摘要

    Args:
        user_id: 用户ID
        db_conn: 数据库连接
    """
    visualizer = MemoryVisualizer(db_conn)
    data = await visualizer.collect_data(user_id)
    visualizer.display_overview(data)
    print()


async def show_top_memories(user_id: int, db_conn, limit: int = 10):
    """显示Top记忆

    Args:
        user_id: 用户ID
        db_conn: 数据库连接
        limit: 显示数量
    """
    visualizer = MemoryVisualizer(db_conn)
    data = await visualizer.collect_data(user_id)
    visualizer.display_top_memories(data, limit)
    print()
