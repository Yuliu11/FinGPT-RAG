"""
财务计算工具
提供金融分析相关的计算函数
"""

from typing import Optional


def growth_rate_calc(current_value: float, previous_value: float, period: Optional[str] = "年度") -> dict:
    """
    计算增长率
    
    Args:
        current_value: 当前期数值
        previous_value: 上期数值
        period: 计算周期（默认："年度"）
        
    Returns:
        包含增长率计算结果的字典
    """
    if previous_value == 0:
        return {
            "error": "上期数值不能为0，无法计算增长率",
            "current_value": current_value,
            "previous_value": previous_value,
            "period": period
        }
    
    # 计算增长率（百分比）
    growth_rate = ((current_value - previous_value) / previous_value) * 100
    
    result = {
        "current_value": current_value,
        "previous_value": previous_value,
        "growth_rate": round(growth_rate, 2),
        "growth_rate_percentage": f"{growth_rate:.2f}%",
        "period": period,
        "calculation": f"(({current_value} - {previous_value}) / {previous_value}) × 100 = {growth_rate:.2f}%"
    }
    
    return result

