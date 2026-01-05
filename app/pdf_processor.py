"""
PDF 处理模块
使用 pdfplumber 提取文本和表格，将表格转换为 Markdown 格式
"""

import pdfplumber
import re
from typing import List, Dict, Optional
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFProcessor:
    """PDF 文档处理器"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        初始化 PDF 处理器
        
        Args:
            chunk_size: 文档块大小
            chunk_overlap: 块重叠大小
            separators: 自定义分隔符列表
        """
        if separators is None:
            separators = ["\n\n", "\n", "。", "，", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从 PDF 文件中提取文本和表格（优化版，支持复杂表格和多模态准备）
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            提取的文本内容（包含 Markdown 格式的表格）
        """
        content_parts = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # 提取文本
                    text = page.extract_text()
                    if text:
                        content_parts.append(f"## 第 {page_num} 页\n\n{text}\n")
                    
                    # 优化表格提取：使用精细化配置
                    # 针对财务表格的特殊处理
                    table_settings = {
                        "vertical_strategy": "lines_strict",  # 严格按线条提取
                        "horizontal_strategy": "lines_strict",
                        "explicit_vertical_lines": page.curves + page.lines,  # 使用所有线条
                        "explicit_horizontal_lines": page.curves + page.lines,
                        "snap_tolerance": 3,  # 容差设置
                        "join_tolerance": 3,
                        "edge_tolerance": 3,
                        "min_words_vertical": 1,  # 最小单元格字数
                        "min_words_horizontal": 1,
                    }
                    
                    # 尝试提取表格
                    tables = page.extract_tables(table_settings=table_settings)
                    
                    # 如果没有提取到表格，尝试更宽松的策略
                    if not tables:
                        table_settings_relaxed = {
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 5,
                            "join_tolerance": 5,
                        }
                        tables = page.extract_tables(table_settings=table_settings_relaxed)
                    
                    if tables:
                        for table_num, table in enumerate(tables, start=1):
                            # 使用优化的表格转换
                            markdown_table = self._table_to_markdown_enhanced(table)
                            if markdown_table:
                                content_parts.append(
                                    f"### 第 {page_num} 页 - 表格 {table_num}\n\n{markdown_table}\n\n"
                                )
        
        except Exception as e:
            raise Exception(f"提取 PDF 内容时出错: {str(e)}")
        
        return "\n".join(content_parts)
    
    def _table_to_markdown_enhanced(self, table: List[List]) -> str:
        """
        增强版表格转 Markdown（支持复杂网格数据和多模态准备）
        
        Args:
            table: 表格数据（二维列表）
            
        Returns:
            Markdown 格式的表格字符串（标准格式，便于 AI 读取）
        """
        if not table or len(table) == 0:
            return ""
        
        # 清理和规范化表格数据
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # 转换为字符串并清理
                    cell_str = str(cell).strip()
                    # 保留财务数据格式（数字、千位分隔符等）
                    # 移除多余的空白字符，但保留必要的空格
                    cell_str = " ".join(cell_str.split())
                    cleaned_row.append(cell_str)
            cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) == 0:
            return ""
        
        # 确定列数（取最大行长度）
        max_cols = max(len(row) for row in cleaned_table) if cleaned_table else 0
        if max_cols == 0:
            return ""
        
        # 统一所有行的列数
        for row in cleaned_table:
            while len(row) < max_cols:
                row.append("")
        
        # 计算每列的最大宽度（用于对齐）
        col_widths = [0] * max_cols
        for row in cleaned_table:
            for col_idx, cell in enumerate(row):
                # 计算显示宽度（中文字符算2个宽度）
                width = self._get_display_width(str(cell))
                col_widths[col_idx] = max(col_widths[col_idx], width)
        
        # 生成标准 Markdown 表格（便于 AI 解析）
        markdown_lines = []
        
        # 表头
        if len(cleaned_table) > 0:
            header = cleaned_table[0]
            header_row = "| " + " | ".join(
                self._pad_cell(str(cell), col_widths[i]) 
                for i, cell in enumerate(header)
            ) + " |"
            markdown_lines.append(header_row)
            
            # 分隔行（标准 Markdown 格式）
            separator = "| " + " | ".join(
                "-" * max(3, col_widths[i])  # 至少3个字符
                for i in range(max_cols)
            ) + " |"
            markdown_lines.append(separator)
            
            # 数据行
            for row in cleaned_table[1:]:
                data_row = "| " + " | ".join(
                    self._pad_cell(str(cell), col_widths[i]) 
                    for i, cell in enumerate(row)
                ) + " |"
                markdown_lines.append(data_row)
        
        return "\n".join(markdown_lines)
    
    def _table_to_markdown(self, table: List[List]) -> str:
        """
        将表格转换为 Markdown 格式，确保财务数据的行列对齐
        
        Args:
            table: 表格数据（二维列表）
            
        Returns:
            Markdown 格式的表格字符串
        """
        if not table or len(table) == 0:
            return ""
        
        # 清理表格数据
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # 转换为字符串并清理
                    cell_str = str(cell).strip()
                    # 保留数字格式，特别是财务数据中的千位分隔符
                    cleaned_row.append(cell_str)
            cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) == 0:
            return ""
        
        # 确定列数（取最大行长度）
        max_cols = max(len(row) for row in cleaned_table) if cleaned_table else 0
        if max_cols == 0:
            return ""
        
        # 统一所有行的列数
        for row in cleaned_table:
            while len(row) < max_cols:
                row.append("")
        
        # 计算每列的最大宽度（用于对齐）
        col_widths = [0] * max_cols
        for row in cleaned_table:
            for col_idx, cell in enumerate(row):
                # 计算显示宽度（中文字符算2个宽度）
                width = self._get_display_width(str(cell))
                col_widths[col_idx] = max(col_widths[col_idx], width)
        
        # 生成 Markdown 表格
        markdown_lines = []
        
        # 表头
        if len(cleaned_table) > 0:
            header = cleaned_table[0]
            header_row = "| " + " | ".join(
                self._pad_cell(str(cell), col_widths[i]) 
                for i, cell in enumerate(header)
            ) + " |"
            markdown_lines.append(header_row)
            
            # 分隔行
            separator = "| " + " | ".join(
                "-" * (col_widths[i] + 1) 
                for i in range(max_cols)
            ) + " |"
            markdown_lines.append(separator)
            
            # 数据行
            for row in cleaned_table[1:]:
                data_row = "| " + " | ".join(
                    self._pad_cell(str(cell), col_widths[i]) 
                    for i, cell in enumerate(row)
                ) + " |"
                markdown_lines.append(data_row)
        
        return "\n".join(markdown_lines)
    
    def _get_display_width(self, text: str) -> int:
        """
        计算文本的显示宽度（中文字符算2个宽度）
        
        Args:
            text: 文本内容
            
        Returns:
            显示宽度
        """
        width = 0
        for char in text:
            # 中文字符、全角字符等算2个宽度
            if ord(char) > 127:
                width += 2
            else:
                width += 1
        return width
    
    def _pad_cell(self, cell: str, width: int) -> str:
        """
        填充单元格内容以达到指定宽度
        
        Args:
            cell: 单元格内容
            width: 目标宽度
            
        Returns:
            填充后的单元格内容
        """
        current_width = self._get_display_width(cell)
        if current_width >= width:
            return cell
        
        # 计算需要添加的空格数
        padding = width - current_width
        return cell + " " * padding
    
    def split_document(self, text: str) -> List[str]:
        """
        使用 RecursiveCharacterTextSplitter 分割文档
        
        Args:
            text: 要分割的文本
            
        Returns:
            分割后的文本块列表
        """
        chunks = self.text_splitter.split_text(text)
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        处理 PDF 文件，提取内容并分割为块
        
        Args:
            pdf_path: PDF 文件路径
            
        Returns:
            文档块列表，每个块包含 'text' 字段
        """
        # 提取文本和表格
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # 分割文档
        chunks = self.split_document(full_text)
        
        # 转换为字典列表
        documents = [{"text": chunk} for chunk in chunks]
        
        return documents

