"""
数据导入脚本
自动扫描 data/raw/ 文件夹，提取元数据，并存储到向量数据库
"""

import os
import sys
import re
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv

# 导入 torch 用于 GPU 检测
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# 自动寻找当前文件所在目录的父目录下的 .env
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# 增加检查逻辑（不打印任何敏感信息）
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("警告：未找到有效的 API Key，请检查 .env 文件")
else:
    print("✓ 成功加载 API Key")

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

# 兼容不同版本的 langchain
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document

from app.pdf_processor import PDFProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingest.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 检测计算设备（GPU/CPU）
def get_device():
    """
    自动检测可用的计算设备
    
    Returns:
        str: 'cuda' 或 'cpu'
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
        logger.info(f"🚀 当前计算设备: CUDA ({device_name})")
        print(f"🚀 当前计算设备: CUDA ({device_name})")
    else:
        device = "cpu"
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch 未安装，使用 CPU 模式")
            print("⚠️  PyTorch 未安装，使用 CPU 模式")
        elif not torch.cuda.is_available():
            logger.info("🚀 当前计算设备: CPU (CUDA 不可用)")
            print("🚀 当前计算设备: CPU (CUDA 不可用)")
        else:
            logger.info("🚀 当前计算设备: CPU")
            print("🚀 当前计算设备: CPU")
    return device

# 在程序启动时检测设备
DEVICE = get_device()


class DataIngester:
    """数据导入器"""
    
    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        vector_db_path: str = "data/vector_db",
        collection_name: str = "financial_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        初始化数据导入器
        
        Args:
            raw_data_dir: 原始数据目录
            vector_db_path: 向量数据库存储路径
            collection_name: Qdrant 集合名称
            chunk_size: 文档块大小
            chunk_overlap: 块重叠大小
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.vector_db_path = Path(vector_db_path)
        self.collection_name = collection_name
        self.pdf_processor = PDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 初始化嵌入模型
        self.embeddings = self._initialize_embeddings()
        
        # 初始化 Qdrant 客户端（用于查询元数据）
        self.client = None
        
        # 确保向量数据库目录存在
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
    
    def _initialize_embeddings(self):
        """
        初始化嵌入模型
        直接使用本地 HuggingFaceEmbeddings 模型（不消耗 API 额度）
        支持 GPU 加速（如果可用）
        
        Returns:
            嵌入模型对象
        """
        logger.info("初始化本地 HuggingFaceEmbeddings 模型...")
        
        try:
            # 优先使用 langchain_huggingface（推荐）
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                logger.info("使用 langchain_huggingface 包")
            except ImportError:
                # 兼容旧版本的 langchain
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    logger.info("使用 langchain_community.embeddings")
                except ImportError:
                    from langchain.embeddings import HuggingFaceEmbeddings
                    logger.info("使用 langchain.embeddings")
            
            # 使用全局检测到的设备
            device = DEVICE
            logger.info(f"加载模型: shibing624/text2vec-base-chinese (设备: {device})")
            embeddings = HuggingFaceEmbeddings(
                model_name='shibing624/text2vec-base-chinese',
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 测试嵌入模型是否可用
            logger.info("测试嵌入模型...")
            test_text = "测试"
            _ = embeddings.embed_query(test_text)
            logger.info(f"✓ 成功初始化 HuggingFaceEmbeddings（本地模型，设备: {device}，不消耗 API 额度）")
            
            return embeddings
        except Exception as e:
            logger.error(f"初始化 HuggingFaceEmbeddings 失败: {str(e)}")
            # 如果 GPU 初始化失败，尝试降级到 CPU
            if DEVICE == "cuda":
                logger.warning("GPU 初始化失败，尝试降级到 CPU...")
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name='shibing624/text2vec-base-chinese',
                        model_kwargs={'device': 'cpu'},
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    _ = embeddings.embed_query("测试")
                    logger.info("✓ 成功降级到 CPU 模式")
                    return embeddings
                except Exception as e2:
                    logger.error(f"CPU 降级也失败: {str(e2)}")
            raise Exception(f"无法初始化嵌入模型: {str(e)}")
    
    def extract_metadata_from_path(self, file_path: Path) -> Dict[str, str]:
        """
        从文件路径提取元数据
        
        Args:
            file_path: PDF 文件路径
            
        Returns:
            包含 company, year, report_type 的元数据字典
        """
        metadata = {
            "company": "",
            "year": "",
            "report_type": "",
            "file_name": file_path.name,
            "file_path": str(file_path)
        }
        
        # 提取公司名（父目录名）
        if file_path.parent.name:
            metadata["company"] = file_path.parent.name
        
        # 从文件名提取年份和报告类型
        file_name = file_path.stem  # 不含扩展名
        
        # 中文数字到阿拉伯数字的映射
        chinese_digit_map = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10'
        }
        
        year = None
        
        # 优先匹配4位阿拉伯数字年份
        four_digit_match = re.search(r'(\d{4})', file_name)
        if four_digit_match:
            year = four_digit_match.group(1)
        else:
            # 匹配中文年份（如：二零二三、二零二四）
            chinese_year_patterns = [
                r'二零([一二三四五六七八九十])',  # 二零二X
                r'二零([一二三四五六七八九十])([一二三四五六七八九十])',  # 二零XX
            ]
            
            for pattern in chinese_year_patterns:
                match = re.search(pattern, file_name)
                if match:
                    if len(match.groups()) == 1:
                        # 二零二X 格式
                        digit = chinese_digit_map.get(match.group(1), '0')
                        year = f"202{digit}"
                    else:
                        # 二零XX 格式
                        tens = chinese_digit_map.get(match.group(1), '0')
                        ones = chinese_digit_map.get(match.group(2), '0')
                        if tens == '10':
                            year = f"20{ones}"
                        else:
                            year = f"20{tens}{ones}"
                    break
            
            # 如果没有找到，尝试查找2位阿拉伯数字年份并转换
            if not year:
                two_digit_match = re.search(r'(\d{2})', file_name)
                if two_digit_match:
                    two_digit = int(two_digit_match.group(1))
                    # 假设是2000-2099年
                    if two_digit < 50:
                        year = f"20{two_digit:02d}"
                    else:
                        year = f"19{two_digit:02d}"
        
        metadata["year"] = year if year else "未知"
        
        # 提取报告类型
        report_type = "未知"
        if "年度" in file_name or "年报" in file_name:
            report_type = "年度报告"
        elif "半年度" in file_name or "半年报" in file_name:
            report_type = "半年度报告"
        elif "季度" in file_name or "季报" in file_name:
            report_type = "季度报告"
        
        metadata["report_type"] = report_type
        
        return metadata
    
    def scan_pdf_files(self) -> List[Path]:
        """
        扫描 data/raw/ 目录下的所有 PDF 文件
        
        Returns:
            PDF 文件路径列表
        """
        pdf_files = []
        
        if not self.raw_data_dir.exists():
            logger.error(f"原始数据目录不存在: {self.raw_data_dir}")
            return pdf_files
        
        # 递归查找所有 PDF 文件
        for pdf_file in self.raw_data_dir.rglob("*.pdf"):
            pdf_files.append(pdf_file)
        
        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")
        return sorted(pdf_files)
    
    def check_file_exists(self, file_hash: str) -> bool:
        """
        检查 Qdrant 中是否已存在具有相同 file_hash 的记录
        
        Args:
            file_hash: 文件的 MD5 哈希值
            
        Returns:
            如果存在返回 True，否则返回 False
        """
        if self.client is None:
            return False
        
        try:
            # 使用 Qdrant 的 scroll 方法配合过滤器查询
            scroll_filter = Filter(
                must=[
                    FieldCondition(
                        key="file_hash",
                        match=MatchValue(value=file_hash)
                    )
                ]
            )
            
            # 只查询一条记录即可判断是否存在
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=1
            )
            
            return len(results) > 0
        except Exception as e:
            # 如果查询失败（可能是集合不存在或字段不存在），返回 False
            logger.debug(f"查询文件哈希时出错（可能集合为空）: {str(e)}")
            return False
    
    def generate_deterministic_id(self, file_name: str, chunk_index: int):
        """
        生成确定性 UUID（基于文件名和分块序号）
        确保同一个块多次写入时使用相同的 ID，实现 upsert
        
        Args:
            file_name: 文件名
            chunk_index: 分块序号
            
        Returns:
            UUID 对象（Qdrant 支持 UUID 对象作为 ID）
        """
        # 使用文件名和分块序号生成确定性 UUID
        namespace = uuid.NAMESPACE_DNS
        unique_string = f"{file_name}_{chunk_index}"
        deterministic_uuid = uuid.uuid5(namespace, unique_string)
        return deterministic_uuid
    
    def process_file(self, pdf_path: Path, vector_store: Optional[Qdrant] = None) -> List[Document]:
        """
        处理单个 PDF 文件（支持增量入库和去重）
        
        Args:
            pdf_path: PDF 文件路径
            vector_store: 向量存储对象（用于检查文件是否已存在）
            
        Returns:
            Document 对象列表（如果文件已存在则返回空列表）
        """
        try:
            logger.info(f"正在检查: {pdf_path.name}")
            
            # 计算文件哈希值（指纹）
            file_hash = self.pdf_processor.calculate_file_hash(str(pdf_path))
            logger.debug(f"文件哈希: {file_hash[:8]}...")
            
            # 检查文件是否已存在
            if self.check_file_exists(file_hash):
                logger.info(f"[SKIP] 文件 {pdf_path.name} 已存在，跳过处理")
                return []
            
            logger.info(f"[NEW] 处理新文件: {pdf_path.name}")
            
            # 提取元数据
            metadata = self.extract_metadata_from_path(pdf_path)
            
            # 处理 PDF
            chunks = self.pdf_processor.process_pdf(str(pdf_path))
            
            # 获取当前时间戳
            processed_at = datetime.now().isoformat()
            
            # 转换为 Document 对象，增加元数据字段
            documents = []
            for idx, chunk in enumerate(chunks):
                # 生成确定性 ID
                doc_id = self.generate_deterministic_id(pdf_path.name, idx)
                
                doc = Document(
                    page_content=chunk["text"],
                    metadata={
                        **metadata,
                        "file_hash": file_hash,
                        "processed_at": processed_at,
                        "source_file": pdf_path.name,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "doc_id": str(doc_id)  # 存储为字符串，使用时转换为 UUID
                    }
                )
                documents.append(doc)
            
            logger.info(f"成功处理 {pdf_path.name}: 生成 {len(documents)} 个文档块")
            return documents
        
        except Exception as e:
            logger.error(f"处理文件 {pdf_path} 时出错: {str(e)}")
            return []
    
    def initialize_vector_store(self) -> Qdrant:
        """
        初始化或加载向量数据库
        
        Returns:
            Qdrant 向量存储对象
        """
        try:
            # 使用本地 Qdrant 客户端
            client = QdrantClient(path=str(self.vector_db_path))
            
            # 保存客户端引用，用于后续查询
            self.client = client
            
            # 检查集合是否存在
            collections = client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # 创建新集合
                # 动态获取嵌入维度
                try:
                    test_embedding = self.embeddings.embed_query("test")
                    embedding_dim = len(test_embedding)
                    logger.info(f"检测到嵌入维度: {embedding_dim}")
                except Exception as e:
                    # 如果无法获取，使用 HuggingFace 模型的默认维度
                    logger.warning(f"无法自动检测嵌入维度: {str(e)}，使用默认值")
                    embedding_dim = 768  # HuggingFace text2vec-base-chinese 维度
                
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建新集合: {self.collection_name} (维度: {embedding_dim})")
            else:
                logger.info(f"使用现有集合: {self.collection_name}")
            
            # 创建向量存储
            vector_store = Qdrant(
                client=client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
            
            return vector_store
        
        except Exception as e:
            logger.error(f"初始化向量数据库失败: {str(e)}")
            raise
    
    def ingest(self, reset_collection: bool = False):
        """
        执行数据导入
        
        Args:
            reset_collection: 是否重置集合（删除现有数据）
        """
        logger.info("=" * 60)
        logger.info("开始数据导入流程")
        logger.info("=" * 60)
        
        # 扫描 PDF 文件
        pdf_files = self.scan_pdf_files()
        
        if len(pdf_files) == 0:
            logger.warning("未找到任何 PDF 文件")
            return
        
        # 初始化向量数据库
        if reset_collection:
            logger.info("重置向量数据库集合...")
            try:
                client = QdrantClient(path=str(self.vector_db_path))
                client.delete_collection(self.collection_name)
                logger.info(f"已删除集合: {self.collection_name}")
            except Exception as e:
                logger.warning(f"删除集合时出错（可能不存在）: {str(e)}")
        
        vector_store = self.initialize_vector_store()
        
        # 处理所有文件（增量入库）
        all_documents = []
        skipped_files = 0
        new_files = 0
        
        with tqdm(total=len(pdf_files), desc="检查 PDF 文件", unit="文件") as pbar:
            for pdf_path in pdf_files:
                documents = self.process_file(pdf_path, vector_store)
                if len(documents) == 0:
                    skipped_files += 1
                else:
                    new_files += 1
                    all_documents.extend(documents)
                pbar.update(1)
        
        # 统计信息
        logger.info("=" * 60)
        logger.info("文件检查完成")
        logger.info(f"  新增文件: {new_files} 个")
        logger.info(f"  跳过文件: {skipped_files} 个（已存在）")
        logger.info(f"  总计文件: {len(pdf_files)} 个")
        logger.info("=" * 60)
        
        if len(all_documents) == 0:
            logger.warning("没有需要入库的新文档块")
            logger.info("=" * 60)
            logger.info("数据导入完成！")
            logger.info(f"总计处理: {len(pdf_files)} 个文件")
            logger.info(f"新增文件: {new_files} 个")
            logger.info(f"跳过文件: {skipped_files} 个")
            logger.info(f"新增文档块: 0 个")
            logger.info(f"向量数据库路径: {self.vector_db_path}")
            logger.info(f"集合名称: {self.collection_name}")
            logger.info("=" * 60)
            return
        
        # 批量添加到向量数据库（使用 upsert 模式）
        logger.info(f"正在将 {len(all_documents)} 个文档块添加到向量数据库（使用 upsert 模式）...")
        
        # 使用 QdrantClient 直接实现 upsert（支持确定性 ID）
        from qdrant_client.models import PointStruct
        
        with tqdm(total=len(all_documents), desc="导入向量数据库", unit="块") as pbar:
            # 分批处理，避免内存问题
            batch_size = 100
            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i + batch_size]
                try:
                    # 准备 upsert 的点
                    points = []
                    for doc in batch:
                        # 生成向量嵌入
                        embedding = self.embeddings.embed_query(doc.page_content)
                        
                        # 获取确定性 ID
                        doc_id = doc.metadata.get("doc_id")
                        if not doc_id:
                            # 如果没有 doc_id，使用文件名和分块序号生成
                            file_name = doc.metadata.get("source_file", "unknown")
                            chunk_index = doc.metadata.get("chunk_index", 0)
                            doc_id = self.generate_deterministic_id(file_name, chunk_index)
                        else:
                            # 如果 doc_id 是字符串，转换为 UUID 对象
                            if isinstance(doc_id, str):
                                try:
                                    doc_id = uuid.UUID(doc_id)
                                except ValueError:
                                    # 如果无法解析为 UUID，重新生成
                                    file_name = doc.metadata.get("source_file", "unknown")
                                    chunk_index = doc.metadata.get("chunk_index", 0)
                                    doc_id = self.generate_deterministic_id(file_name, chunk_index)
                        
                        # 将元数据转换为 Qdrant 格式（所有值必须是基本类型）
                        # 注意：doc_id 不作为 payload，而是作为 PointStruct 的 id
                        payload = {}
                        for key, value in doc.metadata.items():
                            # 跳过 doc_id，因为它已经作为 PointStruct 的 id
                            if key == "doc_id":
                                continue
                            # Qdrant 只支持基本类型（str, int, float, bool, list, dict）
                            if isinstance(value, (str, int, float, bool)):
                                payload[key] = value
                            elif value is None:
                                payload[key] = None
                            else:
                                # 其他类型转换为字符串
                                payload[key] = str(value)
                        
                        # 创建 PointStruct
                        point = PointStruct(
                            id=doc_id,
                            vector=embedding,
                            payload=payload
                        )
                        points.append(point)
                    
                    # 使用 upsert 方法（如果 ID 已存在则更新，否则插入）
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    pbar.update(len(batch))
                except Exception as e:
                    logger.error(f"添加文档批次时出错: {str(e)}")
                    # 如果直接 upsert 失败，尝试使用 LangChain 的 add_documents 作为后备
                    try:
                        logger.warning("尝试使用 LangChain 的 add_documents 作为后备方案...")
                        vector_store.add_documents(batch)
                        pbar.update(len(batch))
                    except Exception as e2:
                        logger.error(f"后备方案也失败: {str(e2)}")
                        pbar.update(len(batch))
        
        logger.info("=" * 60)
        logger.info("数据导入完成！")
        logger.info(f"总计处理: {len(pdf_files)} 个文件")
        logger.info(f"  新增文件: {new_files} 个")
        logger.info(f"  跳过文件: {skipped_files} 个")
        logger.info(f"新增文档块: {len(all_documents)} 个")
        logger.info(f"向量数据库路径: {self.vector_db_path}")
        logger.info(f"集合名称: {self.collection_name}")
        logger.info("=" * 60)
        
        # 释放 GPU 显存
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✓ 已释放 GPU 显存")


def main():
    """主函数"""
    import argparse
    
    # 环境变量已在文件开头加载，这里进行最终检查
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("=" * 60)
        print("错误：未检测到 .env 中的 API Key")
        print("=" * 60)
        print(f"请检查 .env 文件是否存在: {env_path}")
        print("确保 .env 文件包含以下内容：")
        print("  OPENAI_API_KEY=your_api_key_here")
        print("  OPENAI_API_BASE=your_api_base_url")
        print("  MODEL_NAME=your_model_name")
        print("=" * 60)
        print("提示：请参考 .env.example 文件进行配置")
        print("=" * 60)
        sys.exit(1)
    
    # 打印环境变量信息（健壮性检查，不打印敏感信息）
    model_name = os.getenv("MODEL_NAME")
    print("=" * 60)
    print("环境变量检查：")
    if model_name:
        print(f"  MODEL_NAME: {model_name}")
    print(f"  API Key: {'已设置' if api_key else '未设置'}")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="导入 PDF 文档到向量数据库")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="原始数据目录路径（默认: data/raw）"
    )
    parser.add_argument(
        "--vector-db-path",
        type=str,
        default="data/vector_db",
        help="向量数据库存储路径（默认: data/vector_db）"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="financial_documents",
        help="Qdrant 集合名称（默认: financial_documents）"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="文档块大小（默认: 1000）"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="块重叠大小（默认: 200）"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置集合（删除现有数据）"
    )
    
    args = parser.parse_args()
    
    # 创建导入器并执行导入
    ingester = DataIngester(
        raw_data_dir=args.raw_dir,
        vector_db_path=args.vector_db_path,
        collection_name=args.collection_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    ingester.ingest(reset_collection=args.reset)


if __name__ == "__main__":
    main()
