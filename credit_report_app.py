"""
银行授信业务调查报告自动填写应用
"""

import os
import tempfile
import re
import json
import base64
import hashlib
from pathlib import Path
from urllib.parse import urlparse, quote_plus
from urllib.robotparser import RobotFileParser
from io import BytesIO
import subprocess
import shutil

import chromadb
import ollama
import streamlit as st
import pandas as pd
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 导入OCR模块
import sys
sys.path.append('.')
from ocr import MistralOCRProcessor

# 语言配置
TEXTS = {
    "zh": {
        "page_title": "银行授信业务调查报告自动填写系统",
        "header": "🏦 银行授信业务调查报告自动填写系统",
        "applicant_name": "申请人名称（企业/公司）",
        "applicant_placeholder": "请输入申请人企业/公司名称",
        "upload_report": "上传银行授信业务调查报告（PDF格式）",
        "process_button": "🚀 开始处理",
        "searching": "🔍 正在搜索申请人信息...",
        "found_urls": "📝 找到 {} 个相关网页",
        "ocr_processing": "🔄 正在进行OCR处理...",
        "report_parsing": "📋 正在解析报告结构...",
        "generating_summary": "📊 正在生成财报总结...",
        "filling_report": "✍️ 正在填写报告第 {} 部分：{}",
        "download_summary": "📥 下载财报总结报告",
        "download_filled": "📥 下载填写完成的报告",
        "processing_complete": "✅ 处理完成！",
        "error_occurred": "❌ 处理过程中出现错误",
    }
}

def get_text(key: str) -> str:
    """获取当前语言的文本"""
    return TEXTS["zh"].get(key, key)

# 系统提示词
FINANCIAL_SUMMARY_PROMPT = """
你是一位专业的银行调查员。请基于提供的申请人企业信息，生成一份详细的财务状况总结报告。

请包含以下方面的分析：
1. **企业基本情况** - 公司性质、经营范围、注册资本等
2. **财务状况分析** - 资产负债情况、盈利能力、现金流状况
3. **经营情况分析** - 主营业务、市场地位、经营风险等
4. **信用状况评估** - 历史信用记录、还款能力分析
5. **风险因素识别** - 潜在风险点和风险等级评估

请基于提供的上下文信息，详细填写以上各项内容。如果某些信息不足，请明确说明数据来源限制。

上下文信息：{context}
"""

REPORT_FILLING_PROMPT = """
你是一位专业的银行信贷业务专家。请基于提供的申请人企业信息，为银行授信业务调查报告的以下部分提供详细的填写内容。

需要填写的报告部分：
{section_content}

请按照以下要求填写：
1. **严格保持原有的markdown格式和结构**
2. **只在需要填写的空白处、表格空白单元格、或明显需要补充的地方进行填写**
3. **不要新增原报告中没有的章节、表格或内容块**
4. **不要删除或大幅修改原有的标题、格式、表格结构**
5. 内容必须准确、专业、详实，符合银行授信业务规范
6. 数据引用要有来源依据，分析要客观中立

申请人企业信息上下文：
{context}

请直接输出填写后的完整内容，严格保持原有的markdown格式，只在空白或需要填写的地方添加相应信息。
"""

class CreditReportProcessor:
    def __init__(self):
        """初始化处理器"""
        self.collection = None
        self.chroma_client = None
        self.ocr_processor = None
        # OCR缓存目录
        self.ocr_cache_dir = Path("./ocr_cache")
        self.ocr_cache_dir.mkdir(exist_ok=True)
        self.init_components()
    
    def init_components(self):
        """初始化组件"""
        # 初始化向量数据库
        self.init_vector_db()
        
        # OCR处理器将在需要时动态初始化
    
    def init_ocr_processor(self, api_key: str):
        """动态初始化OCR处理器"""
        if api_key:
            self.ocr_processor = MistralOCRProcessor(api_key)
        else:
            self.ocr_processor = None
    
    def get_file_hash(self, file_content: bytes) -> str:
        """计算文件内容的SHA256哈希值"""
        return hashlib.sha256(file_content).hexdigest()
    
    def get_cached_ocr_result(self, file_hash: str) -> str:
        """获取缓存的OCR结果"""
        cache_file = self.ocr_cache_dir / f"{file_hash}.md"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def save_ocr_result_to_cache(self, file_hash: str, markdown_content: str):
        """保存OCR结果到缓存"""
        cache_file = self.ocr_cache_dir / f"{file_hash}.md"
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def init_vector_db(self):
        """初始化向量数据库"""
        ollama_ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name="paraphrase-multilingual:latest",
        )

        self.chroma_client = chromadb.PersistentClient(
            path="./credit-report-db", 
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="applicant_info",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"},
        )
    
    def search_applicant_info(self, applicant_name: str, num_results: int = 20) -> list[str]:
        """搜索申请人企业信息"""
        import asyncio
        
        try:
            # 获取调试模式状态
            debug_mode = getattr(st.session_state, 'debug_mode', False)
            
            # 简化搜索策略 - 只使用申请人名称进行搜索
            search_query = applicant_name
            
            # 根据调试模式显示不同级别的信息
            if debug_mode:
                st.info(f"📝 搜索查询: {search_query}")
            
            # 直接搜索申请人名称
            urls = asyncio.run(self.get_web_urls(search_query, num_results, debug_mode))
            
            if debug_mode:
                st.success(f"🎯 找到 {len(urls)} 个相关链接")
            else:
                st.success(f"🎯 找到 {len(urls)} 个相关链接")
            
            return urls
            
        except Exception as e:
            st.error(f"搜索失败: {str(e)}")
            return []
    
    async def get_web_urls(self, search_term: str, num_results: int = 5, debug_mode: bool = False) -> list[str]:
        """通过Bing搜索获取URL"""
        try:
            query = search_term
            # 确保查询字符串的编码正确
            encoded_query = quote_plus(query, encoding='utf-8')
            search_url = f"https://www.bing.com/search?q={encoded_query}&count={num_results}"
            
            # 根据调试模式显示信息
            if debug_mode:
                st.info(f"🔍 搜索查询: {query}")
                st.info(f"🌐 搜索URL长度: {len(search_url)} 字符")
                st.code(search_url, language="text")
            
            # 总是打印到console用于调试，即使不在debug模式
            print(f"=== 搜索调试信息 ===")
            print(f"原始查询: {query}")
            print(f"编码查询: {encoded_query}")
            print(f"完整URL: {search_url}")
            print(f"URL长度: {len(search_url)} 字符")
            print("==================")
            
            crawler_config = CrawlerRunConfig(
                excluded_tags=["nav", "footer", "header", "form", "img"],
                only_text=False,
                exclude_social_media_links=False,
                keep_data_attributes=False,
                cache_mode=CacheMode.BYPASS,
                remove_overlay_elements=True,
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
                page_timeout=30000,
                delay_before_return_html=3.0,
            )
            
            browser_config = BrowserConfig(headless=True, text_mode=False, light_mode=True)
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # 确保传递的URL是完整的
                print(f"传递给crawler的URL: {search_url}")
                result = await crawler.arun(url=search_url, config=crawler_config)
                
                if not result.html:
                    if debug_mode:
                        st.warning(f"未获取到搜索结果页面内容，查询: {query}")
                    print(f"未获取到HTML内容，URL: {search_url}")
                    return []
                
                urls = self.extract_bing_links(result.html, num_results)
                if debug_mode:
                    if urls:
                        st.success(f"✅ 从查询 '{query}' 中找到 {len(urls)} 个链接")
                        st.write("找到的链接:")
                        for i, url in enumerate(urls, 1):
                            st.write(f"{i}. {url}")
                    else:
                        st.warning(f"⚠️ 查询 '{query}' 未找到有效链接")
                
                print(f"提取到 {len(urls)} 个链接")
                return self.check_robots_txt(urls)
                
        except Exception as e:
            error_msg = f"网页搜索失败，查询: {search_term}, 错误: {str(e)}"
            if debug_mode:
                st.error(error_msg)
            else:
                # 在非调试模式下，只记录到日志，不显示给用户
                print(error_msg)
            return []
    
    def extract_bing_links(self, html: str, max_results: int = 5) -> list[str]:
        """从Bing搜索结果中提取链接"""
        urls = []
        
        if getattr(st.session_state, 'debug_mode', False):
            print(f"HTML内容长度: {len(html)} 字符")
            print("HTML前500字符:")
            print(html[:500])
            print("=" * 50)
        
        # 主要搜索结果链接 - 这是最可靠的方式
        h2_pattern = r'<h2[^>]*><a[^>]*href=["\']([^"\']+)["\']'
        h2_matches = re.findall(h2_pattern, html, re.IGNORECASE)
        
        exclude_domains = ['bing.com', 'microsoft.com', 'msn.com', 'live.com', 'microsofttranslator.com']
        
        if getattr(st.session_state, 'debug_mode', False):
            print(f"从h2标签找到 {len(h2_matches)} 个链接")
        
        for href in h2_matches:
            try:
                # 清理URL
                if '&amp;' in href:
                    href = href.split('&amp;')[0]
                
                if href.startswith(('http://', 'https://')):
                    if not any(domain in href for domain in exclude_domains):
                        urls.append(href)
                        if getattr(st.session_state, 'debug_mode', False):
                            print(f"✅ 添加URL: {href}")
                        
                        if len(urls) >= max_results:
                            break
                            
            except Exception as e:
                if getattr(st.session_state, 'debug_mode', False):
                    print(f"❌ 处理URL时出错: {e}")
                continue
        
        if getattr(st.session_state, 'debug_mode', False):
            print(f"最终提取到 {len(urls)} 个有效URL")
            for i, url in enumerate(urls, 1):
                print(f"{i}. {url}")
        
        return urls
    
    def check_robots_txt(self, urls: list[str]) -> list[str]:
        """检查robots.txt"""
        allowed_urls = []
        for url in urls:
            try:
                robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
                rp = RobotFileParser(robots_url)
                rp.read()
                if rp.can_fetch("*", url):
                    allowed_urls.append(url)
            except Exception:
                allowed_urls.append(url)
        return allowed_urls
    
    def crawl_webpages(self, urls: list[str], query: str) -> list[CrawlResult]:
        """爬取网页内容"""
        import asyncio
        
        try:
            return asyncio.run(self._crawl_webpages_async(urls, query))
        except Exception as e:
            st.error(f"网页爬取失败: {str(e)}")
            return []
    
    async def _crawl_webpages_async(self, urls: list[str], query: str) -> list[CrawlResult]:
        """异步爬取网页内容"""
        try:
            bm25_filter = BM25ContentFilter(user_query=query, bm25_threshold=0.5)
            md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

            crawler_config = CrawlerRunConfig(
                markdown_generator=md_generator,
                excluded_tags=["nav", "footer", "header", "form", "img", "a"],
                only_text=True,
                exclude_social_media_links=True,
                keep_data_attributes=False,
                cache_mode=CacheMode.BYPASS,
                remove_overlay_elements=True,
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                page_timeout=20000,
            )
            
            browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)

            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = await crawler.arun_many(urls, config=crawler_config)
                return results
        except Exception as e:
            st.error(f"异步网页爬取失败: {str(e)}")
            return []
    
    def add_to_vector_database(self, results: list[CrawlResult]):
        """将爬取结果添加到向量数据库"""
        total_documents = 0
        
        for result in results:
            documents, metadatas, ids = [], [], []
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=150,
                separators=["\n\n", "\n", "。", "？", "！", ". ", ", ", " ", ""]
            )
            
            if result.markdown_v2 and result.markdown_v2.fit_markdown:
                markdown_result = result.markdown_v2.fit_markdown
            else:
                continue

            temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding='utf-8')
            temp_file.write(markdown_result)
            temp_file.flush()

            try:
                loader = UnstructuredMarkdownLoader(temp_file.name, mode="single")
                docs = loader.load()
                all_splits = text_splitter.split_documents(docs)
            except Exception:
                all_splits = []
            finally:
                os.unlink(temp_file.name)

            normalized_url = self.normalize_url(result.url)

            if all_splits:
                for idx, split in enumerate(all_splits):
                    documents.append(split.page_content)
                    metadatas.append({"source": result.url})
                    ids.append(f"{normalized_url}_{idx}")

                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                total_documents += len(documents)
        
        return total_documents
    
    def normalize_url(self, url):
        """规范化URL"""
        return (
            url.replace("https://", "")
            .replace("www.", "")
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
    
    def process_pdf_with_ocr(self, pdf_file, api_key: str) -> str:
        """使用OCR处理PDF文件，支持缓存功能"""
        # 读取文件内容并计算哈希
        file_content = pdf_file.read()
        file_hash = self.get_file_hash(file_content)
        
        # 检查是否有缓存的结果
        cached_result = self.get_cached_ocr_result(file_hash)
        if cached_result:
            st.info("✅ 使用缓存的OCR结果，跳过重复处理")
            return cached_result
        
        # 使用提供的API Key初始化OCR处理器
        self.init_ocr_processor(api_key)
        
        if not self.ocr_processor:
            raise Exception("OCR处理器未初始化，请检查MISTRAL_API_KEY")
        
        # 保存上传的文件到临时位置
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # 创建临时输出目录
            output_dir = tempfile.mkdtemp()
            
            # 进行OCR处理
            response_data, extracted_images = self.ocr_processor.process_pdf(
                tmp_file_path, output_dir, "markdown"
            )
            
            # 读取生成的markdown文件
            markdown_file = os.path.join(output_dir, "ocr_result.md")
            if os.path.exists(markdown_file):
                with open(markdown_file, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                # 保存到缓存
                self.save_ocr_result_to_cache(file_hash, markdown_content)
                
                return markdown_content
            else:
                raise Exception("OCR处理失败，未生成markdown文件")
                
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass
    
    def parse_report_sections(self, markdown_content: str) -> list[dict]:
        """解析报告的章节结构"""
        sections = []
        lines = markdown_content.split('\n')
        current_section = {"title": "", "content": "", "level": 0}
        
        for line in lines:
            if line.strip().startswith('#'):
                # 保存前一个section
                if current_section["title"]:
                    sections.append(current_section.copy())
                
                # 开始新的section
                level = len(line) - len(line.lstrip('#'))
                title = line.strip('#').strip()
                current_section = {
                    "title": title,
                    "content": line + "\n",
                    "level": level
                }
            else:
                current_section["content"] += line + "\n"
        
        # 添加最后一个section
        if current_section["title"]:
            sections.append(current_section)
        
        return sections
    
    def query_vector_db(self, query: str, n_results: int = 10) -> str:
        """查询向量数据库"""
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            documents = results.get("documents")[0] if results.get("documents") else []
            return " ".join(documents)
        except Exception:
            return ""
    
    def filter_think_content(self, content: str) -> str:
        """过滤掉内容中的<think></think>标签及其内容"""
        if not content:
            return content
            
        # 使用正则表达式移除<think>...</think>内容，支持多行和嵌套
        patterns = [
            r'<think>.*?</think>',  # 标准格式
            r'<thinking>.*?</thinking>',  # 可能的变体
            r'<THINK>.*?</THINK>',  # 大写变体
            r'<THINKING>.*?</THINKING>',  # 大写变体
        ]
        
        cleaned_content = content
        for pattern in patterns:
            cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL | re.IGNORECASE)
        
        # 清理多余的空行和空白字符
        cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_content)
        cleaned_content = re.sub(r'^\s+|\s+$', '', cleaned_content, flags=re.MULTILINE)
        
        return cleaned_content.strip()
    
    def call_llm(self, prompt: str, context: str = "") -> str:
        """调用LLM生成内容"""
        try:
            full_prompt = prompt.format(context=context)
            
            response = ollama.chat(
                model="deepseek-r1:32b",
                messages=[{
                    "role": "user",
                    "content": full_prompt
                }]
            )
            
            # 过滤掉思考内容
            raw_content = response["message"]["content"]
            filtered_content = self.filter_think_content(raw_content)
            
            return filtered_content
        except Exception as e:
            return f"LLM调用失败: {str(e)}"
    
    def generate_financial_summary(self, applicant_name: str) -> str:
        """生成财报总结"""
        context = self.query_vector_db(f"{applicant_name} 财务 经营")
        return self.call_llm(FINANCIAL_SUMMARY_PROMPT, context)
    
    def fill_report_section(self, section_content: str, applicant_name: str) -> str:
        """填写报告章节"""
        # 提取章节中的关键信息进行向量检索
        section_keywords = self.extract_section_keywords(section_content)
        # 结合申请人名称和章节关键词进行检索
        query = f"{applicant_name} {section_keywords}"
        context = self.query_vector_db(query)
        return self.call_llm(REPORT_FILLING_PROMPT.format(section_content=section_content, context=context))
    
    def extract_section_keywords(self, section_content: str) -> str:
        """从章节内容中提取关键词用于向量检索"""
        # 移除markdown语法标记
        import re
        text = re.sub(r'[#*\-|]', ' ', section_content)
        
        # 提取关键词（中文词汇）
        keywords = re.findall(r'[\u4e00-\u9fff]+', text)
        
        # 过滤掉过短的词和常见词
        filtered_keywords = []
        common_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '了', '也', '就', '都', '将', '为', '从', '等', '及', '以', '可', '能', '应', '要', '如', '下', '上', '中', '内', '外', '前', '后', '左', '右'}
        
        for keyword in keywords:
            if len(keyword) >= 2 and keyword not in common_words:
                filtered_keywords.append(keyword)
        
        # 返回前10个关键词
        return ' '.join(filtered_keywords[:10])
    
    def convert_markdown_to_word(self, markdown_text: str, filename: str = "output.docx") -> bytes:
        """将Markdown转换为Word文档"""
        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()
            
            # 保存Markdown到临时文件
            qmd_path = os.path.join(temp_dir, "output.qmd")
            with open(qmd_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            
            # 使用quarto转换
            docx_filename = filename
            docx_path = os.path.join(temp_dir, docx_filename)
            
            current_dir = os.getcwd()
            try:
                os.chdir(temp_dir)
                subprocess.run(
                    ["quarto", "render", "output.qmd", "--to", "docx", "-o", docx_filename],
                    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            finally:
                os.chdir(current_dir)
            
            # 读取生成的文档
            with open(docx_path, "rb") as f:
                docx_bytes = f.read()
            
            # 清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return docx_bytes
            
        except Exception as e:
            raise Exception(f"转换失败: {str(e)}")
    
    def clear_ocr_cache(self):
        """清理OCR缓存"""
        try:
            if self.ocr_cache_dir.exists():
                shutil.rmtree(self.ocr_cache_dir)
                self.ocr_cache_dir.mkdir(exist_ok=True)
                return True
        except Exception:
            pass
        return False
    
    def get_cache_info(self):
        """获取缓存信息"""
        if not self.ocr_cache_dir.exists():
            return {"count": 0, "size": "0 MB"}
        
        try:
            cache_files = list(self.ocr_cache_dir.glob("*.md"))
            count = len(cache_files)
            
            total_size = sum(f.stat().st_size for f in cache_files)
            size_mb = total_size / (1024 * 1024)
            
            return {
                "count": count,
                "size": f"{size_mb:.2f} MB"
            }
        except Exception:
            return {"count": 0, "size": "0 MB"}

# 主应用
def main():
    st.set_page_config(
        page_title=get_text("page_title"),
        page_icon="🏦",
        layout="wide"
    )
    
    st.header(get_text("header"))
    
    # 初始化处理器
    if 'processor' not in st.session_state:
        st.session_state.processor = CreditReportProcessor()
    
    processor = st.session_state.processor
    
    # 侧边栏输入
    with st.sidebar:
        st.subheader("📝 输入信息")
        
        # 申请人名称输入
        applicant_name = st.text_input(
            get_text("applicant_name"),
            value="锐道企业管理咨询(上海)有限公司",
            placeholder=get_text("applicant_placeholder"),
            key="applicant_name"
        )
        
        # OCR API Key输入
        ocr_api_key = st.text_input(
            "Mistral OCR API Key",
            value="EoBu0h6XasHmXH2Y2izoqWO43shRUT4D",
            placeholder="请输入Mistral OCR API Key",
            type="password",
            key="ocr_api_key",
            help="用于PDF OCR处理的Mistral API密钥"
        )
        
        # PDF文件上传
        uploaded_file = st.file_uploader(
            get_text("upload_report"),
            type=["pdf"],
            key="pdf_upload"
        )
        
        # 处理按钮
        process_clicked = st.button(
            get_text("process_button"),
            type="primary",
            disabled=not (applicant_name and uploaded_file and ocr_api_key)
        )
        
        # OCR缓存管理
        st.subheader("🗂️ OCR缓存管理")
        cache_info = processor.get_cache_info()
        st.info(f"缓存文件数量: {cache_info['count']}\n缓存大小: {cache_info['size']}")
        
        if st.button("🗑️ 清理OCR缓存"):
            if processor.clear_ocr_cache():
                st.success("✅ OCR缓存已清理")
                st.rerun()
            else:
                st.error("❌ 清理缓存失败")
        
        # 调试模式开关
        st.subheader("🔧 调试选项")
        debug_mode = st.checkbox("显示详细搜索日志", value=False, help="开启后会显示详细的搜索过程信息")
        
        # 将调试模式状态保存到session state
        st.session_state.debug_mode = debug_mode
    
    # 主内容区域
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 财报总结")
        summary_placeholder = st.empty()
        
        if 'financial_summary' in st.session_state:
            summary_placeholder.markdown(st.session_state.financial_summary)
            
            # 下载财报总结按钮
            if st.button(get_text("download_summary")):
                try:
                    docx_bytes = processor.convert_markdown_to_word(
                        st.session_state.financial_summary,
                        f"{applicant_name}_财报总结.docx"
                    )
                    st.download_button(
                        label="💾 下载财报总结文档",
                        data=docx_bytes,
                        file_name=f"{applicant_name}_财报总结.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                except Exception as e:
                    st.error(f"生成文档失败: {str(e)}")
    
    with col2:
        st.subheader("📋 填写进度")
        progress_placeholder = st.empty()
        filled_content_placeholder = st.empty()
        
        if 'filled_report' in st.session_state:
            filled_content_placeholder.markdown(st.session_state.filled_report)
            
            # 下载填写完成的报告按钮
            if st.button(get_text("download_filled")):
                try:
                    docx_bytes = processor.convert_markdown_to_word(
                        st.session_state.filled_report,
                        f"{applicant_name}_授信调查报告.docx"
                    )
                    st.download_button(
                        label="💾 下载填写完成的报告",
                        data=docx_bytes,
                        file_name=f"{applicant_name}_授信调查报告.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                except Exception as e:
                    st.error(f"生成文档失败: {str(e)}")
    
    # 处理逻辑
    if process_clicked:
        try:
            # 1. 搜索申请人信息
            with st.spinner(get_text("searching")):
                urls = processor.search_applicant_info(applicant_name)
                if urls:
                    st.success(get_text("found_urls").format(len(urls)))
                    
                    # 爬取网页并建立向量数据库
                    results = processor.crawl_webpages(urls, applicant_name)
                    total_docs = processor.add_to_vector_database(results)
                    st.info(f"📚 已将 {total_docs} 个文档片段添加到数据库")
                else:
                    st.warning("未找到相关网页信息")
            
            # 2. OCR处理PDF
            with st.spinner(get_text("ocr_processing")):
                markdown_content = processor.process_pdf_with_ocr(uploaded_file, ocr_api_key)
                st.success("✅ OCR处理完成")
            
            # 3. 生成财报总结
            with st.spinner(get_text("generating_summary")):
                financial_summary = processor.generate_financial_summary(applicant_name)
                # 确保过滤掉think内容
                financial_summary = processor.filter_think_content(financial_summary)
                st.session_state.financial_summary = financial_summary
                summary_placeholder.markdown(financial_summary)
            
            # 4. 解析报告结构并逐段填写
            with st.spinner(get_text("report_parsing")):
                sections = processor.parse_report_sections(markdown_content)
                st.info(f"📋 解析到 {len(sections)} 个报告段落")
            
            # 5. 逐段填写报告
            filled_sections = []
            progress_bar = st.progress(0)
            
            for i, section in enumerate(sections):
                section_title = section["title"] or f"第{i+1}段"
                
                # 更新进度
                progress = (i + 1) / len(sections)
                progress_bar.progress(progress)
                progress_placeholder.text(get_text("filling_report").format(i+1, section_title))
                
                # 填写当前段落
                filled_content = processor.fill_report_section(section["content"], applicant_name)
                # 确保过滤掉think内容
                filled_content = processor.filter_think_content(filled_content)
                filled_sections.append(filled_content)
                
                # 实时更新显示内容
                current_filled = "\n\n".join(filled_sections)
                st.session_state.filled_report = current_filled
                filled_content_placeholder.markdown(current_filled)
            
            # 完成处理
            progress_placeholder.success(get_text("processing_complete"))
            st.balloons()
            
        except Exception as e:
            st.error(f"{get_text('error_occurred')}: {str(e)}")

if __name__ == "__main__":
    main() 