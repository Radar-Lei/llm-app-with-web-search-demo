"""
LLM App with Web Search
"""

import asyncio
import os
import tempfile
import re
from urllib.parse import urlparse, quote_plus
from urllib.robotparser import RobotFileParser

import chromadb
import ollama
import streamlit as st
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.
Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. When the context supports an answer, ensure your response is clear, concise, and directly addresses the question.
5. When there is no context, just say you have no context and stop immediately.
6. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.
7. Avoid explaining why you cannot answer or speculating about missing details. Simply state that you lack sufficient context when necessary.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.
6. Do not mention what you received in context, just focus on answering based on the context.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def call_llm(prompt: str, with_context: bool = True, context: str | None = None):
    """Calls the LLM model with the given prompt and optional context.

    Args:
        prompt (str): The user prompt/question to send to the LLM
        with_context (bool, optional): Whether to include system context. Defaults to True.
        context (str | None, optional): Additional context to provide to the LLM. Defaults to None.

    Yields:
        str: Generated text chunks from the LLM response stream

    Returns:
        Generator[str, None, None]: A generator yielding response chunks
    """
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"Context: {context}, Question: {prompt}",
        },
    ]

    if not with_context:
        messages.pop(0)
        messages[0]["content"] = prompt

    response = ollama.chat(model="deepseek-r1:32b", stream=True, messages=messages)

    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    """Creates or retrieves a vector collection for storing embeddings.

    Creates an embedding function using Ollama and initializes a persistent ChromaDB client.
    Returns both the collection and client objects.

    Returns:
        tuple[chromadb.Collection, chromadb.Client]: A tuple containing:
            - The ChromaDB collection for storing embeddings
            - The ChromaDB client instance
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="paraphrase-multilingual:latest",
    )

    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db", settings=Settings(anonymized_telemetry=False)
    )
    return (
        chroma_client.get_or_create_collection(
            name="web_llm",
            embedding_function=ollama_ef,
            metadata={"hnsw:space": "cosine"},
        ),
        chroma_client,
    )


def normalize_url(url):
    """Normalizes a URL by removing common prefixes and replacing special characters.

    Args:
        url (str): The URL to normalize.

    Returns:
        str: The normalized URL with https://, www. removed and /, -, . replaced with underscores.

    Example:
        >>> normalize_url("https://www.example.com/path")
        "example_com_path"
    """
    normalized_url = (
        url.replace("https://", "")
        .replace("www.", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    print("Normalized URL", normalized_url)
    return normalized_url


def add_to_vector_database(results: list[CrawlResult]):
    """Adds crawl results to a vector database for semantic search.

    Takes a list of crawl results, processes the markdown content by splitting it into chunks,
    and stores the chunks in a ChromaDB vector collection with associated metadata.

    Args:
        results (list[CrawlResult]): List of crawl results containing markdown content and URLs

    Returns:
        None

    Note:
        - Uses RecursiveCharacterTextSplitter to split markdown into chunks
        - Creates temporary markdown files for processing
        - Normalizes URLs for use as document IDs
        - Upserts documents, metadata and IDs to ChromaDB collection
    """
    collection, _ = get_vector_collection()

    for result in results:
        documents, metadatas, ids = [], [], []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        if result.markdown_v2:
            markdown_result = result.markdown_v2.fit_markdown
        else:
            continue

        temp_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
        temp_file.write(markdown_result)
        temp_file.flush()

        loader = UnstructuredMarkdownLoader(temp_file.name, mode="single")
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        os.unlink(temp_file.name)  # Delete the temporary file

        normalized_url = normalize_url(result.url)

        if all_splits:
            for idx, split in enumerate(all_splits):
                documents.append(split.page_content)
                metadatas.append({"source": result.url})
                ids.append(f"{normalized_url}_{idx}")

            print("Upsert collection: ", id(collection))
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )


async def crawl_webpages(urls: list[str], prompt: str) -> CrawlResult:
    """Asynchronously crawls multiple webpages and extracts relevant content based on a prompt.

    Args:
        urls (list[str]): List of URLs to crawl
        prompt (str): Query text used to filter relevant content from the pages

    Returns:
        CrawlResult: Results from crawling containing filtered markdown content and metadata

    Note:
        Uses BM25 content filtering to extract relevant sections based on the prompt.
        Configures crawler to exclude navigation elements, forms, images etc.
        Runs in headless browser mode with text-only extraction.
    """
    if re.search(r'[\u4e00-\u9fff]', prompt):
        bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=0.8)
    else:
        bm25_filter = BM25ContentFilter(user_query=prompt, bm25_threshold=1.2)
        
    md_generator = DefaultMarkdownGenerator(content_filter=bm25_filter)

    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "form", "img", "a"],
        only_text=True,
        exclude_social_media_links=True,
        keep_data_attributes=False,
        cache_mode=CacheMode.BYPASS,
        remove_overlay_elements=True,
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
        page_timeout=20000,  # in ms: 20 seconds
    )
    browser_config = BrowserConfig(headless=True, text_mode=True, light_mode=True)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls, config=crawler_config)
        return results


def check_robots_txt(urls: list[str]) -> list[str]:
    """Checks robots.txt files to determine which URLs are allowed to be crawled.

    Args:
        urls (list[str]): List of URLs to check against their robots.txt files.

    Returns:
        list[str]: List of URLs that are allowed to be crawled according to robots.txt rules.
            If a robots.txt file is missing or there's an error, the URL is assumed to be allowed.
    """
    allowed_urls = []

    for url in urls:
        try:
            robots_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt"
            rp = RobotFileParser(robots_url)
            rp.read()

            if rp.can_fetch("*", url):
                allowed_urls.append(url)

        except Exception:
            # If robots.txt is missing or there's any error, assume URL is allowed
            allowed_urls.append(url)

    return allowed_urls


async def get_web_urls(search_term: str, num_results: int = 10) -> list[str]:
    """使用 crawl4ai 通过 Bing 搜索并返回过滤后的 URL。

    使用无头浏览器访问 Bing 搜索页面，解析搜索结果并提取链接，
    排除某些域名并检查 robots.txt 合规性。

    Args:
        search_term (str): 要使用的搜索查询
        num_results (int, optional): 返回的最大搜索结果数量。默认为 10。

    Returns:
        list[str]: 根据 robots.txt 允许爬取的 URL 列表

    Raises:
        Exception: 如果网络搜索失败，打印错误消息并停止执行
    """
    try:
        # 构建 Bing 搜索 URL
        discard_domains = ["youtube.com", "britannica.com", "vimeo.com"]
        query = search_term
        # 检查是否包含中文字符
        if not re.search(r'[\u4e00-\u9fff]', query):
            for domain in discard_domains:
                query += f" -site:{domain}"
        
        encoded_query = quote_plus(query)
        search_url = f"https://www.bing.com/search?q={encoded_query}&count={num_results}"
        
        print(f"正在搜索: {search_url}")
        
        # 配置爬虫
        crawler_config = CrawlerRunConfig(
            excluded_tags=["nav", "footer", "header", "form", "img"],
            only_text=False,  # 我们需要保留链接
            exclude_social_media_links=False,
            keep_data_attributes=False,
            cache_mode=CacheMode.BYPASS,
            remove_overlay_elements=True,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            page_timeout=30000,  # 30 秒超时
            delay_before_return_html=3.0,  # 等待页面加载
        )
        
        browser_config = BrowserConfig(
            headless=True, 
            text_mode=False,  # 需要 HTML 来提取链接
            light_mode=True
        )
        
        # 爬取搜索结果页面
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=search_url, config=crawler_config)
            
            if not result.html:
                raise Exception("无法获取搜索结果页面的 HTML 内容")
            
            # 解析搜索结果链接
            urls = extract_bing_links(result.html, num_results)
            
            if not urls:
                print("未找到搜索结果链接")
                return []
            
            print(f"找到 {len(urls)} 个搜索结果")
            
            # 检查 robots.txt 合规性
            return check_robots_txt(urls)
            
    except Exception as e:
        error_msg = f"❌ 从网络获取结果失败: {str(e)}"
        print(error_msg)
        st.write(error_msg)
        st.stop()


def extract_bing_links(html: str, max_results: int = 10) -> list[str]:
    """从 Bing 搜索结果 HTML 中提取链接。

    Args:
        html (str): Bing 搜索结果页面的 HTML 内容
        max_results (int): 提取的最大链接数量

    Returns:
        list[str]: 提取的 URL 列表
    """
    from urllib.parse import unquote
    
    urls = []
    
    # 模式 1: Bing 搜索结果链接 - 查找标题链接
    # Bing 通常在 h2 标签中包含结果标题链接
    title_pattern = r'<h2[^>]*><a[^>]*href=["\']([^"\']+)["\']'
    title_matches = re.findall(title_pattern, html, re.IGNORECASE)
    
    for href in title_matches:
        try:
            if href.startswith(('http://', 'https://')):
                # 排除 Bing 和 Microsoft 相关的域名
                exclude_domains = ['bing.com', 'microsoft.com', 'msn.com', 'live.com']
                if not any(domain in href for domain in exclude_domains):
                    urls.append(href)
                    if len(urls) >= max_results:
                        break
        except Exception:
            continue
    
    # 模式 2: 查找所有以 http 开头的链接，但排除导航和广告链接
    general_pattern = r'href=["\']([^"\']*(?:https?://[^"\']+))["\']'
    general_matches = re.findall(general_pattern, html)
    
    exclude_domains = ['bing.com', 'microsoft.com', 'msn.com', 'live.com', 'microsofttranslator.com']
    exclude_keywords = ['privacy', 'terms', 'help', 'support', 'feedback', 'advertising']
    
    for href in general_matches:
        try:
            if href.startswith(('http://', 'https://')):
                # 排除特定域名和关键词
                if (not any(domain in href for domain in exclude_domains) and
                    not any(keyword in href.lower() for keyword in exclude_keywords)):
                    
                    # 确保链接看起来是真实的内容页面
                    if len(href) > 20 and '.' in href:  # 基本的有效性检查
                        urls.append(href)
                        if len(urls) >= max_results:
                            break
        except Exception:
            continue
    
    # 模式 3: 查找特定的结果链接类
    result_link_pattern = r'<cite[^>]*>([^<]+)</cite>'
    cite_matches = re.findall(result_link_pattern, html, re.IGNORECASE)
    
    for cite_text in cite_matches:
        try:
            # 从 cite 标签的文本中提取 URL
            if cite_text.startswith(('http://', 'https://')):
                exclude_domains = ['bing.com', 'microsoft.com', 'msn.com', 'live.com']
                if not any(domain in cite_text for domain in exclude_domains):
                    urls.append(cite_text.split(' ')[0])  # 取第一个可能的 URL
                    if len(urls) >= max_results:
                        break
            elif '/' in cite_text and '.' in cite_text:
                # 可能是省略了协议的 URL
                full_url = f"https://{cite_text.split(' ')[0]}"
                exclude_domains = ['bing.com', 'microsoft.com', 'msn.com', 'live.com']
                if not any(domain in full_url for domain in exclude_domains):
                    urls.append(full_url)
                    if len(urls) >= max_results:
                        break
        except Exception:
            continue
    
    # 去重
    unique_urls = []
    seen = set()
    for url in urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)
    
    print(f"提取到 {len(unique_urls)} 个唯一链接")
    return unique_urls[:max_results]


async def run():
    st.set_page_config(page_title="LLM with Web Search")

    st.header("🔍 LLM Web Search")
    prompt = st.text_area(
        label="Put your query here",
        placeholder="Add your query...",
        label_visibility="hidden",
    )
    is_web_search = st.toggle("Enable web search", value=False, key="enable_web_search")
    go = st.button(
        "⚡️ Go",
    )

    collection, chroma_client = get_vector_collection()

    if prompt and go:
        if is_web_search:
            web_urls = await get_web_urls(search_term=prompt)
            if not web_urls:
                st.write("No results found.")
                st.stop()

            results = await crawl_webpages(urls=web_urls, prompt=prompt)
            add_to_vector_database(results)

            qresults = collection.query(query_texts=[prompt], n_results=10)
            context = qresults.get("documents")[0]

            chroma_client.delete_collection(
                name="web_llm"
            )  # Delete collection after use

            llm_response = call_llm(
                context=context, prompt=prompt, with_context=is_web_search
            )
            st.write_stream(llm_response)
        else:
            llm_response = call_llm(prompt=prompt, with_context=is_web_search)
            st.write_stream(llm_response)


if __name__ == "__main__":
    asyncio.run(run())
