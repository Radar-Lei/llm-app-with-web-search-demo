"""
LLM App with Web Search
"""

import asyncio
import os
import tempfile
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import chromadb
import lmstudio as lms
import streamlit as st
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.models import CrawlResult
from duckduckgo_search import DDGS
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
    """Calls the LM Studio model with the given prompt and optional context.

    Args:
        prompt (str): The user prompt/question to send to the LLM
        with_context (bool, optional): Whether to include system context. Defaults to True.
        context (str | None, optional): Additional context to provide to the LLM. Defaults to None.

    Yields:
        str: Generated text chunks from the LLM response stream

    Returns:
        Generator[str, None, None]: A generator yielding response chunks
    """
    # 初始化LM Studio模型
    model = lms.llm("deepseek-r1-distill-qwen-32b")
    
    if with_context:
        # 创建一个聊天对象，包含系统提示
        chat = lms.Chat(system_prompt)
        # 添加用户消息
        chat.add_user_message(f"Context: {context}, Question: {prompt}")
    else:
        # 不使用系统上下文，直接创建聊天
        chat = lms.Chat()
        chat.add_user_message(prompt)

    # 使用流式响应
    prediction_stream = model.respond_stream(chat)
    
    for fragment in prediction_stream:
        yield fragment.content


def get_vector_collection() -> tuple[chromadb.Collection, chromadb.Client]:
    """Creates or retrieves a vector collection for storing embeddings.

    Creates an embedding function using HuggingFace and initializes a persistent ChromaDB client.
    Returns both the collection and client objects.

    Returns:
        tuple[chromadb.Collection, chromadb.Client]: A tuple containing:
            - The ChromaDB collection for storing embeddings
            - The ChromaDB client instance
    """
    hg_embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    chroma_client = chromadb.PersistentClient(
        path="./web-search-llm-db", settings=Settings(anonymized_telemetry=False)
    )
    return (
        chroma_client.get_or_create_collection(
            name="web_llm",
            embedding_function=hg_embeddings,
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
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", " ", ""],
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

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))


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
    if is_chinese(prompt):
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


def get_web_urls(search_term: str, num_results: int = 8) -> list[str]:
    """Performs a web search and returns filtered URLs.

    Uses DuckDuckGo search to find relevant web pages, excluding certain domains and checking
    robots.txt compliance.

    Args:
        search_term (str): The search query to use
        num_results (int, optional): Maximum number of search results to return. Defaults to 10.

    Returns:
        list[str]: List of URLs that are allowed to be crawled according to robots.txt

    Raises:
        Exception: If web search fails, prints error message and stops execution
    """
    try:
        discard_urls = ["youtube.com", "britannica.com", "vimeo.com"]
        for url in discard_urls:
            search_term += f" -site:{url}"

        results = DDGS().text(search_term, max_results=num_results)
        results = [result["href"] for result in results]

        return check_robots_txt(results)

    except Exception as e:
        error_msg = ("❌ Failed to fetch results from the web", str(e))
        print(error_msg)
        st.write(error_msg)
        st.stop()


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
            web_urls = get_web_urls(search_term=prompt)
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
