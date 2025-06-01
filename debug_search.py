#!/usr/bin/env python3
"""
调试脚本：测试中英文web搜索的差异
"""

import asyncio
import re
from app import get_web_urls, crawl_webpages, add_to_vector_database, get_vector_collection

async def debug_search():
    """调试搜索功能"""
    
    # 测试查询
    english_query = "who win 2025 grammy prize"
    chinese_query = "哪些人获得了2025年格莱美奖"
    
    for query_name, query in [("英文", english_query), ("中文", chinese_query)]:
        print(f"\n{'='*50}")
        print(f"测试 {query_name} 查询: {query}")
        print(f"{'='*50}")
        
        # 1. 测试URL获取
        print(f"1. 获取搜索URL...")
        try:
            urls = await get_web_urls(search_term=query, num_results=5)
            print(f"   找到 {len(urls)} 个URL:")
            for i, url in enumerate(urls[:3], 1):  # 只显示前3个
                print(f"   {i}. {url}")
        except Exception as e:
            print(f"   错误: {e}")
            continue
        
        if not urls:
            print(f"   {query_name}查询未找到URL，跳过后续测试")
            continue
            
        # 2. 测试网页爬取
        print(f"2. 爬取网页内容...")
        try:
            results = await crawl_webpages(urls=urls[:3], prompt=query)  # 只爬取前3个
            print(f"   爬取了 {len(results)} 个页面")
            
            content_count = 0
            for i, result in enumerate(results, 1):
                if result.markdown_v2 and result.markdown_v2.fit_markdown:
                    content_length = len(result.markdown_v2.fit_markdown)
                    print(f"   页面 {i}: 内容长度 {content_length} 字符")
                    if content_length > 0:
                        content_count += 1
                        # 显示内容样本
                        sample = result.markdown_v2.fit_markdown[:200] + "..."
                        print(f"   内容样本: {sample}")
                else:
                    print(f"   页面 {i}: 无内容")
            
            print(f"   有效内容页面: {content_count}/{len(results)}")
                    
        except Exception as e:
            print(f"   爬取错误: {e}")
            continue
        
        # 3. 测试向量数据库
        print(f"3. 测试向量数据库...")
        try:
            collection, chroma_client = get_vector_collection()
            
            # 清空之前的数据
            try:
                chroma_client.delete_collection(name="web_llm")
                collection, chroma_client = get_vector_collection()
            except:
                pass
            
            add_to_vector_database(results)
            
            # 查询向量数据库
            qresults = collection.query(query_texts=[query], n_results=5)
            context_docs = qresults.get("documents")[0] if qresults.get("documents") else []
            
            print(f"   向量查询结果: {len(context_docs)} 个文档片段")
            if context_docs:
                total_length = sum(len(doc) for doc in context_docs)
                print(f"   总上下文长度: {total_length} 字符")
                
                # 显示第一个文档片段的样本
                if context_docs[0]:
                    sample = context_docs[0][:200] + "..."
                    print(f"   第一个片段样本: {sample}")
            else:
                print(f"   ⚠️ 警告: {query_name}查询没有找到相关文档!")
                
                # 尝试查看数据库中所有文档
                all_docs = collection.get()
                print(f"   数据库中总文档数: {len(all_docs['documents']) if all_docs['documents'] else 0}")
                
            # 清理
            chroma_client.delete_collection(name="web_llm")
            
        except Exception as e:
            print(f"   向量数据库错误: {e}")
        
        print(f"{query_name} 查询测试完成")

if __name__ == "__main__":
    asyncio.run(debug_search()) 