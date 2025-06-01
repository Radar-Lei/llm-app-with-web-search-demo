import os
import sys
import json
import base64
import re
from pathlib import Path
import markdown
from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import fitz  # PyMuPDF库用于提取PDF图片
import uuid
import shutil
import argparse


class PdfImageExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        
    def extract_images(self, output_folder):
        """从PDF提取所有图片并保存到指定文件夹"""
        os.makedirs(output_folder, exist_ok=True)
        
        image_paths = []
        try:
            pdf_document = fitz.open(self.pdf_path)
            
            for page_index in range(len(pdf_document)):
                page = pdf_document[page_index]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # 创建唯一的图像名称
                    image_filename = f"page{page_index+1}_img{img_index+1}.{image_ext}"
                    image_path = os.path.join(output_folder, image_filename)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_paths.append({
                        "path": image_path,
                        "filename": image_filename,
                        "page": page_index
                    })
                    
            pdf_document.close()
            return image_paths
            
        except Exception as e:
            print(f"提取图片时出错: {str(e)}")
            return []


class MistralOCRProcessor:
    def __init__(self, api_key, model="mistral-ocr-latest"):
        self.api_key = api_key
        self.model = model
        self.client = Mistral(api_key=api_key)
        
    def process_pdf(self, pdf_path, output_dir, output_format="markdown"):
        """处理PDF文件并返回OCR结果"""
        try:
            pdf_file = Path(pdf_path)
            
            # 创建临时图片文件夹
            temp_image_folder = os.path.join(os.path.dirname(pdf_path), f"temp_images_{uuid.uuid4().hex}")
            os.makedirs(temp_image_folder, exist_ok=True)
            
            # 提取PDF中的图片
            print("正在提取PDF中的图片...")
            extractor = PdfImageExtractor(pdf_path)
            extracted_images = extractor.extract_images(temp_image_folder)
            print(f"提取了 {len(extracted_images)} 张图片")
            
            # 上传文件
            print("上传文件中...")
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )
            
            signed_url = self.client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            
            # OCR处理
            print("OCR处理中...")
            pdf_response = self.client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model=self.model,
                include_image_base64=False,  # 降低API费用
            )
            
            response_dict = json.loads(pdf_response.model_dump_json())
            
            # 保存结果
            self.save_results(response_dict, extracted_images, output_dir, output_format)
            
            # 清理
            try:
                self.client.files.delete(file_id=uploaded_file.id)
                print("临时文件已删除")
            except Exception as e:
                print(f"警告: 无法删除临时文件: {str(e)}")
            
            # 清理临时图片文件夹
            try:
                if temp_image_folder and os.path.exists(temp_image_folder):
                    shutil.rmtree(temp_image_folder, ignore_errors=True)
                    print("临时图片文件夹已清理")
            except Exception as e:
                print(f"警告: 清理临时文件时出错: {str(e)}")
                
            return response_dict, extracted_images
            
        except Exception as e:
            print(f"处理过程中出错: {str(e)}")
            return None, None
    
    def save_results(self, response_data, extracted_images, output_dir, output_format):
        """保存OCR结果"""
        if not response_data:
            print("没有要保存的结果")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图片文件夹
        images_folder = os.path.join(output_dir, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 复制已提取的图片到目标文件夹
        for img_info in extracted_images:
            try:
                src_path = img_info["path"]
                dest_path = os.path.join(images_folder, img_info["filename"])
                shutil.copy2(src_path, dest_path)
            except Exception as e:
                print(f"复制图片时出错: {str(e)}")
        
        # 确定正确的文件扩展名
        if output_format == "markdown":
            file_ext = "md"  # 对markdown使用.md扩展名
        else:
            file_ext = output_format  # html和json保持不变
        
        output_file = os.path.join(output_dir, f"ocr_result.{file_ext}")
        
        # 收集每页中的所有图片ID和原始markdown
        page_markdowns = []
        image_ids_by_page = []
        
        for page in response_data.get("pages", []):
            # 收集这个页面上的所有图片ID
            page_img_ids = []
            for img in page.get("images", []):
                if "id" in img:
                    page_img_ids.append(img["id"])
            
            image_ids_by_page.append(page_img_ids)
            page_markdowns.append(page.get("markdown", ""))
        
        # 处理每页的markdown，替换图片引用
        updated_markdowns = []
        
        for page_idx, (page_md, page_img_ids) in enumerate(zip(page_markdowns, image_ids_by_page)):
            updated_md = page_md
            
            # 找出对应这个页面的提取图片
            page_extracted_images = [img for img in extracted_images if img["page"] == page_idx]
            
            # 为这个页面的每个图片ID创建替换
            for img_idx, img_id in enumerate(page_img_ids):
                if img_idx < len(page_extracted_images):
                    # 获取对应的提取图片文件名
                    img_filename = page_extracted_images[img_idx]["filename"]
                    new_img_path = f"images/{img_filename}"
                    
                    # 查找并替换markdown中的图片引用
                    pattern = r'!\[(.*?)\]\(' + re.escape(img_id) + r'\)'
                    replacement = r'![\1](' + new_img_path + r')'
                    updated_md = re.sub(pattern, replacement, updated_md)
                    
                    # 打印调试信息
                    print(f"替换页面 {page_idx+1} 图片: {img_id} -> {new_img_path}")
            
            updated_markdowns.append(updated_md)
        
        # 连接所有更新后的markdown内容
        markdown_text = "\n\n".join(updated_markdowns)
        
        if output_format == "html":
            # 将markdown转换为HTML
            md = markdown.Markdown(extensions=["tables"])
            html_content = md.convert(markdown_text)
            
            # 添加HTML包装
            result = f"""<!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>OCR Result</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0 auto;
                        max-width: 800px;
                        padding: 20px;
                    }}
                    img {{ max-width: 100%; height: auto; }}
                    h1, h2, h3 {{ margin-top: 1.5em; }}
                    p {{ margin: 1em 0; }}
                </style>
            </head>
            <body>
            {html_content}
            <hr>
            <p style="text-align: right; color: #666; font-size: 0.8em;">
                Generated by Mistral OCR CLI | by Lei Da (David) | greatradar@gmail.com
            </p>
            </body>
            </html>"""
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
        elif output_format == "json":
            # 为JSON添加提取的图片路径信息
            json_data = response_data.copy()
            # 在JSON中也更新图片路径
            for page_idx, page in enumerate(json_data.get("pages", [])):
                page_extracted_images = [img for img in extracted_images if img["page"] == page_idx]
                for img_idx, img in enumerate(page.get("images", [])):
                    if img_idx < len(page_extracted_images):
                        img_filename = page_extracted_images[img_idx]["filename"]
                        img["local_path"] = f"images/{img_filename}"
            
            # 添加元数据信息，包括作者信息
            json_data["metadata"] = {
                "app": "Mistral OCR CLI",
                "author": "Lei Da (David)",
                "contact": "greatradar@gmail.com"
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
        else:  # markdown
            # 在markdown文件末尾添加作者信息
            markdown_text += "\n\n---\n\n*Generated by Mistral OCR CLI | by Lei Da (David) | greatradar@gmail.com*"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
        
        print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Mistral OCR 命令行工具")
    parser.add_argument("pdf_path", help="要处理的PDF文件路径")
    parser.add_argument("-o", "--output", default="./ocr_output", help="输出目录 (默认: ./ocr_output)")
    parser.add_argument("-f", "--format", choices=["markdown", "html", "json"], 
                       default="markdown", help="输出格式 (默认: markdown)")
    parser.add_argument("-k", "--api-key", help="Mistral API Key (或使用环境变量 MISTRAL_API_KEY)")
    parser.add_argument("-m", "--model", default="mistral-ocr-latest", help="OCR模型 (默认: mistral-ocr-latest)")
    
    args = parser.parse_args()
    
    # 检查PDF文件是否存在
    if not os.path.exists(args.pdf_path):
        print(f"错误: 文件 '{args.pdf_path}' 不存在")
        sys.exit(1)
    
    if not args.pdf_path.lower().endswith('.pdf'):
        print("错误: 请提供一个PDF文件")
        sys.exit(1)
    
    # 获取API Key
    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("错误: 请提供Mistral API Key，使用 -k 参数或设置环境变量 MISTRAL_API_KEY")
        sys.exit(1)
    
    print("="*50)
    print("Mistral OCR CLI")
    print("by Lei Da (David) | greatradar@gmail.com")
    print("="*50)
    print(f"PDF文件: {args.pdf_path}")
    print(f"输出目录: {args.output}")
    print(f"输出格式: {args.format}")
    print(f"模型: {args.model}")
    print("="*50)
    
    # 创建处理器并处理PDF
    processor = MistralOCRProcessor(api_key, args.model)
    response_data, extracted_images = processor.process_pdf(
        args.pdf_path, args.output, args.format
    )
    
    if response_data:
        print("="*50)
        print("处理完成!")
        print(f"提取了 {len(extracted_images)} 张图片")
        print(f"处理了 {len(response_data.get('pages', []))} 页")
        print(f"结果保存在: {args.output}")
        print("="*50)
    else:
        print("处理失败")
        sys.exit(1)


if __name__ == "__main__":
    main() 