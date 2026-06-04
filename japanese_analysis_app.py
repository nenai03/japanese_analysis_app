# -*- coding: utf-8 -*-
"""
Streamlit版日语文本分析工具：文本清洗 + 依存分析 + MDD计算 + 多文件对比
适配Ginza模型（解决Streamlit Cloud权限/模型下载问题）
部署命令：streamlit run japanese_analysis_app.py
"""
import re
import os
import streamlit as st
import spacy
import ginza  # 必须导入Ginza，确保模型加载正常
import pandas as pd
from collections import Counter
from docx import Document
import numpy as np
from io import BytesIO
import time

# ===================== 1. 初始化配置 =====================
# 页面基础设置
st.set_page_config(
    page_title="日语文本分析工具",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载Ginza日语模型（核心修改：替换为ja_ginza，无权限问题）
@st.cache_resource(show_spinner="正在加载日语NLP模型...")
def load_spacy_model():
    try:
        # Ginza模型无需单独下载，pip install后直接加载
        nlp = spacy.load("ja_ginza")
        return nlp
    except Exception as e:
        st.error(f"Ginza模型加载失败：{str(e)}")
        st.info("请确认requirements.txt包含 ginza 和 ja-ginza，并重启应用")
        st.stop()

# 初始化模型（全局可用）
nlp = load_spacy_model()

# 初始化Session State（保存多文件数据、分析结果）
if "file_list" not in st.session_state:
    st.session_state.file_list = []  # 格式：[{name:文件名, content:内容, cleaned_content:清洗后内容, analysis:分析数据, mdd:值, stats:统计信息}, ...]
if "active_file_idx" not in st.session_state:
    st.session_state.active_file_idx = -1
if "cleaned_text" not in st.session_state:
    st.session_state.cleaned_text = ""

# ===================== 2. 核心功能函数（增强版） =====================
def remove_all_stars(text):
    """清除所有星号分隔符"""
    text = re.sub(r'[\*＊]+', '', text)
    text = re.sub(r'(\s+[\*＊]\s+)+', '', text)
    text = re.sub(r'[\*＊]', '', text)
    return text

def clean_aozora_format(text):
    """青空文库格式清洗（增强版）"""
    if not text:
        return ""
    
    # 基础清洗
    text = remove_all_stars(text)
    text = re.sub(r'｜', '', text)
    text = re.sub(r'《.*?》', '', text)
    text = re.sub(r'［＃.*?］', '', text)
    
    # 移除青空文库固定提示行
    remove_patterns = [
        r'この作品は縦書きでレイアウトされています。',
        r'また、ご覧になる機種により、表示の差異が認められることがあります。',
        r'一部の漢字が簡略字で表示されていることがあります。',
        r'〔＃.*〕',
        r'［＃.*］'
    ]
    for pat in remove_patterns:
        text = re.sub(pat, '', text)
    
    # 清理空行和首尾空格
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+$', '', text)
    
    return text.strip()

def keep_all_valid(text):
    """保留有效字符（用于字数统计）"""
    if not text:
        return ""
    # 保留日语字符、汉字、基本标点和换行
    pattern = r'[^\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u0020-\u007E\uFF00-\uFFEF\u2460-\u2473\n。，、！？；：""''()（）【】「」｛｝、・]'
    return re.sub(pattern, '', text)

def split_sentences_aozora(text):
    """智能断句（增强版）"""
    if not text:
        return ""
    
    # 保留引号内内容不被错误断句
    quotes = re.findall(r'「[^」]*」', text)
    temp_text = re.sub(r'「[^」]*」', '<<QUOTE>>', text)
    
    # 按句末标点断句
    temp_text = re.sub(r'([。！？])', r'\1\n', temp_text)
    
    # 还原引号内容
    for q in quotes:
        temp_text = temp_text.replace('<<QUOTE>>', q, 1)
    
    # 清理多余换行和空格
    temp_text = re.sub(r'\n+', '\n', temp_text)
    temp_text = re.sub(r'^\s+|\s+$', '', temp_text, flags=re.MULTILINE)
    temp_text = remove_all_stars(temp_text)
    
    # 过滤空行
    lines = [line.strip() for line in temp_text.split('\n') if line.strip()]
    return '\n'.join(lines)

def read_uploaded_file(uploaded_file):
    """读取Streamlit上传的文件（增强错误处理）"""
    file_name = uploaded_file.name
    try:
        if file_name.lower().endswith(".txt"):
            # 支持多种编码（适配日语文件）
            encodings = ['utf-8', 'shift_jis', 'euc-jp', 'iso-8859-1']
            content = ""
            for encoding in encodings:
                try:
                    content = uploaded_file.getvalue().decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            if not content:
                st.warning(f"文件{file_name}编码不支持（仅支持UTF-8/Shift_JIS/EUC-JP）")
                return "", ""
            return file_name, content
        
        elif file_name.lower().endswith(".docx"):
            doc = Document(BytesIO(uploaded_file.getvalue()))
            content = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return file_name, content
        
        else:
            st.warning(f"仅支持TXT/Word文件，跳过：{file_name}")
            return "", ""
    
    except Exception as e:
        st.error(f"读取文件{file_name}失败：{str(e)}")
        return "", ""

def calculate_mdd(analysis_data):
    """计算MDD（平均依存距离）"""
    if not analysis_data:
        return 0.0
    total_distance = 0
    valid_count = 0
    for row in analysis_data:
        dep_id = row[1]
        head_id = row[6]
        if dep_id != head_id and head_id > 0:  # 增加有效性校验
            total_distance += abs(dep_id - head_id)
            valid_count += 1
    return round(total_distance / valid_count, 4) if valid_count > 0 else 0.0

def analyze_text(content, progress_bar=None):
    """执行依存分析 + 基础统计（带进度条）"""
    if not content:
        return [], 0.0, {}
    
    # 分句
    lines = [line.rstrip("\n") for line in content.split("\n")]
    sentences = [line.strip() for line in lines if line.strip()]
    if not sentences:
        return [], 0.0, {}
    
    # 依存分析（Ginza和spaCy接口完全兼容，无需修改）
    analysis_data = []
    total_sents = len(sentences)
    
    for sentence_id, sent in enumerate(sentences, 1):
        # 更新进度条
        if progress_bar:
            progress_bar.progress(sentence_id / total_sents, text=f"分析句子 {sentence_id}/{total_sents}")
        
        doc = nlp(sent)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            row = [
                sentence_id,
                token.i + 1, token.text, token.lemma_, token.pos_, token.tag_,
                token.head.i + 1, token.head.text, token.head.lemma_, token.head.pos_, token.head.tag_, token.dep_
            ]
            analysis_data.append(row)
        time.sleep(0.01)  # 防止界面卡死
    
    # 计算MDD
    mdd_value = calculate_mdd(analysis_data)
    
    # 基础统计
    total_chars = 0
    total_sentences = len(sentences)
    total_tokens = 0
    token_lemmas = []
    lemma_tag_counter = Counter()
    
    for sent in sentences:
        clean_sent = re.sub(r'[\s。，、！？；：""''()（）【】「」｛｝、・]', '', sent)
        total_chars += len(clean_sent)
        doc = nlp(sent)
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            total_tokens += 1
            lemma = token.lemma_
            tag = token.tag_
            token_lemmas.append(lemma)
            lemma_tag_counter[(lemma, tag)] += 1
    
    avg_sent_len = round(total_chars / total_sentences, 2) if total_sentences > 0 else 0
    total_lemma_types = len(set(token_lemmas))
    
    stats = {
        "总字数": total_chars,
        "总句子数": total_sentences,
        "平均句长": avg_sent_len,
        "总词素数": total_tokens,
        "总词素种类数": total_lemma_types
    }
    
    return analysis_data, mdd_value, stats

def get_excel_bytes(df, sheet_name="结果"):
    """将DataFrame转为Excel字节流（用于下载）"""
    output = BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Excel生成失败：{e}")
        return BytesIO()

# ===================== 3. 网页界面布局 =====================
def main():
    st.title("📝 日语文本分析工具（Ginza版）")
    st.divider()
    
    # 侧边栏：文件上传 + 操作
    with st.sidebar:
        st.header("📂 文件管理")
        
        # 多文件上传
        uploaded_files = st.file_uploader(
            "上传TXT/Word文件（支持多文件）",
            type=["txt", "docx"],
            accept_multiple_files=True,
            help="支持UTF-8/Shift_JIS编码的TXT文件和Word文档"
        )
        
        # 上传文件处理
        if st.button("✅ 导入选中文件", type="primary") and uploaded_files:
            with st.spinner("正在导入文件..."):
                st.session_state.file_list = []
                success_count = 0
                for file in uploaded_files:
                    file_name, content = read_uploaded_file(file)
                    if content:
                        st.session_state.file_list.append({
                            "name": file_name,
                            "content": content,
                            "cleaned_content": "",
                            "analysis_data": [],
                            "mdd": 0.0,
                            "stats": {}
                        })
                        success_count += 1
                st.session_state.active_file_idx = 0 if success_count > 0 else -1
                st.success(f"成功导入 {success_count} 个文件！")
        
        # 文件列表选择
        if st.session_state.file_list:
            file_names = [f["name"] for f in st.session_state.file_list]
            selected_file = st.selectbox(
                "选择要分析的文件",
                file_names,
                index=st.session_state.active_file_idx if st.session_state.active_file_idx >=0 else 0
            )
            st.session_state.active_file_idx = file_names.index(selected_file)
            active_file = st.session_state.file_list[st.session_state.active_file_idx]
            
            # 操作按钮组
            st.divider()
            st.subheader("🔧 操作面板")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 单文件依存分析"):
                    progress_bar = st.progress(0, text="准备分析...")
                    with st.spinner("正在执行依存分析..."):
                        analysis_data, mdd, stats = analyze_text(active_file["content"], progress_bar)
                        st.session_state.file_list[st.session_state.active_file_idx]["analysis_data"] = analysis_data
                        st.session_state.file_list[st.session_state.active_file_idx]["mdd"] = mdd
                        st.session_state.file_list[st.session_state.active_file_idx]["stats"] = stats
                    progress_bar.empty()
                    st.success(f"「{selected_file}」分析完成！")
            
            with col2:
                if st.button("📊 批量依存分析"):
                    total_files = len(st.session_state.file_list)
                    progress_bar = st.progress(0, text="准备批量分析...")
                    with st.spinner("批量分析中..."):
                        processed = 0
                        for idx, f in enumerate(st.session_state.file_list):
                            progress_bar.progress((idx+1)/total_files, text=f"分析 {f['name']} ({idx+1}/{total_files})")
                            analysis_data, mdd, stats = analyze_text(f["content"])
                            st.session_state.file_list[idx]["analysis_data"] = analysis_data
                            st.session_state.file_list[idx]["mdd"] = mdd
                            st.session_state.file_list[idx]["stats"] = stats
                            processed += 1
                    progress_bar.empty()
                    st.success(f"批量分析完成！共处理 {processed} 个文件")
    
    # 主内容区：标签页切换
    tab1, tab2, tab3, tab4 = st.tabs(["📄 原始文本", "🧹 文本清洗", "📈 依存分析结果", "📊 MDD横向对比"])
    
    # 标签页1：原始文本
    with tab1:
        if st.session_state.file_list and st.session_state.active_file_idx >=0:
            active_file = st.session_state.file_list[st.session_state.active_file_idx]
            st.subheader(f"原始文本：{active_file['name']}")
            
            # 显示文本基本信息
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("原始字符数", len(active_file["content"]))
            with col2:
                st.metric("原始行数", len([l for l in active_file["content"].split("\n") if l.strip()]))
            with col3:
                file_type = "TXT" if active_file["name"].lower().endswith(".txt") else "Word"
                st.metric("文件类型", file_type)
            
            st.text_area(
                "内容",
                value=active_file["content"],
                height=500,
                key="raw_text"
            )
        else:
            st.info("💡 请先在侧边栏上传并导入TXT/Word文件")
    
    # 标签页2：文本清洗
    with tab2:
        if st.session_state.file_list and st.session_state.active_file_idx >=0:
            active_file = st.session_state.file_list[st.session_state.active_file_idx]
            st.subheader(f"文本清洗：{active_file['name']}")
            
            # 执行清洗
            if st.button("🧹 执行清洗", type="primary"):
                with st.spinner("正在清洗文本..."):
                    cleaned_text = clean_aozora_format(active_file["content"])
                    cleaned_text = keep_all_valid(cleaned_text)
                    cleaned_text = split_sentences_aozora(cleaned_text)
                    st.session_state.cleaned_text = cleaned_text
                    # 保存到文件数据中
                    st.session_state.file_list[st.session_state.active_file_idx]["cleaned_content"] = cleaned_text
                
                # 清洗结果展示
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("清洗后字符数", len(cleaned_text))
                with col2:
                    st.metric("清洗后句子数", len([l for l in cleaned_text.split("\n") if l.strip()]))
                with col3:
                    reduction = round((1 - len(cleaned_text)/len(active_file["content"]))*100, 2) if len(active_file["content"]) > 0 else 0
                    st.metric("冗余内容去除率", f"{reduction}%")
                
                st.text_area(
                    "清洗后内容",
                    value=cleaned_text,
                    height=400,
                    key="cleaned_text_area"
                )
                
                # 下载清洗结果
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 下载TXT格式",
                        data=cleaned_text,
                        file_name=f"{os.path.splitext(active_file['name'])[0]}_清洗后.txt",
                        mime="text/plain",
                        type="primary"
                    )
                with col2:
                    # 转为Word字节流
                    doc = Document()
                    for line in cleaned_text.split("\n"):
                        if line.strip():
                            doc.add_paragraph(line)
                    doc_bytes = BytesIO()
                    doc.save(doc_bytes)
                    doc_bytes.seek(0)
                    st.download_button(
                        label="💾 下载Word格式",
                        data=doc_bytes,
                        file_name=f"{os.path.splitext(active_file['name'])[0]}_清洗后.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
            else:
                # 显示上次清洗结果（如果有）
                if active_file.get("cleaned_content"):
                    st.text_area(
                        "清洗后内容（上次结果）",
                        value=active_file["cleaned_content"],
                        height=400,
                        key="prev_cleaned_text"
                    )
                else:
                    st.info("💡 点击「执行清洗」按钮开始文本清洗")
        else:
            st.info("💡 请先在侧边栏上传并导入文件")
    
    # 标签页3：依存分析结果
    with tab3:
        if st.session_state.file_list and st.session_state.active_file_idx >=0:
            active_file = st.session_state.file_list[st.session_state.active_file_idx]
            st.subheader(f"依存分析结果：{active_file['name']}")
            
            if active_file["analysis_data"]:
                # 转换为DataFrame展示（Ginza分析结果和原spaCy格式一致）
                analysis_header = [
                    "句子ID", "从属ID", "从属词", "词原型", "从属词性", "从属词性细标",
                    "核心ID", "核心词", "核心词原型", "核心词性", "核心词性细标", "依存关系"
                ]
                df_analysis = pd.DataFrame(active_file["analysis_data"], columns=analysis_header)
                
                # 展示表格（支持分页、筛选）
                st.dataframe(
                    df_analysis,
                    use_container_width=True,
                    height=500
                )
                
                # 下载分析结果
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 下载Excel格式",
                        data=get_excel_bytes(df_analysis, "依存分析结果"),
                        file_name=f"{os.path.splitext(active_file['name'])[0]}_依存分析结果.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
                with col2:
                    # TXT格式下载
                    txt_data = "\t".join(analysis_header) + "\n"
                    for row in active_file["analysis_data"]:
                        txt_data += "\t".join(map(str, row)) + "\n"
                    st.download_button(
                        label="💾 下载TXT格式",
                        data=txt_data,
                        file_name=f"{os.path.splitext(active_file['name'])[0]}_依存分析结果.txt",
                        mime="text/plain"
                    )
                
                # 展示基础统计
                st.divider()
                st.subheader("📈 文本基础统计")
                stats = active_file["stats"]
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("总字数", stats.get("总字数", 0))
                col2.metric("总句子数", stats.get("总句子数", 0))
                col3.metric("平均句长", stats.get("平均句长", 0))
                col4.metric("总词素数", stats.get("总词素数", 0))
                col5.metric("MDD值", active_file["mdd"])
            else:
                st.warning(f"⚠️「{active_file['name']}」尚未执行依存分析，请在侧边栏点击「单文件依存分析」按钮")
        else:
            st.info("💡 请先在侧边栏上传并导入文件")
    
    # 标签页4：MDD横向对比
    with tab4:
        st.subheader("多文件MDD横向对比")
        if st.session_state.file_list and any(f["mdd"] > 0 for f in st.session_state.file_list):
            # 整理对比数据
            compare_data = []
            for f in st.session_state.file_list:
                stats = f["stats"]
                compare_data.append({
                    "文件名": f["name"],
                    "总句子数": stats.get("总句子数", 0),
                    "总词数": stats.get("总词素数", 0),
                    "总依存关系数": len(f["analysis_data"]),
                    "平均依存距离(MDD)": f["mdd"],
                    "平均句长": stats.get("平均句长", 0)
                })
            df_compare = pd.DataFrame(compare_data)
            
            # 展示对比表格
            st.dataframe(
                df_compare,
                use_container_width=True,
                height=400
            )
            
            # 下载对比结果
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 下载Excel格式",
                    data=get_excel_bytes(df_compare, "MDD横向对比"),
                    file_name="多文件MDD对比结果.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
            with col2:
                # TXT格式
                txt_data = "\t".join(df_compare.columns) + "\n"
                for _, row in df_compare.iterrows():
                    txt_data += "\t".join(map(str, row.values)) + "\n"
                st.download_button(
                    label="💾 下载TXT格式",
                    data=txt_data,
                    file_name="多文件MDD对比结果.txt",
                    mime="text/plain"
                )
            
            # MDD可视化
            st.divider()
            st.subheader("📊 MDD值可视化对比")
            chart_data = df_compare.set_index("文件名")["平均依存距离(MDD)"]
            st.bar_chart(
                chart_data,
                y_label="平均依存距离(MDD)",
                x_label="文件名",
                use_container_width=True,
                color="#1f77b4"
            )
            
            # MDD统计汇总
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("平均MDD值", round(df_compare["平均依存距离(MDD)"].mean(), 4))
            with col2:
                st.metric("最高MDD值", df_compare["平均依存距离(MDD)"].max())
            with col3:
                st.metric("最低MDD值", df_compare["平均依存距离(MDD)"].min())
                
        else:
            st.info("💡 请先执行「单文件依存分析」或「批量依存分析」，再查看MDD对比")

if __name__ == "__main__":
    main()
