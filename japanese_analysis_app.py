# -*- coding: utf-8 -*-
"""
Streamlit版日语文本分析工具：文本清洗 + 依存分析 + MDD计算 + 多文件对比
部署命令：streamlit run japanese_analysis_app.py
"""
import re
import os
import streamlit as st
import spacy
import pandas as pd
from collections import Counter
from docx import Document
import numpy as np
from io import BytesIO

# ===================== 1. 初始化配置 =====================
# 页面基础设置
st.set_page_config(
    page_title="日语文本分析工具",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载日语spaCy模型（首次运行自动下载，需联网）
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("ja_core_news_sm")
        return nlp
    except Exception as e:
        st.error(f"日语模型加载失败：{e}")
        st.info("请先在终端执行：python -m spacy download ja_core_news_sm")
        st.stop()
nlp = load_spacy_model()

# 初始化Session State（保存多文件数据、分析结果）
if "file_list" not in st.session_state:
    st.session_state.file_list = []  # 格式：[{name:文件名, content:内容, analysis:分析数据, mdd:值, stats:统计信息}, ...]
if "active_file_idx" not in st.session_state:
    st.session_state.active_file_idx = -1

# ===================== 2. 核心功能函数（复用原有逻辑） =====================
def remove_all_stars(text):
    """清除所有星号分隔符"""
    text = re.sub(r'[\*＊]+', '', text)
    text = re.sub(r'(\s+[\*＊]\s+)+', '', text)
    text = re.sub(r'[\*＊]', '', text)
    return text

def clean_aozora_format(text):
    """青空文库格式清洗"""
    text = remove_all_stars(text)
    text = re.sub(r'｜', '', text)
    text = re.sub(r'《.*?》', '', text)
    text = re.sub(r'［＃.*?］', '', text)
    remove_lines = [
        r'この作品は縦書きでレイアウトされています。',
        r'また、ご覧になる機種により、表示の差異が認められることがあります。',
        r'一部の漢字が簡略字で表示されていることがあります。'
    ]
    for pat in remove_lines:
        text = re.sub(pat, '', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def keep_all_valid(text):
    """保留有效字符（用于字数统计）"""
    pattern = r'[^\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u0020-\u007E\uFF00-\uFFEF\u2460-\u2473\n]'
    return re.sub(pattern, '', text)

def split_sentences_aozora(text):
    """智能断句"""
    if not text:
        return ""
    quotes = re.findall(r'「[^」]*」', text)
    temp_text = re.sub(r'「[^」]*」', '<<QUOTE>>', text)
    temp_text = re.sub(r'([。！？])', r'\1\n', temp_text)
    for q in quotes:
        temp_text = temp_text.replace('<<QUOTE>>', q, 1)
    temp_text = re.sub(r'\n+', '\n', temp_text)
    temp_text = re.sub(r'^\s+|\s+$', '', temp_text, flags=re.MULTILINE)
    temp_text = remove_all_stars(temp_text)
    lines = [line.strip() for line in temp_text.split('\n') if line.strip()]
    return '\n'.join(lines)

def read_uploaded_file(uploaded_file):
    """读取Streamlit上传的文件"""
    file_name = uploaded_file.name
    try:
        if file_name.lower().endswith(".txt"):
            content = uploaded_file.getvalue().decode("utf-8")
            return file_name, content
        elif file_name.lower().endswith(".docx"):
            doc = Document(BytesIO(uploaded_file.getvalue()))
            content = "\n".join([para.text for para in doc.paragraphs])
            return file_name, content
        else:
            st.warning(f"仅支持TXT/Word文件，跳过：{file_name}")
            return "", ""
    except Exception as e:
        st.error(f"读取文件{file_name}失败：{e}")
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
        if dep_id != head_id:
            total_distance += abs(dep_id - head_id)
            valid_count += 1
    return total_distance / valid_count if valid_count > 0 else 0.0

def analyze_text(content):
    """执行依存分析 + 基础统计"""
    if not content:
        return [], 0.0, {}
    
    # 分句
    lines = [line.rstrip("\n") for line in content.split("\n")]
    sentences = [line.strip() for line in lines if line.strip()]
    if not sentences:
        return [], 0.0, {}
    
    # 依存分析
    analysis_data = []
    for sentence_id, sent in enumerate(sentences, 1):
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
    
    avg_sent_len = total_chars / total_sentences if total_sentences > 0 else 0
    total_lemma_types = len(set(token_lemmas))
    
    stats = {
        "总字数": total_chars,
        "总句子数": total_sentences,
        "平均句长": round(avg_sent_len, 2),
        "总词素数": total_tokens,
        "总词素种类数": total_lemma_types
    }
    
    return analysis_data, mdd_value, stats

def get_excel_bytes(df, sheet_name="结果"):
    """将DataFrame转为Excel字节流（用于下载）"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

# ===================== 3. 网页界面布局 =====================
def main():
    st.title("📝 日语文本分析工具（Streamlit版）")
    st.divider()
    
    # 侧边栏：文件上传 + 操作
    with st.sidebar:
        st.header("📂 文件管理")
        
        # 多文件上传
        uploaded_files = st.file_uploader(
            "上传TXT/Word文件（支持多文件）",
            type=["txt", "docx"],
            accept_multiple_files=True
        )
        
        # 上传文件处理
        if st.button("✅ 导入选中文件") and uploaded_files:
            st.session_state.file_list = []
            for file in uploaded_files:
                file_name, content = read_uploaded_file(file)
                if content:
                    st.session_state.file_list.append({
                        "name": file_name,
                        "content": content,
                        "analysis_data": [],
                        "mdd": 0.0,
                        "stats": {}
                    })
            st.success(f"成功导入 {len(st.session_state.file_list)} 个文件！")
        
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
                    with st.spinner("正在执行依存分析..."):
                        analysis_data, mdd, stats = analyze_text(active_file["content"])
                        st.session_state.file_list[st.session_state.active_file_idx]["analysis_data"] = analysis_data
                        st.session_state.file_list[st.session_state.active_file_idx]["mdd"] = mdd
                        st.session_state.file_list[st.session_state.active_file_idx]["stats"] = stats
                    st.success(f"「{selected_file}」分析完成！")
            
            with col2:
                if st.button("📊 批量依存分析"):
                    with st.spinner("批量分析中..."):
                        processed = 0
                        for idx, f in enumerate(st.session_state.file_list):
                            analysis_data, mdd, stats = analyze_text(f["content"])
                            st.session_state.file_list[idx]["analysis_data"] = analysis_data
                            st.session_state.file_list[idx]["mdd"] = mdd
                            st.session_state.file_list[idx]["stats"] = stats
                            processed += 1
                    st.success(f"批量分析完成！共处理 {processed} 个文件")
    
    # 主内容区：标签页切换
    tab1, tab2, tab3, tab4 = st.tabs(["📄 原始文本", "🧹 文本清洗", "📈 依存分析结果", "📊 MDD横向对比"])
    
    # 标签页1：原始文本
    with tab1:
        if st.session_state.file_list and st.session_state.active_file_idx >=0:
            active_file = st.session_state.file_list[st.session_state.active_file_idx]
            st.subheader(f"原始文本：{active_file['name']}")
            st.text_area(
                "内容",
                value=active_file["content"],
                height=500,
                key="raw_text"
            )
        else:
            st.info("请先在侧边栏上传并导入文件")
    
    # 标签页2：文本清洗
    with tab2:
        if st.session_state.file_list and st.session_state.active_file_idx >=0:
            active_file = st.session_state.file_list[st.session_state.active_file_idx]
            st.subheader(f"文本清洗：{active_file['name']}")
            
            # 执行清洗
            if st.button("🧹 执行清洗"):
                cleaned_text = clean_aozora_format(active_file["content"])
                cleaned_text = keep_all_valid(cleaned_text)
                cleaned_text = split_sentences_aozora(cleaned_text)
                st.session_state.cleaned_text = cleaned_text
                
                # 清洗结果展示
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
                        mime="text/plain"
                    )
                with col2:
                    # 转为Word字节流
                    doc = Document()
                    for line in cleaned_text.split("\n"):
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
                st.info("点击「执行清洗」按钮查看结果")
        else:
            st.info("请先在侧边栏上传并导入文件")
    
    # 标签页3：依存分析结果
    with tab3:
        if st.session_state.file_list and st.session_state.active_file_idx >=0:
            active_file = st.session_state.file_list[st.session_state.active_file_idx]
            st.subheader(f"依存分析结果：{active_file['name']}")
            
            if active_file["analysis_data"]:
                # 转换为DataFrame展示
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
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
                col5.metric("MDD值", round(active_file["mdd"], 4))
            else:
                st.warning(f"「{active_file['name']}」尚未执行依存分析，请在侧边栏点击分析按钮")
        else:
            st.info("请先在侧边栏上传并导入文件")
    
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
                    "平均依存距离(MDD)": round(f["mdd"], 4),
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
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
            st.bar_chart(
                df_compare.set_index("文件名")["平均依存距离(MDD)"],
                y_label="MDD值",
                use_container_width=True
            )
        else:
            st.info("请先执行批量/单文件依存分析，再查看MDD对比")

if __name__ == "__main__":
    main()