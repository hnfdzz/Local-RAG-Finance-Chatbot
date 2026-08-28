import os
import re
import glob
import json
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import streamlit as st
import config
from src.loader import DocumentProcessor
from src.vectorstore import VectorStoreManager
from src.rag_chain import FinanceRAGChain
from src.utils import extract_citations

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

st.set_page_config(
    page_title="Finance RAG Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #171717 !important;
        color: #ECECF1;
    }

    div[data-testid="stSidebar"] {
        background-color: #0F0F11 !important;
        border-right: 1px solid #262626 !important;
    }

    .claude-hero {
        text-align: center;
        max-width: 720px;
        margin: 20px auto 20px auto;
        padding: 0 10px;
    }

    .claude-hero h1 {
        font-size: 2.2rem;
        font-weight: 600;
        color: #F3F4F6;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }

    .claude-hero p {
        font-size: 0.98rem;
        color: #9CA3AF;
        line-height: 1.5;
    }

    .status-card-claude {
        background: #18181B;
        border: 1px solid #27272A;
        border-radius: 12px;
        padding: 16px;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .status-title-claude {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #71717A;
        text-transform: uppercase;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }

    .info-label { color: #A1A1AA; }
    .info-value { color: #F4F4F5; font-weight: 600; font-family: monospace; }

    .stButton > button {
        background-color: #212121 !important;
        color: #E5E7EB !important;
        border: 1px solid #333333 !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: #2A2A2A !important;
        border-color: #D97706 !important;
        color: #FFFFFF !important;
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #D97706 0%, #B45309 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        max-width: 800px;
        margin: 0 auto;
    }

    div[data-testid="stChatInput"] {
        max-width: 800px;
        margin: 0 auto;
    }

    div[data-testid="stChatInput"] > div {
        background-color: #212121 !important;
        border: 1px solid #333333 !important;
        border-radius: 16px !important;
    }

    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #D97706 !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        color: #F3F4F6 !important;
    }
</style>
""", unsafe_allow_html=True)

def parse_financial_metrics(chunks):
    text = " ".join([c.page_content for c in chunks])
    income = 0.0
    expenses = 0.0
    categories = {}
    anomalies = []

    income_patterns = re.findall(r'(?i)(gaji|pemasukan|pendapatan|transfer masuk)\s*[:\-=]?\s*(?:rp\.?\s*)?([\d\.]+)', text)
    for tag, val_str in income_patterns:
        try:
            val = float(val_str.replace('.', ''))
            income += val
        except ValueError:
            pass

    expense_patterns = re.findall(r'(?i)(belanja|makanan|makan|transportasi|bensin|sewa|listrik|air|internet|hiburan|kebutuhan|pos|asuransi|cicilan)\s*[:\-=]?\s*(?:rp\.?\s*)?([\d\.]+)', text)
    for cat, val_str in expense_patterns:
        try:
            val = float(val_str.replace('.', ''))
            expenses += val
            cat_name = cat.capitalize()
            categories[cat_name] = categories.get(cat_name, 0.0) + val
        except ValueError:
            pass

    if income > 0:
        for cat, val in categories.items():
            if val > (income * 0.30):
                anomalies.append(f"Kategori '{cat}' menyerap >30% pemasukan (Rp {val:,.0f}).")

    if income == 0 and expenses == 0:
        numbers = [float(n.replace('.', '')) for n in re.findall(r'Rp\s*([\d\.]+)', text) if n.replace('.', '').isdigit()]
        if len(numbers) >= 2:
            income = max(numbers)
            expenses = sum(numbers) - income
            categories = {"Kebutuhan Utama": expenses * 0.7, "Lain-lain": expenses * 0.3}

    return {
        "income": income,
        "expenses": expenses,
        "categories": categories,
        "anomalies": anomalies
    }

def save_session_to_file(custom_name=""):
    if not st.session_state.messages:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    title = custom_name.strip() if custom_name.strip() else f"Sesi {datetime.now().strftime('%d %b %Y %H:%M')}"
    filename = f"session_{timestamp}.json"
    filepath = os.path.join(SESSIONS_DIR, filename)
    
    data = {
        "title": title,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": st.session_state.messages,
        "processed_files": st.session_state.processed_files,
        "financial_metrics": st.session_state.financial_metrics
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return title

def load_session_from_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.messages = data.get("messages", [])
        st.session_state.processed_files = data.get("processed_files", [])
        st.session_state.financial_metrics = data.get("financial_metrics", None)
        return True
    except Exception as e:
        st.error(f"Gagal memuat sesi: {str(e)}")
        return False

def get_all_saved_sessions():
    files = sorted(glob.glob(os.path.join(SESSIONS_DIR, "*.json")), reverse=True)
    sessions = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                sessions.append({
                    "path": file_path,
                    "title": d.get("title", "Sesi Tanpa Judul"),
                    "timestamp": d.get("timestamp", "")
                })
        except Exception:
            pass
    return sessions

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "financial_metrics" not in st.session_state:
    st.session_state.financial_metrics = None

if "vectorstore_manager" not in st.session_state:
    st.session_state.vectorstore_manager = VectorStoreManager()
    st.session_state.vectorstore_manager.initialize_db()

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = FinanceRAGChain()

if "preset_query" not in st.session_state:
    st.session_state.preset_query = None

with st.sidebar:
    st.markdown("### 🛡️ **Privacy-First RAG**")
    st.caption("Seluruh pemrosesan data keuangan berjalan 100% lokal tanpa API Cloud.")
    
    st.divider()

    st.markdown("##### 💾 **Simpan & Muat Sesi**")
    
    session_title_input = st.text_input("Judul Sesi (Opsional)", placeholder="misal: Analisis Rekening Agustus", key="sess_name")
    if st.button("💾 Simpan Sesi Saat Ini", use_container_width=True):
        if st.session_state.messages:
            saved_title = save_session_to_file(session_title_input)
            st.success(f"Tersimpan: '{saved_title}'")
        else:
            st.warning("Belum ada percakapan untuk disimpan.")

    saved_sessions = get_all_saved_sessions()
    if saved_sessions:
        session_options = {f"{s['title']} ({s['timestamp']})": s['path'] for s in saved_sessions}
        selected_session = st.selectbox("Pilih Sesi Lama", list(session_options.keys()))
        
        col_load, col_del = st.columns([2, 1])
        with col_load:
            if st.button("📂 Muat", use_container_width=True):
                path = session_options[selected_session]
                if load_session_from_file(path):
                    st.success("Sesi dimuat!")
                    st.rerun()
        with col_del:
            if st.button("🗑️ Hapus", use_container_width=True):
                path = session_options[selected_session]
                if os.path.exists(path):
                    os.remove(path)
                    st.rerun()

    st.divider()
    
    st.markdown("##### 📄 **Unggah Laporan Keuangan**")
    uploaded_files = st.file_uploader(
        "Pilih berkas PDF (Rekening Koran / Laporan Keuangan)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Unggah dokumen PDF keuangan Anda di sini."
    )
    
    if st.button("🚀 Proses & Indeks Dokumen", type="primary", use_container_width=True):
        if uploaded_files:
            processor = DocumentProcessor()
            all_chunks = []
            
            with st.spinner("Mengekstrak & mentransformasi PDF ke vektor lokal..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    try:
                        chunks = processor.process_and_split(tmp_path)
                        for chunk in chunks:
                            chunk.metadata["source_file"] = uploaded_file.name
                        all_chunks.extend(chunks)
                        
                        if uploaded_file.name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(uploaded_file.name)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                if all_chunks:
                    st.session_state.vectorstore_manager.add_documents(all_chunks)
                    st.session_state.financial_metrics = parse_financial_metrics(all_chunks)
                    st.success(f"Berhasil mengindeks {len(all_chunks)} chunks dari {len(uploaded_files)} dokumen!")
        else:
            st.warning("Silakan pilih minimal satu berkas PDF terlebih dahulu.")

    st.divider()
    
    st.markdown(f"""
    <div class="status-card-claude">
        <div class="status-title-claude">
            <div class="pulse-dot"></div>
            STATUS SISTEM LOKAL
        </div>
        <div class="info-row">
            <span class="info-label">MODEL LLM</span>
            <span class="info-value">{config.LLM_MODEL}</span>
        </div>
        <div class="info-row">
            <span class="info-label">EMBEDDING</span>
            <span class="info-value">MiniLM-L6-v2</span>
        </div>
        <div class="info-row">
            <span class="info-label">DOKUMEN TERINDEKS</span>
            <span class="info-value">{len(st.session_state.processed_files)} File</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("➕ Chat Baru (Reset)", use_container_width=True):
        st.session_state.vectorstore_manager.reset_database()
        st.session_state.messages = []
        st.session_state.processed_files = []
        st.session_state.financial_metrics = None
        st.session_state.preset_query = None
        st.rerun()

st.markdown("""
<div class="claude-hero">
    <h1>Local RAG Finance Intelligence</h1>
    <p>Analisis dokumen transaksi & laporan keuangan secara instan, presisi, dan 100% offline.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.processed_files and st.session_state.financial_metrics:
    metrics = st.session_state.financial_metrics
    inc = metrics.get("income", 0.0)
    exp = metrics.get("expenses", 0.0)
    cats = metrics.get("categories", {})
    anomalies = metrics.get("anomalies", [])

    with st.expander("📊 **Dashboard Visual & Smart Health Check**", expanded=True):
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Pemasukan", f"Rp {inc:,.0f}" if inc > 0 else "T/A")
        with col_m2:
            st.metric("Total Pengeluaran", f"Rp {exp:,.0f}" if exp > 0 else "T/A")
        with col_m3:
            sisa = inc - exp if inc > 0 else 0
            st.metric("Sisa Cashflow", f"Rp {sisa:,.0f}" if inc > 0 else "T/A", delta=f"{(sisa/inc*100):.1f}%" if inc > 0 else None)

        if anomalies or (inc > 0 and exp > (inc * 0.8)):
            st.markdown("##### 🚨 Smart Alerts")
            if inc > 0 and exp > (inc * 0.8):
                st.warning("⚠️ **Rasio Pengeluaran Tinggi**: Pengeluaran Anda melebihi 80% dari total pemasukan.")
            for alert in anomalies:
                st.error(f"🔍 **Anomali**: {alert}")
        elif inc > 0:
            st.success("✅ **Health Check Good**: Arus kas stabil, tidak ditemukan pembengkakan biaya mencurigakan.")

        if cats or (inc > 0 or exp > 0):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                df_cat = pd.DataFrame(list(cats.items()), columns=["Kategori", "Jumlah"]) if cats else pd.DataFrame([["Pengeluaran", exp]], columns=["Kategori", "Jumlah"])
                fig_pie = px.pie(df_cat, values="Jumlah", names="Kategori", title="Alokasi Pengeluaran", hole=0.45, color_discrete_sequence=px.colors.sequential.Darkmint)
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ECECF1", margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_c2:
                df_bar = pd.DataFrame({
                    "Tipe": ["Pemasukan", "Pengeluaran"],
                    "Jumlah [Rp]": [inc, exp]
                })
                fig_bar = px.bar(df_bar, x="Tipe", y="Jumlah [Rp]", title="Perbandingan Cashflow", color="Tipe", color_discrete_map={"Pemasukan": "#10B981", "Pengeluaran": "#EF4444"})
                fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#ECECF1", margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

if not st.session_state.messages:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 **Total Pengeluaran**\n\nHitung total pengeluaran dari berkas", use_container_width=True):
            st.session_state.preset_query = "Berapa total pengeluaran yang tercatat dalam dokumen ini?"
            st.rerun()
            
    with col2:
        if st.button("🔍 **Transaksi Terbesar**\n\nCari 3 transaksi dengan nilai tertinggi", use_container_width=True):
            st.session_state.preset_query = "Sebutkan 3 transaksi terbesar beserta tanggal dan keterangannya!"
            st.rerun()
            
    with col3:
        if st.button("📑 **Ringkasan Transaksi**\n\nBuat ringkasan arus kas & aktivitas", use_container_width=True):
            st.session_state.preset_query = "Buatkan ringkasan singkat dari seluruh aktivitas transaksi pada dokumen ini."
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("📌 Rujukan Halaman PDF"):
                for cite in message["citations"]:
                    st.caption(f"• **{cite['source']}** (Halaman {cite['page']})")
                    st.text(f'  "{cite["snippet"]}"')

active_input = st.chat_input("Tanyakan sesuatu tentang data keuangan Anda...")

if st.session_state.preset_query and not active_input:
    active_input = st.session_state.preset_query
    st.session_state.preset_query = None

if active_input:
    st.session_state.messages.append({"role": "user", "content": active_input})
    with st.chat_message("user"):
        st.markdown(active_input)

    with st.chat_message("assistant"):
        with st.spinner("Menganalisis dokumen keuangan lokal..."):
            try:
                retriever = st.session_state.vectorstore_manager.get_retriever(k=4)
                chain = st.session_state.rag_chain.create_chain(retriever)
                
                response = chain.invoke({"input": active_input})
                answer = response.get("answer", "Maaf, tidak dapat menghasilkan jawaban.")
                source_docs = response.get("context", [])
                
                citations = extract_citations(source_docs)
                
                st.markdown(answer)
                
                if citations:
                    with st.expander("📌 Rujukan Halaman PDF"):
                        for cite in citations:
                            st.caption(f"• **{cite['source']}** (Halaman {cite['page']})")
                            st.text(f'  "{cite["snippet"]}"')

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })

            except Exception as e:
                error_msg = f"Terjadi kesalahan saat memproses pertanyaan: {str(e)}"
                st.error(error_msg)
                st.info("Pastikan layanan Ollama sudah berjalan di latar belakang (misal: `ollama run phi3:mini`).")