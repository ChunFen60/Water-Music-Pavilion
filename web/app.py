import os
import sys
import time
import tempfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.append(BASE_DIR)
try:
    from modules.midi_analyzer import analyze_midi
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
from web.auth import (
    create_user, verify_user, log_visit, get_stats,
    is_admin, get_all_users, get_visit_logs, get_user_visit_counts,
    promote_to_admin
)
from web.social import (
    send_friend_request, get_pending_requests, get_sent_requests,
    accept_friend_request, reject_friend_request, get_friends, are_friends,
    send_private_message, get_private_messages,
    create_group, join_group, get_my_groups, get_group_info,
    get_group_members, search_groups_by_name,
    send_group_message, get_group_messages,
    upload_file, get_shared_files, get_file
)

# ---- new analysis modules ----
try:
    from modules.emotion_model import (
        compute_from_analyze_result,
        compute_from_dataframe_row,
    )
    EMOTION_MODEL_AVAILABLE = True
except ImportError:
    EMOTION_MODEL_AVAILABLE = False

# ---- dimensionality reduction ----
UMAP_AVAILABLE = False
TSNE_AVAILABLE = False
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap as umap
        UMAP_AVAILABLE = True
    except ImportError:
        pass

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    pass

# ---- interactive table helper ----
def _interactive_table(df, key="table", page_size=20, height=400):
    """Display a DataFrame with search, sort, pagination, and CSV export."""
    # Search
    search = st.text_input("Search", placeholder="Type to filter...",
                          key=f"search_{key}", label_visibility="collapsed")
    if search:
        mask = np.column_stack([
            df[col].astype(str).str.contains(search, case=False, na=False)
            for col in df.columns
        ]).any(axis=1)
        filtered = df[mask]
    else:
        filtered = df

    # Sort
    sort_col = st.selectbox("Sort by", ["(none)"] + list(filtered.columns),
                           key=f"sort_{key}", label_visibility="collapsed")
    if sort_col != "(none)":
        ascending = st.checkbox("Ascending", value=True, key=f"asc_{key}")
        filtered = filtered.sort_values(sort_col, ascending=ascending)

    # Pagination
    n = len(filtered)
    total_pages = max(1, (n + page_size - 1) // page_size)
    if f"page_{key}" not in st.session_state:
        st.session_state[f"page_{key}"] = 0
    page = st.session_state[f"page_{key}"]
    page = max(0, min(page, total_pages - 1))

    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        if st.button("Prev", key=f"prev_{key}", disabled=(page == 0)):
            st.session_state[f"page_{key}"] -= 1
            st.rerun()
    with c2:
        if st.button("Next", key=f"next_{key}", disabled=(page >= total_pages - 1)):
            st.session_state[f"page_{key}"] += 1
            st.rerun()
    with c3:
        st.caption(f"Page {page+1} of {total_pages}  ({n} rows)")
    with c4:
        csv = filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Export CSV", csv, f"{key}.csv",
                          "text/csv", key=f"dl_{key}")

    start = page * page_size
    st.dataframe(filtered.iloc[start:start+page_size],
                use_container_width=True, height=height)

# =========================
# Session State Init
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"
if "visit_logged" not in st.session_state:
    st.session_state.visit_logged = False
# --- social session state ---
if "friends_tab" not in st.session_state:
    st.session_state.friends_tab = "friends"  # search | requests | friends
if "active_chat" not in st.session_state:
    st.session_state.active_chat = None
if "groups_tab" not in st.session_state:
    st.session_state.groups_tab = "my_groups"  # my_groups | create | join
if "active_group" not in st.session_state:
    st.session_state.active_group = None
# --- search session state ---
if "search_result" not in st.session_state:
    st.session_state.search_result = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="Water Music Pavilion",
    page_icon="🎹",
    layout="wide"
)

# =========================
# 环境变量
# =========================
load_dotenv()

# Streamlit Cloud 密钥注入环境变量
try:
    for _k, _v in st.secrets.items():
        if _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:
    pass

api_key = os.getenv("DEEPSEEK_API_KEY")

# =========================
# 项目路径
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

csv_path = os.path.join(
    BASE_DIR,
    "database",
    "music_dataset.csv"
)

# =========================
# 读取数据
# =========================
df = pd.read_csv(csv_path)

# =========================
# 自动识别作曲家字段
# =========================
possible_cols = [
    "canonical_composer",
    "composer",
    "artist"
]

composer_col = next(
    (col for col in possible_cols if col in df.columns),
    None
)

if composer_col is None:
    st.error("❌ 未找到作曲家字段")
    st.write(df.columns.tolist())
    st.stop()

# =========================
# 自动检测数值字段
# =========================
numeric_cols = (
    df.select_dtypes(include="number")
    .columns
    .tolist()
)

pitch_col = next(
    (c for c in numeric_cols if "pitch" in c.lower()),
    None
)

duration_col = next(
    (c for c in numeric_cols if "duration" in c.lower()),
    None
)

# =========================
# DeepSeek Client
# =========================
client = None

if api_key:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

# =========================
# 记录访问
# =========================
if not st.session_state.visit_logged:
    visitor = st.session_state.username if st.session_state.logged_in else None
    log_visit(visitor)
    st.session_state.visit_logged = True

# =========================
# 登录 / 注册
# =========================
if not st.session_state.logged_in:

    st.markdown("""
    <style>
    /* 隐藏 Streamlit 默认顶栏 */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* 项目标题横幅 */
    .project-banner {
        text-align: center;
        padding: 36px 20px 10px 20px;
    }
    .project-banner h1 {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #4F6FDE, #8B9FE8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 6px;
    }
    .project-banner p {
        color: #888;
        font-size: 14px;
        letter-spacing: 1px;
    }
    /* 登录卡片 */
    .auth-container {
        max-width: 420px;
        margin: 24px auto 0 auto;
        padding: 40px 36px;
        border-radius: 16px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(128,128,128,0.15);
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
    }
    .auth-container h2 {
        text-align: center;
        margin-bottom: 28px;
        font-weight: 700;
    }
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }
    </style>

    <div class="project-banner">
        <h1>Water Music<br>Pavilion</h1>
        <p>EXPLORE THE ART OF PIANO MUSIC</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)

        if st.session_state.auth_mode == "login":
            st.markdown("## 🎹 Sign In")
        else:
            st.markdown("## 🎹 Create Account")

        with st.form("auth_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            if st.session_state.auth_mode == "register":
                password_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")

            submitted = st.form_submit_button(
                "Sign In" if st.session_state.auth_mode == "login" else "Create Account",
                type="primary",
                use_container_width=True
            )

            if submitted:
                if not username or not password:
                    st.error("Please fill in all fields.")
                elif st.session_state.auth_mode == "login":
                    if verify_user(username, password):
                        admin_name = os.getenv("ADMIN_USERNAME", "")
                        if admin_name and username == admin_name:
                            promote_to_admin(username)
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:  # register
                    if password != password_confirm:
                        st.error("Passwords do not match.")
                    elif len(password) < 4:
                        st.error("Password must be at least 4 characters.")
                    elif create_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Username already exists. Please choose another.")

        # 切换登录/注册
        if st.session_state.auth_mode == "login":
            st.markdown("Don't have an account?  ")
            if st.button("Create one now →", key="switch_register", use_container_width=True):
                st.session_state.auth_mode = "register"
                st.rerun()
        else:
            st.markdown("Already have an account?  ")
            if st.button("← Back to Sign In", key="switch_login", use_container_width=True):
                st.session_state.auth_mode = "login"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# =========================
# 页面标题
# =========================
st.title("🎹 Water Music Pavilion")

# =========================
# Sidebar Navigation
# =========================
st.sidebar.markdown("""
<style>
/* 整体侧边栏导航按钮容器 */
div[data-testid="stSidebar"] div.stButton > button {
    width: 100%;
    text-align: left;
    padding: 12px 18px;
    border-radius: 8px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    background: transparent;
    color: inherit;
    font-size: 15px;
    font-weight: 500;
    margin-bottom: 4px;
    transition: all 0.2s ease;
    cursor: pointer;
}
div[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(99, 126, 237, 0.12);
    border-color: rgba(99, 126, 237, 0.5);
}
div[data-testid="stSidebar"] div.stButton > button:active,
div[data-testid="stSidebar"] div.stButton > button:focus {
    background: linear-gradient(135deg, #4F6FDE, #6C8CEE);
    border-color: #4F6FDE;
    color: #fff;
    box-shadow: 0 2px 8px rgba(79, 111, 222, 0.35);
}
/* active 状态通过 session state 控制，用 markdown 高亮当前项 */
.nav-active-item {
    background: linear-gradient(135deg, #4F6FDE, #6C8CEE);
    border: 1px solid #4F6FDE;
    border-radius: 8px;
    padding: 12px 18px;
    color: #fff;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 4px;
    display: block;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🎹 Navigation")

nav_items = [
    "📊 Dataset Overview",
    "🎼 Composer Analysis",
    "🎵 MIDI Analysis",
    "🎭 Emotion Analysis",
    "🤖 AI Music Assistant"
]

nav_items.append("💬 Friends & Chat")
nav_items.append("👥 Groups")

if is_admin(st.session_state.username):
    nav_items.append("🔐 Admin Panel")

# 初始化 session state
if "nav_option" not in st.session_state:
    st.session_state.nav_option = nav_items[0]

# 渲染导航按钮
for item in nav_items:
    is_active = st.session_state.nav_option == item
    btn_label = f"▶ {item}" if is_active else f"  {item}"
    if st.sidebar.button(
        btn_label,
        key=f"nav_{item}",
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        st.session_state.nav_option = item
        st.rerun()

nav_option = st.session_state.nav_option

st.sidebar.markdown("---")

# =========================
# Section: Dataset Overview
# =========================
if nav_option == "📊 Dataset Overview":

    st.header("📊 Dataset Overview")

    # -- 数据集总览统计 --
    total_pieces = len(df)
    total_composers = df[composer_col].dropna().astype(str).nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Pieces", total_pieces)
    col2.metric("Total Composers", total_composers)
    col3.metric("Avg Pitch Range", f"{df['pitch_range'].mean():.0f} st")
    col4.metric("Avg Note Density", f"{df['note_density'].mean():.1f}/s")

    # -- 作曲家作品数量柱状图 --
    st.subheader("🎼 Pieces per Composer")

    composer_counts = (
        df[composer_col]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )
    composer_counts.columns = ["Composer", "Count"]

    fig_bar = px.bar(
        composer_counts,
        x="Composer",
        y="Count",
        color="Composer",
        title="Number of Pieces by Composer",
        labels={"Count": "Pieces"}
    )
    fig_bar.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig_bar, use_container_width=True)

    # -- Style Landscape: Pitch Range vs Note Density --
    st.subheader("🎹 Stylistic Landscape: Range vs Note Density")

    fig_style = px.scatter(
        df,
        x="pitch_range",
        y="note_density",
        color=composer_col,
        size="tempo",
        hover_name="title",
        title="Each Work — Pitch Range × Note Density (bubble size = tempo)",
        labels={
            "pitch_range": "Pitch Range (semitones)",
            "note_density": "Note Density (notes/sec)"
        },
        opacity=0.75
        
    )
    fig_style.update_layout(height=520, showlegend=False)
    st.plotly_chart(fig_style, use_container_width=True)

    # -- Rhythmic vs Melodic Complexity --
    st.subheader("🎵 Rhythmic vs Melodic Complexity")

    fig_rm = px.scatter(
        df,
        x="rhythm_std",
        y="melodic_complexity",
        color=composer_col,
        hover_name="title",
        title="Rhythmic Variation × Melodic Complexity",
        labels={
            "rhythm_std": "Rhythm Variation (std)",
            "melodic_complexity": "Melodic Complexity"
        },
        opacity=0.75
    )
    fig_rm.update_layout(height=480, showlegend=False)
    st.plotly_chart(fig_rm, use_container_width=True)

    # -- Rhythm Style Quadrant --
    st.subheader("🎵 Rhythm Style Quadrant: Density × Variation")

    med_density = df["note_density"].median()
    med_rhythm = df["rhythm_std"].median()

    fig_quad = px.scatter(
        df,
        x="note_density",
        y="rhythm_std",
        color=composer_col,
        size="tempo",
        hover_name="title",
        title="Rhythm Style Map — 4 Quadrants of Piano Writing",
        labels={
            "note_density": "Note Density (notes/sec)",
            "rhythm_std": "Rhythm Variation (std)"
        },
        opacity=0.7
    )
    fig_quad.add_hline(y=med_rhythm, line_dash="dash", line_color="gray", opacity=0.4)
    fig_quad.add_vline(x=med_density, line_dash="dash", line_color="gray", opacity=0.4)
    fig_quad.add_annotation(x=df["note_density"].max() * 0.92, y=df["rhythm_std"].max() * 0.95,
                             text="🏇 Virtuoso", showarrow=False, font=dict(color="#888", size=11))
    fig_quad.add_annotation(x=df["note_density"].max() * 0.92, y=df["rhythm_std"].min() * 1.05,
                             text="⚙️ Motor-like", showarrow=False, font=dict(color="#888", size=11))
    fig_quad.add_annotation(x=df["note_density"].min() * 1.08, y=df["rhythm_std"].max() * 0.95,
                             text="🎭 Free Rubato", showarrow=False, font=dict(color="#888", size=11))
    fig_quad.add_annotation(x=df["note_density"].min() * 1.08, y=df["rhythm_std"].min() * 1.05,
                             text="🌫 Minimalist", showarrow=False, font=dict(color="#888", size=11))
    fig_quad.update_layout(height=520, showlegend=False)
    st.plotly_chart(fig_quad, use_container_width=True)

    # -- Composer Similarity Network --
    st.subheader("🔗 Composer Style Map")

    try:
        features_net = ["pitch_range", "note_density", "tempo", "rhythm_std",
                        "melodic_complexity", "avg_velocity", "pitch_variance"]
        composer_profiles = df.groupby(composer_col)[features_net].mean().dropna()
        profiles_np = composer_profiles.values
        means = profiles_np.mean(axis=0)
        stds = profiles_np.std(axis=0)
        stds[stds == 0] = 1
        profiles_norm = (profiles_np - means) / stds

        # cosine similarity matrix
        norms_arr = np.linalg.norm(profiles_norm, axis=1)
        sim_matrix = np.dot(profiles_norm, profiles_norm.T) / np.outer(norms_arr, norms_arr)

        composer_names = composer_profiles.index.tolist()
        n_comp = len(composer_names)

        # Dimensionality reduction method selector
        methods = ["UMAP", "t-SNE", "PCA"]
        available = [UMAP_AVAILABLE, TSNE_AVAILABLE, True]
        method_labels = [
            f"{m} {'(recommended)' if m=='UMAP' else ''}{' (unavailable)' if not a else ''}"
            for m, a in zip(methods, available)
        ]
        default_method = "UMAP" if UMAP_AVAILABLE else ("t-SNE" if TSNE_AVAILABLE else "PCA")
        chosen_method = default_method
        # Only show radio if multiple methods available
        if sum(available) > 1:
            chosen_method = st.radio(
                "Projection method", method_labels,
                index=[i for i, a in enumerate(available) if a and methods[i]==default_method][0]
                if default_method in methods else 0,
                horizontal=True, label_visibility="collapsed"
            )
            # Strip availability note
            for m in methods:
                if m in chosen_method:
                    chosen_method = m
                    break

        # Compute 2D embedding
        if chosen_method == "UMAP" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_neighbors=min(8, n_comp-1), min_dist=0.3,
                               n_components=2, random_state=42)
            embedding = reducer.fit_transform(profiles_norm)
            method_name = "UMAP"
        elif chosen_method == "t-SNE" and TSNE_AVAILABLE:
            tsne = TSNE(n_components=2, perplexity=min(6, n_comp-1),
                       random_state=42)
            embedding = tsne.fit_transform(profiles_norm)
            method_name = "t-SNE"
        else:
            # PCA via SVD
            centered = profiles_norm - profiles_norm.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            embedding = centered @ Vt[:2].T
            method_name = "PCA"

        # Jitter text positions to reduce overlap
        np.random.seed(42)
        positions = embedding.copy()
        jitter = np.random.randn(n_comp, 2) * np.std(positions, axis=0) * 0.03
        text_pos = positions + jitter

        fig_net = go.Figure()

        # Draw similarity edges (threshold 0.4 for cleaner visualization)
        for i in range(n_comp):
            sims = sim_matrix[i].copy()
            sims[i] = -1
            top_k = np.argsort(sims)[-3:]
            for j in top_k:
                if i < j and sims[j] > 0.4:
                    fig_net.add_trace(go.Scatter(
                        x=[positions[i, 0], positions[j, 0]],
                        y=[positions[i, 1], positions[j, 1]],
                        mode="lines",
                        line=dict(width=sims[j] * 1.5, color="Grey"),
                        opacity=0.25,
                        hoverinfo="none",
                        showlegend=False
                    ))

        fig_net.add_trace(go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            mode="markers",
            marker=dict(size=12, color=np.arange(n_comp), colorscale="Viridis",
                       showscale=False, line=dict(width=1, color="white")),
            hovertext=composer_names,
            hoverinfo="text",
            showlegend=False
        ))

        # Text labels with jitter (as separate trace for cleaner rendering)
        for i, name in enumerate(composer_names):
            fig_net.add_annotation(
                x=text_pos[i, 0], y=text_pos[i, 1],
                text=name, showarrow=False,
                font=dict(size=8, color="#ccc"),
                bgcolor="rgba(0,0,0,0.5)", borderpad=2
            )

        fig_net.update_layout(
            title=f"Composer Style Similarity Map ({method_name} — closer = more similar)",
            height=620,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_net, use_container_width=True)
    except Exception as e:
        st.warning("Composer network visualization unavailable. Check app logs for details.")
        import sys
        print(f"NETWORK_ERROR: {e}", file=sys.stderr)

    # -- 数据表预览 --
    st.subheader("🧾 Full Dataset Preview")
    _interactive_table(df, key="dataset_overview", page_size=25, height=600)

# =========================
# Section: Composer Analysis
# =========================
elif nav_option == "🎼 Composer Analysis":

    composer_list = sorted(
        df[composer_col]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_composers = st.sidebar.multiselect(
        "Select Composers (1-5)",
        composer_list,
        default=[composer_list[0]],
        max_selections=5
    )

    if not selected_composers:
        st.warning("Select at least one composer to view analysis.")
        st.stop()

    # Primary composer for detailed view
    primary = selected_composers[0]
    composer_df = df[df[composer_col].astype(str) == primary]
    other_composers = [c for c in selected_composers if c != primary]

    st.header(f"🎼 Composer Style Profile: {', '.join(selected_composers)}")

    # -- 关键风格指标（主要作曲家）--
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Pieces", len(composer_df))
    col2.metric("Pitch Range", f"{composer_df['pitch_range'].mean():.0f} st")
    col3.metric("Note Density", f"{composer_df['note_density'].mean():.1f}/s")
    col4.metric("Tempo", f"{composer_df['tempo'].mean():.0f} BPM")
    col5.metric("Melodic Complexity", f"{composer_df['melodic_complexity'].mean():.1f}")

    # -- 风格雷达图：多作曲家对比 --
    st.subheader("🎹 Style Radar: Multi-Composer Comparison")

    import plotly.graph_objects as go
    from plotly import colors as pcolors

    dimensions = ["pitch_range", "note_density", "tempo", "rhythm_std", "melodic_complexity"]
    dim_labels = ["Pitch Range", "Note Density", "Tempo", "Rhythm Variation", "Melodic Complexity"]

    # Normalize to 0-1
    norms = {}
    for d in dimensions:
        vmin, vmax = df[d].min(), df[d].max()
        norms[d] = (vmin, vmax)

    fig_radar = go.Figure()
    palette = pcolors.qualitative.Plotly[:max(len(selected_composers), 2)]

    for idx, comp in enumerate(selected_composers):
        comp_df_i = df[df[composer_col].astype(str) == comp]
        vals = []
        for d in dimensions:
            vmin, vmax = norms[d]
            vals.append(round((comp_df_i[d].mean() - vmin) / (vmax - vmin), 3))
        color = palette[idx % len(palette)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=dim_labels + [dim_labels[0]],
            fill='toself',
            name=comp,
            line=dict(color=color, width=2),
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)')
        ))

    # Global average
    global_vals = []
    for d in dimensions:
        vmin, vmax = norms[d]
        global_vals.append(round((df[d].mean() - vmin) / (vmax - vmin), 3))
    fig_radar.add_trace(go.Scatterpolar(
        r=global_vals + [global_vals[0]],
        theta=dim_labels + [dim_labels[0]],
        fill='toself',
        name='Global Average',
        line=dict(color='#aaa', width=1.5, dash='dash'),
        fillcolor='rgba(170, 170, 170, 0.1)'
    ))

    n_comp = len(selected_composers)
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
        height=400 + 20 * n_comp,
        margin=dict(t=40, b=40, l=60, r=60),
        legend=dict(orientation="h", y=-0.1, font=dict(size=11))
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # -- 风格散点：高亮所有选中作曲家 --
    st.subheader("🎹 Style Position: Range × Density")

    combined_parts = []
    color_map = {}
    palette_full = pcolors.qualitative.Plotly[:len(selected_composers)]
    for idx, comp in enumerate(selected_composers):
        subset = df[df[composer_col].astype(str) == comp].copy()
        subset["Group"] = comp
        combined_parts.append(subset)
        color_map[comp] = palette_full[idx % len(palette_full)]

    others = df[~df[composer_col].astype(str).isin(selected_composers)].copy()
    others["Group"] = "Other Composers"
    combined_parts.append(others)
    color_map["Other Composers"] = "#ccc"

    combined = pd.concat(combined_parts)

    fig_pos = px.scatter(
        combined,
        x="pitch_range",
        y="note_density",
        color="Group",
        size="tempo",
        hover_name="title",
        title=f"{' & '.join(selected_composers)} vs All Others",
        color_discrete_map=color_map,
        labels={
            "pitch_range": "Pitch Range (semitones)",
            "note_density": "Note Density (notes/sec)"
        },
        opacity=0.8
    )
    fig_pos.update_layout(height=480)
    st.plotly_chart(fig_pos, use_container_width=True)

    # -- 节奏-旋律复杂度散点 --
    st.subheader("🎵 Rhythmic vs Melodic Profile")

    fig_prof = px.scatter(
        combined,
        x="rhythm_std",
        y="melodic_complexity",
        color="Group",
        hover_name="title",
        title=f"{' & '.join(selected_composers)} — Rhythm × Melody Space",
        color_discrete_map=color_map,
        labels={
            "rhythm_std": "Rhythm Variation (std)",
            "melodic_complexity": "Melodic Complexity"
        },
        opacity=0.8
    )
    fig_prof.update_layout(height=420)
    st.plotly_chart(fig_prof, use_container_width=True)

    # -- 音域分布小提琴图（Top 20 作曲家）--
    st.subheader("🎻 Pitch Range Distribution by Composer")

    top_20 = df[composer_col].value_counts().head(20).index.tolist()
    top_df = df[df[composer_col].isin(top_20)]

    fig_violin = px.violin(
        top_df,
        x=composer_col,
        y="pitch_range",
        color=composer_col,
        box=True,
        points="outliers",
        title="Tessitura Spread — Top 20 Composers (wider = broader keyboard range)",
        labels={"pitch_range": "Pitch Range (semitones)"}
    )
    fig_violin.update_layout(height=420, showlegend=False, xaxis_tickangle=-45)
    st.plotly_chart(fig_violin, use_container_width=True)

    # -- 力度 × 音高密度热力图 --
    st.subheader("🎼 Velocity × Pitch Density Landscape")

    fig_vp = px.density_heatmap(
        df,
        x="avg_velocity",
        y="avg_pitch",
        title="Touch & Range Map — Where do composers place their notes?",
        labels={
            "avg_velocity": "Average Velocity (MIDI 0-127)",
            "avg_pitch": "Average Pitch (MIDI note)"
        },
        color_continuous_scale="Blues"
    )
    fig_vp.update_layout(height=460)
    st.plotly_chart(fig_vp, use_container_width=True)

    # -- 作品列表 --
    st.subheader("🧾 Works by " + primary)
    emotion_display = "emotion_label" if "emotion_label" in composer_df.columns else "emotion"
    cols_display = ["title"]
    if emotion_display in composer_df.columns:
        cols_display.append(emotion_display)
    elif "emotion_legacy" in composer_df.columns:
        cols_display.append("emotion_legacy")
    cols_display += ["pitch_range", "note_density", "tempo", "melodic_complexity"]
    _interactive_table(
        composer_df[[c for c in cols_display if c in composer_df.columns]],
        key="composer_works", page_size=20, height=600
    )

# =========================
# Section: MIDI Analysis
# =========================
elif nav_option == "🎵 MIDI Analysis":

    st.header("🎵 MIDI Upload & Analysis")

    if not MIDI_AVAILABLE:
        st.warning("⚠ MIDI module not available. Please check your installation.")
    else:
        uploaded_file = st.file_uploader(
            "Upload MIDI File",
            type=["mid", "midi"]
        )

        if uploaded_file is not None:

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mid"
                ) as tmp_file:

                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name

                result = analyze_midi(temp_path)

                if "error" in result:
                    st.error(result["error"])

                else:

                    # ---- Build feature vector from uploaded MIDI ----
                    notes_arr = pd.DataFrame(result["notes_data"])
                    velocities = notes_arr["velocity"].values
                    dynamic_range = velocities.max() - velocities.min()

                    # weighted features (F-ratio from dataset as weight)
                    # excludes tempo & avg_velocity (F<7, noisy/performance-dependent)
                    features_sim = [
                        "pitch_range", "pitch_variance", "velocity_variance",
                        "melodic_complexity", "note_density", "rhythm_std",
                    ]
                    feature_weights = np.array([62.2, 32.3, 18.5, 14.4, 7.0, 2.5])
                    feature_weights = feature_weights / feature_weights.sum()

                    fv_upload = {k: result[k] for k in features_sim}

                    ds_mean = df[features_sim].mean()
                    ds_std = df[features_sim].std()
                    ds_std[ds_std == 0] = 1

                    uploaded_norm = (np.array([fv_upload[k] for k in features_sim]) - ds_mean.values) / ds_std.values

                    # ---- Piece-level weighted k-NN ----
                    ds_norm = (df[features_sim] - ds_mean) / ds_std
                    w = feature_weights
                    ds_weighted = ds_norm.values * np.sqrt(w)
                    up_weighted = uploaded_norm * np.sqrt(w)
                    ds_cos = np.dot(ds_weighted, up_weighted) / (
                        np.linalg.norm(ds_weighted, axis=1) * np.linalg.norm(up_weighted) + 1e-10
                    )
                    top5_idx = np.argsort(ds_cos)[-5:][::-1]
                    top5_pieces = df.iloc[top5_idx]

                    # ---- Composer-level weighted similarity ----
                    composer_profiles = df.groupby(composer_col)[features_sim].mean()
                    cp_norm = (composer_profiles - ds_mean) / ds_std
                    cp_weighted = cp_norm.values * np.sqrt(w)
                    cp_cos = np.dot(cp_weighted, up_weighted) / (
                        np.linalg.norm(cp_weighted, axis=1) * np.linalg.norm(up_weighted) + 1e-10
                    )
                    top_comp_idx = np.argsort(cp_cos)[-5:][::-1]
                    top_composers = [
                        (composer_profiles.index[i], round(cp_cos[i] * 100, 1))
                        for i in top_comp_idx
                    ]

                    # ---- meaningful metrics ----
                    st.markdown("---")
                    st.subheader("🎼 Musical Feature Profile")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("🎹 Pitch Range", f"{result['pitch_range']} st",
                                help="Keyboard range — wider = more dramatic")
                    col2.metric("🎼 Polyphony", f"{result.get('avg_polyphony', 0):.1f} voices",
                                help="Average simultaneous notes — higher = denser counterpoint (Bach~3.5, Chopin~2.0)")
                    col3.metric("🎨 Chromaticism", f"{result.get('pitch_entropy', 0):.3f}",
                                help="Pitch-class entropy 0-1 — higher = more chromatic (Romantic), lower = diatonic (Baroque)")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("📊 Note Density", f"{result['note_density']:.1f} n/s",
                                help="Notes per second — higher = busier texture")
                    col2.metric("🎵 Melodic Complexity", f"{result['melodic_complexity']:.1f}",
                                help="Average pitch interval between consecutive notes")
                    col3.metric("🎶 Rhythm Variation", f"{result['rhythm_std']:.2f}",
                                help="Std of note durations — higher = more rhythmic variety")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🔑 Key", f"{result['key']} ({result['key_confidence']}%)",
                                help="Detected musical key")
                    col2.metric("⏱ Tempo", f"{result['tempo']:.0f} BPM")
                    col3.metric("💥 Dynamic Range", f"{int(dynamic_range)}",
                                help="Velocity range (MIDI 0-127)")
                    col4.metric("📝 Total Notes", result["total_notes"])

                    # ---- Feature radar vs dataset average ----
                    st.markdown("---")
                    st.subheader("🎯 Style Radar: Your MIDI vs Dataset Average")

                    radar_dims = dict(zip(
                        ["Pitch Range", "Pitch Variance", "Velocity Var.",
                         "Melodic Complx.", "Note Density", "Rhythm Variation"],
                        features_sim
                    ))

                    radar_midi = []
                    radar_global = []
                    for label, feat in radar_dims.items():
                        vmin, vmax = df[feat].min(), df[feat].max()
                        rng = vmax - vmin if vmax != vmin else 1
                        radar_midi.append((fv_upload[feat] - vmin) / rng)
                        radar_global.append((df[feat].mean() - vmin) / rng)

                    radar_labels = list(radar_dims.keys())
                    fig_radar_midi = go.Figure()
                    fig_radar_midi.add_trace(go.Scatterpolar(
                        r=radar_midi + [radar_midi[0]],
                        theta=radar_labels + [radar_labels[0]],
                        fill="toself", name="Your MIDI",
                        line=dict(color="#4F6FDE", width=2),
                        fillcolor="rgba(79,111,222,0.25)"
                    ))
                    fig_radar_midi.add_trace(go.Scatterpolar(
                        r=radar_global + [radar_global[0]],
                        theta=radar_labels + [radar_labels[0]],
                        fill="toself", name="Dataset Average",
                        line=dict(color="#aaa", width=1.5, dash="dash"),
                        fillcolor="rgba(170,170,170,0.1)"
                    ))
                    fig_radar_midi.update_layout(
                        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                        height=380,
                        margin=dict(t=40, b=20, l=60, r=60),
                        legend=dict(orientation="h", y=-0.1)
                    )
                    st.plotly_chart(fig_radar_midi, use_container_width=True)

                    # ---- Composer Similarity ----
                    st.markdown("---")
                    st.subheader("🎼 Composer Style Similarity")
                    st.caption("Cosine similarity on 8-dim musical feature vector")

                    comp_data = []
                    for name, score in top_composers:
                        comp_data.append({"Composer": name, "Similarity": f"{score}%"})
                    comp_sim_df = pd.DataFrame(comp_data)

                    # bar chart
                    fig_comp = px.bar(
                        comp_sim_df, x="Similarity", y="Composer",
                        orientation="h",
                        title="Top 5 Closest Composers",
                        color="Similarity", color_continuous_scale="Blues",
                        text="Similarity"
                    )
                    fig_comp.update_traces(textposition="outside")
                    fig_comp.update_layout(height=260, yaxis=dict(autorange="reversed"), showlegend=False)
                    st.plotly_chart(fig_comp, use_container_width=True)

                    st.caption("Most similar: **" + top_composers[0][0] + "** (" + str(top_composers[0][1]) + "%)")

                    # ---- Nearest pieces (emotion reference) ----
                    st.markdown("---")
                    st.subheader("🎭 Closest Matches in Dataset")
                    st.caption("These 5 classical works have the most similar musical features to your uploaded MIDI.")

                    ref_data = []
                    for i, (_, row) in enumerate(top5_pieces.iterrows()):
                        emotion_col_nn = "emotion_label" if "emotion_label" in top5_pieces.columns else "emotion"
                    ref_data.append({
                        "Composer": row[composer_col],
                        "Title": row["title"],
                        "Emotion": row.get(emotion_col_nn, row.get("emotion", "")),
                        "Similarity": f"{round(ds_cos[top5_idx[i]] * 100, 1)}%"
                    })
                    _interactive_table(pd.DataFrame(ref_data), key="nearest_matches", page_size=5)

                    if emotion_col_nn in top5_pieces.columns:
                        predominant_emotion = top5_pieces[emotion_col_nn].mode().values[0] if len(top5_pieces) > 0 else "Calm"
                    else:
                        predominant_emotion = "Calm"
                    st.info(f"Predominant emotional character: **{predominant_emotion}** "
                            f"(based on dataset nearest-neighbor matching)")

                    # ---- VA Position for uploaded MIDI ----
                    if EMOTION_MODEL_AVAILABLE:
                        st.markdown("---")
                        st.subheader("🎯 Your MIDI in Emotion Space")
                        va = compute_from_analyze_result(result)
                        col_v, col_a, col_l = st.columns(3)
                        col_v.metric("Valence", f"{va['valence_score']:.3f}",
                                    help="Pleasantness (0-1)")
                        col_a.metric("Arousal", f"{va['arousal_score']:.3f}",
                                    help="Energy/intensity (0-1)")
                        col_l.metric("Emotion", va['emotion_label'],
                                    help=f"Sub-region: {va['sub_region']}")

                    # ---- Harmony Analysis ----
                    if "chord_types" in result and result.get("chord_types"):
                        st.markdown("---")
                        st.subheader("🎸 Harmony Analysis")

                        hcol1, hcol2, hcol3 = st.columns(3)
                        hcol1.metric("Chord Diversity",
                                    f"{result.get('chord_diversity', 0):.3f}")
                        hcol2.metric("Harmonic Rhythm",
                                    f"{result.get('harmonic_rhythm', 0):.2f} chg/s")
                        hcol3.metric("Tonic/Dominant",
                                    f"{result.get('tonic_dominant_ratio', 0.5):.2f}")

                        hcol1.metric("Bass Step Ratio",
                                    f"{result.get('bass_motion_step_ratio', 0):.2f}")
                        hcol2.metric("Chromatic Chords",
                                    f"{result.get('chromatic_pct', 0)*100:.1f}%")
                        hcol3.metric("Total Chords",
                                    f"{result.get('total_chords_detected', 0)}")

                        # Chord type pie chart
                        chord_types = result["chord_types"]
                        if chord_types:
                            chord_pie = px.pie(
                                names=list(chord_types.keys()),
                                values=list(chord_types.values()),
                                title="Chord Type Distribution",
                                hole=0.4
                            )
                            chord_pie.update_layout(height=300, showlegend=True)
                            st.plotly_chart(chord_pie, use_container_width=True)

                        # Cadence timeline
                        cadences = result.get("cadences", [])
                        if cadences:
                            st.subheader("🎵 Cadence Map")
                            cad_df = pd.DataFrame(cadences)
                            fig_cad = px.scatter(
                                cad_df, x="time", y="type",
                                color="type", size="strength",
                                title="Cadences Along the Timeline",
                                labels={"time": "Time (s)", "type": "Cadence Type"}
                            )
                            fig_cad.update_layout(height=250, showlegend=False)
                            st.plotly_chart(fig_cad, use_container_width=True)
                        else:
                            pc = result.get("cadence_counts", {})
                            if pc:
                                st.caption(f"Cadence counts: {pc}")

                    # ---- Piano Roll ----
                    st.markdown("---")
                    st.subheader("🎹 Piano Roll Visualization")

                    notes_df_piano = pd.DataFrame(result["notes_data"])
                    fig_roll = px.scatter(
                        notes_df_piano,
                        x="start", y="pitch",
                        size="duration", color="velocity",
                        title="Interactive Piano Roll",
                        hover_data=["start", "end", "duration", "pitch", "velocity"]
                    )
                    fig_roll.update_layout(height=600)
                    st.plotly_chart(fig_roll, use_container_width=True)

                    # ---- AI Interpretation ----
                    if client:
                        st.markdown("---")

                        top5_desc = "\n".join([
                            f"  {i+1}. {ref_data[i]['Composer']} — {ref_data[i]['Title']} ({ref_data[i]['Emotion']}, similarity: {ref_data[i]['Similarity']})"
                            for i in range(len(ref_data))
                        ])

                        prompt = f"""Analyze this classical piano piece based on its complete musical profile.

Musical Profile:
- Key: {result['key']} (confidence: {result['key_confidence']}%)
- Tempo: {result['tempo']:.0f} BPM
- Pitch Range: {result['pitch_range']} semitones
- Note Density: {result['note_density']:.1f} notes/sec
- Melodic Complexity: {result['melodic_complexity']:.1f} (average interval between notes)
- Rhythm Variation: {result['rhythm_std']:.2f} (std of durations)
- Dynamic Range: {int(dynamic_range)} (MIDI velocity)
- Polyphony: {result.get('avg_polyphony', 0):.1f} avg simultaneous voices
  (Baroque counterpoint ~3.5, Romantic melody+accomp ~2.0, solo ~1.0)
- Chromaticism: {result.get('pitch_entropy', 0):.3f} (0-1 scale)
  (diatonic ~0.70, moderately chromatic ~0.85, highly chromatic ~0.95)

The 5 most stylistically similar pieces in our classical piano dataset:
{top5_desc}

Interpret the musical meaning behind these numbers. What style/era does this profile suggest? How do the polyphony and chromaticism values specifically inform the composer attribution? Provide a concise but insightful analysis of style, emotion, and performance character."""

                        with st.spinner("AI Analyzing Music..."):

                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a professional classical music analyst "
                                            "with expertise in piano repertoire, music theory, "
                                            "and performance practice. Provide insightful, "
                                            "musically-informed analysis in natural language. "
                                            "Do NOT just restate the numbers — interpret what "
                                            "they mean musically."
                                        )
                                    },
                                    {"role": "user", "content": prompt}
                                ]
                            )

                            answer = response.choices[0].message.content

                            st.subheader("🤖 AI Music Interpretation")
                            st.success(answer)

            except Exception as e:
                st.error(f"MIDI Analysis Error: {e}")

# =========================
# Section: Emotion Analysis
# =========================
elif nav_option == "🎭 Emotion Analysis":

    st.header("🎭 Emotion Analysis Dashboard")

    # Detect emotion columns
    has_va = "valence_score" in df.columns and "arousal_score" in df.columns
    has_label = "emotion_label" in df.columns
    emotion_col = (
        "emotion_label" if has_label else
        next((c for c in df.columns if "emotion" in c.lower()), None)
    )

    if has_va:
        # ---- Valence-Arousal 2D Space ----
        st.subheader("🎯 Valence-Arousal Emotion Space")

        # Quadrant boundary
        v_med = df["valence_score"].median()
        a_med = df["arousal_score"].median()

        fig_va = px.scatter(
            df,
            x="valence_score",
            y="arousal_score",
            color=emotion_col if has_label else composer_col,
            size="tempo",
            hover_name="title",
            hover_data=[composer_col] + ([emotion_col] if has_label else []),
            title="Each Piece in Valence-Arousal Space (size = tempo)",
            labels={
                "valence_score": "Valence (pleasantness) →",
                "arousal_score": "Arousal (energy) →",
            },
            opacity=0.7
        )
        fig_va.add_hline(y=a_med, line_dash="dash", line_color="grey", opacity=0.4)
        fig_va.add_vline(x=v_med, line_dash="dash", line_color="grey", opacity=0.4)

        # Quadrant labels
        fig_va.add_annotation(x=0.95, y=0.95, text="Passionate", showarrow=False,
                             font=dict(color="#888", size=11), xref="paper", yref="paper")
        fig_va.add_annotation(x=0.95, y=0.03, text="Agitated", showarrow=False,
                             font=dict(color="#888", size=11), xref="paper", yref="paper")
        fig_va.add_annotation(x=0.03, y=0.95, text="Tender/Calm", showarrow=False,
                             font=dict(color="#888", size=11), xref="paper", yref="paper")
        fig_va.add_annotation(x=0.03, y=0.03, text="Melancholic", showarrow=False,
                             font=dict(color="#888", size=11), xref="paper", yref="paper")

        fig_va.update_layout(height=520)
        st.plotly_chart(fig_va, use_container_width=True)

        # ---- Emotion Distribution ----
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Emotion Label Distribution")
            if has_label:
                emo_counts = df["emotion_label"].value_counts().reset_index()
                emo_counts.columns = ["Emotion", "Count"]
                fig_pie = px.pie(emo_counts, names="Emotion", values="Count",
                                title="Emotion Quadrants", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)

            # VA score metrics
            col_v, col_a = st.columns(2)
            col_v.metric("Mean Valence", f"{df['valence_score'].mean():.3f}")
            col_a.metric("Mean Arousal", f"{df['arousal_score'].mean():.3f}")

        with col2:
            st.subheader("VA by Composer")
            comp_va = df.groupby(composer_col)[["valence_score", "arousal_score"]].mean()
            comp_va = comp_va[comp_va.index.isin(
                df[composer_col].value_counts().head(15).index
            )]
            fig_comp_va = px.scatter(
                comp_va.reset_index(),
                x="valence_score", y="arousal_score",
                text=composer_col,
                title="Composer Average VA Position (Top 15)",
                labels={"valence_score": "Valence", "arousal_score": "Arousal"},
            )
            fig_comp_va.update_traces(textposition="top center", textfont=dict(size=9))
            fig_comp_va.update_layout(height=420)
            st.plotly_chart(fig_comp_va, use_container_width=True)

        # ---- Features by Emotion Quadrant ----
        st.subheader("🎹 Musical Features by Emotion Quadrant")

        if has_label:
            col_left, col_right = st.columns(2)
            with col_left:
                fig_box1 = px.box(df, x="emotion_label", y="pitch_range",
                                 color="emotion_label",
                                 title="Pitch Range by Emotion",
                                 labels={"pitch_range": "Pitch Range (st)"})
                fig_box1.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig_box1, use_container_width=True)
            with col_right:
                fig_box2 = px.box(df, x="emotion_label", y="note_density",
                                 color="emotion_label",
                                 title="Note Density by Emotion",
                                 labels={"note_density": "Note Density (n/s)"})
                fig_box2.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig_box2, use_container_width=True)

        # ---- Legacy Emotion (if available) ----
        legacy_col = next((c for c in df.columns if "emotion_legacy" in c.lower()), None)
        if legacy_col and has_label:
            with st.expander("Compare with Legacy Emotion Labels"):
                st.caption("Old rule-based labels vs new VA-based labels")
                cross = pd.crosstab(df[legacy_col], df["emotion_label"])
                st.dataframe(cross, use_container_width=True)

    else:
        # Fallback: old emotion column
        st.warning("VA emotion model not yet applied. Run scripts/regenerate_emotions.py to enable the new emotion dashboard.")

        if emotion_col:
            st.subheader("Emotion Distribution (Legacy)")
            emotion_counts = (
                df[emotion_col].dropna().astype(str).value_counts().reset_index()
            )
            emotion_counts.columns = ["Emotion", "Count"]
            fig_pie = px.pie(emotion_counts, names="Emotion", values="Count",
                            title="Emotion Distribution", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

    # ---- 情绪风格空间 (always available, needs some emotion col) ----
    display_col = emotion_col if emotion_col else composer_col
    st.subheader("🎵 Style Space Colored by Emotion")

    fig_emo_space = px.scatter(
        df,
        x="pitch_range",
        y="note_density",
        color=display_col,
        size="tempo",
        hover_name="title",
        title="Works by Emotion — Pitch Range vs Note Density",
        labels={
            "pitch_range": "Pitch Range (semitones)",
            "note_density": "Note Density (notes/sec)"
        },
        opacity=0.75
    )
    fig_emo_space.update_layout(height=480)
    st.plotly_chart(fig_emo_space, use_container_width=True)

    # ---- Correlation Matrix ----
    st.subheader("🔗 Feature Correlation Matrix")

    num_cols_for_corr = (
        df.select_dtypes(include="number")
        .columns.tolist()
    )
    if len(num_cols_for_corr) >= 2:
        corr_df = df[num_cols_for_corr].corr()
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            title="Numeric Feature Correlations",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation analysis.")

# =========================
# Section: AI Music Assistant
# =========================
elif nav_option == "🤖 AI Music Assistant":

    st.header("🤖 AI Music Assistant")

    if client:

        st.markdown("""
        Ask me anything about classical music — composers, styles,
        music theory, or get analysis help!
        """)

        user_question = st.text_area(
            "Your Question",
            placeholder="e.g. What are the characteristics of Chopin's nocturnes?",
            height=100
        )

        if st.button("Ask AI", type="primary"):

            try:

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional "
                                "classical music analyst "
                                "with deep knowledge of "
                                "music theory, history, "
                                "and performance practice."
                            )
                        },
                        {"role": "user", "content": user_question}
                    ]
                )

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                st.subheader("💬 AI Response")
                st.success(answer)

            except Exception as e:
                st.error(f"AI Error: {e}")

        # -- 快速问题 --
        st.sidebar.markdown("### 💡 Quick Questions")
        quick_q = st.sidebar.selectbox(
            "Pick a question",
            [
                "— Choose —",
                "Compare Baroque vs Romantic style",
                "Explain sonata form structure",
                "What makes Chopin unique?",
                "Describe impressionism in music",
                "How does counterpoint work?",
                "What are the mood characteristics of minor keys?"
            ]
        )

        if st.sidebar.button("Ask Quick Question") and quick_q != "— Choose —":

            with st.spinner("AI is thinking..."):

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional "
                                "classical music analyst."
                            )
                        },
                        {"role": "user", "content": quick_q}
                    ]
                )

                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                st.subheader("💬 AI Response")
                st.success(answer)

    else:
        st.warning("⚠ DeepSeek API Key not detected. Please set DEEPSEEK_API_KEY in .env file.")

# =========================
# Section: Friends & Chat
# =========================
elif nav_option == "💬 Friends & Chat":

    st.header("💬 Friends & Chat")

    username = st.session_state.username

    # -- sub-tabs --
    tab_names = ["👥 My Friends", "🔍 Search Users", "📩 Requests"]
    tab_keys = ["friends", "search", "requests"]
    current_idx = tab_keys.index(st.session_state.friends_tab) if st.session_state.friends_tab in tab_keys else 0

    cols = st.columns(len(tab_names))
    for i, (name, key) in enumerate(zip(tab_names, tab_keys)):
        with cols[i]:
            btn_type = "primary" if key == st.session_state.friends_tab else "secondary"
            if st.button(name, key=f"ftab_{key}", use_container_width=True, type=btn_type):
                st.session_state.friends_tab = key
                st.session_state.active_chat = None
                st.rerun()

    st.markdown("---")

    # ======== Search Users ========
    if st.session_state.friends_tab == "search":
        st.subheader("🔍 Find Users")

        # init session state for search persistence
        if "search_result" not in st.session_state:
            st.session_state.search_result = None  # None | 'self' | 'found' | 'not_found'
        if "search_query" not in st.session_state:
            st.session_state.search_query = ""

        search_name = st.text_input("Enter username to search", placeholder="Type a username...")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Search", type="primary", use_container_width=True):
                query = search_name.strip()
                if query:
                    st.session_state.search_query = query
                    if query == username:
                        st.session_state.search_result = "self"
                    else:
                        from web.auth import user_exists
                        if user_exists(query):
                            st.session_state.search_result = "found"
                        else:
                            st.session_state.search_result = "not_found"
                else:
                    st.session_state.search_query = ""
                    st.session_state.search_result = None

        # display saved search result (persists across reruns)
        sq = st.session_state.search_query
        sr = st.session_state.search_result
        if sq and sr:
            if sr == "self":
                st.warning("That's you!")
            elif sr == "found":
                if are_friends(username, sq):
                    st.info(f"You are already friends with **{sq}**.")
                else:
                    st.success(f"User **{sq}** found!")
                    if st.button(f"➕ Send Friend Request to {sq}", type="primary", key="send_req_btn"):
                        ok, msg = send_friend_request(username, sq)
                        if ok:
                            st.session_state.search_result = "sent"
                            st.success(msg)
                        else:
                            st.warning(msg)
            elif sr == "sent":
                st.success(f"Friend request sent to **{sq}**! ✅")
            elif sr == "not_found":
                st.error(f"User '{sq}' not found.")

        # show sent requests
        st.markdown("---")
        st.caption("📤 Sent Requests")
        sent = get_sent_requests(username)
        if sent:
            for rid, to_user, status, created in sent:
                status_icon = {"pending": "⏳", "accepted": "✅", "rejected": "❌"}.get(status, "")
                st.write(f"{status_icon} To: **{to_user}** — {status} ({created})")
        else:
            st.caption("No sent requests.")

    # ======== Requests ========
    elif st.session_state.friends_tab == "requests":
        st.subheader("📩 Friend Requests")
        pending = get_pending_requests(username)
        if pending:
            for rid, from_user, created in pending:
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"👤 **{from_user}**")
                    st.caption(f"Sent: {created}")
                with col2:
                    if st.button("✅ Accept", key=f"acc_{rid}", type="primary", use_container_width=True):
                        ok, msg = accept_friend_request(rid, username)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                with col3:
                    if st.button("❌ Reject", key=f"rej_{rid}", use_container_width=True):
                        ok, msg = reject_friend_request(rid, username)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                st.markdown("---")
        else:
            st.info("No pending friend requests.")

    # ======== My Friends ========
    else:
        friends = get_friends(username)

        if st.session_state.active_chat and st.session_state.active_chat in friends:
            # --- Chat Window ---
            chat_with = st.session_state.active_chat

            col_back, col_title = st.columns([1, 5])
            with col_back:
                if st.button("← Back", key="back_friends", use_container_width=True):
                    st.session_state.active_chat = None
                    st.rerun()
            with col_title:
                st.subheader(f"💬 Chat with {chat_with}")

            st.markdown("---")

            # Message display area
            msgs = get_private_messages(username, chat_with, limit=50)
            chat_html = '<div style="max-height:400px;overflow-y:auto;padding:8px;">'
            if msgs:
                for sender, content, sent_at in msgs:
                    if sender == username:
                        chat_html += (
                            f'<div style="text-align:right;margin:6px 0;">'
                            f'<span style="background:#4F6FDE;color:#fff;padding:8px 14px;'
                            f'border-radius:14px 14px 0 14px;display:inline-block;max-width:75%;'
                            f'text-align:left;word-wrap:break-word;">{content}</span>'
                            f'<div style="font-size:11px;color:#888;margin-top:2px;">{sent_at}</div>'
                            f'</div>'
                        )
                    else:
                        chat_html += (
                            f'<div style="text-align:left;margin:6px 0;">'
                            f'<span style="background:#333;color:#ddd;padding:8px 14px;'
                            f'border-radius:14px 14px 14px 0;display:inline-block;max-width:75%;'
                            f'text-align:left;word-wrap:break-word;">'
                            f'<strong>{sender}</strong><br>{content}</span>'
                            f'<div style="font-size:11px;color:#888;margin-top:2px;">{sent_at}</div>'
                            f'</div>'
                        )
            else:
                chat_html += '<div style="text-align:center;color:#888;padding:40px;">No messages yet. Say hello!</div>'
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)

            # Send message
            with st.form("pm_form", clear_on_submit=True):
                col_input, col_send = st.columns([4, 1])
                with col_input:
                    msg_text = st.text_input("Message", placeholder="Type a message...", label_visibility="collapsed")
                with col_send:
                    send_btn = st.form_submit_button("Send ▶", type="primary", use_container_width=True)
                if send_btn and msg_text.strip():
                    send_private_message(username, chat_with, msg_text.strip())
                    st.rerun()

            time.sleep(3)
            st.rerun()

        else:
            st.subheader("👥 My Friends")
            if friends: 
                for f in friends:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"🎹 **{f}**")
                    with col2:
                        if st.button("💬 Chat", key=f"chat_{f}", type="primary", use_container_width=True):
                            st.session_state.active_chat = f
                            st.rerun()
                    st.markdown("---")
            else:
                st.info("No friends yet. Search for users to add friends!")

# =========================
# Section: Groups
# =========================
elif nav_option == "👥 Groups":

    st.header("👥 Groups")

    username = st.session_state.username

    # -- sub-tabs --
    tab_names = ["📋 My Groups", "➕ Create Group", "🔗 Join Group"]
    tab_keys = ["my_groups", "create", "join"]
    current_idx = tab_keys.index(st.session_state.groups_tab) if st.session_state.groups_tab in tab_keys else 0

    cols = st.columns(len(tab_names))
    for i, (name, key) in enumerate(zip(tab_names, tab_keys)):
        with cols[i]:
            btn_type = "primary" if key == st.session_state.groups_tab else "secondary"
            if st.button(name, key=f"gtab_{key}", use_container_width=True, type=btn_type):
                st.session_state.groups_tab = key
                st.session_state.active_group = None
                st.rerun()

    st.markdown("---")

    # ======== My Groups ========
    if st.session_state.groups_tab == "my_groups":
        my_groups = get_my_groups(username)

        if st.session_state.active_group:
            active_gc = st.session_state.active_group
            if not is_group_member(active_gc, username):
                st.warning("You are no longer a member of this group.")
                st.session_state.active_group = None
                st.rerun()

            ginfo = get_group_info(active_gc)
            if ginfo is None:
                st.error("Group not found.")
                st.session_state.active_group = None
                st.rerun()

            # --- Group Chat Window ---
            col_back, col_title = st.columns([1, 5])
            with col_back:
                if st.button("← Back", key="back_groups", use_container_width=True):
                    st.session_state.active_group = None
                    st.rerun()
            with col_title:
                st.subheader(f"💬 {ginfo[1]}")
                st.caption(f"Code: {ginfo[0]} | Created by: {ginfo[2]}")

            # Members sidebar
            members = get_group_members(active_gc)
            with st.expander(f"👥 Members ({len(members)})"):
                for m_name, m_joined in members:
                    st.write(f"🎹 **{m_name}** — joined {m_joined}")

            st.markdown("---")

            # Group messages
            gmsgs = get_group_messages(active_gc, limit=50)
            chat_html = '<div style="max-height:350px;overflow-y:auto;padding:8px;">'
            if gmsgs:
                for sender, content, sent_at in gmsgs:
                    if sender == username:
                        chat_html += (
                            f'<div style="text-align:right;margin:6px 0;">'
                            f'<span style="background:#4F6FDE;color:#fff;padding:8px 14px;'
                            f'border-radius:14px 14px 0 14px;display:inline-block;max-width:75%;'
                            f'text-align:left;word-wrap:break-word;">{content}</span>'
                            f'<div style="font-size:11px;color:#888;margin-top:2px;">{sent_at}</div>'
                            f'</div>'
                        )
                    else:
                        chat_html += (
                            f'<div style="text-align:left;margin:6px 0;">'
                            f'<span style="background:#333;color:#ddd;padding:8px 14px;'
                            f'border-radius:14px 14px 14px 0;display:inline-block;max-width:75%;'
                            f'text-align:left;word-wrap:break-word;">'
                            f'<strong>{sender}</strong><br>{content}</span>'
                            f'<div style="font-size:11px;color:#888;margin-top:2px;">{sent_at}</div>'
                            f'</div>'
                        )
            else:
                chat_html += '<div style="text-align:center;color:#888;padding:40px;">No messages yet.</div>'
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)

            # Send group message
            with st.form("gm_form", clear_on_submit=True):
                col_input, col_send = st.columns([4, 1])
                with col_input:
                    gmsg_text = st.text_input("Message", placeholder="Type a message...", label_visibility="collapsed", key="gmsg_input")
                with col_send:
                    gsend_btn = st.form_submit_button("Send ▶", type="primary", use_container_width=True)
                if gsend_btn and gmsg_text.strip():
                    send_group_message(active_gc, username, gmsg_text.strip())
                    st.rerun()

            st.markdown("---")

            # ---- File Sharing ----
            st.subheader("📎 Shared Files")

            uploaded_file = st.file_uploader(
                "Upload a file to share (max 5MB, PDF/MIDI/images)",
                type=["pdf", "mid", "midi", "png", "jpg", "jpeg", "txt", "musicxml", "xml"],
                key="group_file_uploader"
            )
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                file_type = uploaded_file.name.split(".")[-1] if "." in uploaded_file.name else "unknown"
                ok, msg = upload_file(active_gc, username, uploaded_file.name, file_type, file_bytes)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

            shared = get_shared_files(active_gc)
            if shared:
                for fid, f_user, f_name, f_type, f_size, f_time in shared:
                    size_kb = f_size / 1024
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        icon = {"pdf": "📄", "mid": "🎵", "midi": "🎵", "png": "🖼", "jpg": "🖼", "jpeg": "🖼", "txt": "📝", "musicxml": "🎼", "xml": "🎼"}.get(f_type, "📁")
                        st.write(f"{icon} **{f_name}** ({size_kb:.1f} KB) — from {f_user}")
                        st.caption(f_time)
                    with col2:
                        filedata = get_file(fid)
                        if filedata:
                            st.download_button(
                                "⬇ Download",
                                data=filedata[2],
                                file_name=filedata[0],
                                mime="application/octet-stream",
                                key=f"dl_{fid}",
                                use_container_width=True
                            )
                    st.markdown("---")
            else:
                st.caption("No shared files yet.")

            time.sleep(3)
            st.rerun()

        else:
            if my_groups:
                for gcode, gname, creator, created in my_groups:
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"📋 **{gname}**")
                        st.caption(f"Code: {gcode}")
                    with col2:
                        st.caption(f"Created by: {creator}")
                    with col3:
                        if st.button("Enter", key=f"enter_{gcode}", type="primary", use_container_width=True):
                            st.session_state.active_group = gcode
                            st.rerun()
                    st.markdown("---")
            else:
                st.info("You haven't joined any groups yet. Create one or join with a code!")

    # ======== Create Group ========
    elif st.session_state.groups_tab == "create":
        st.subheader("➕ Create a New Group")
        group_name = st.text_input("Group Name", placeholder="Enter a group name...")
        if st.button("Create Group", type="primary", use_container_width=True):
            if not group_name.strip():
                st.warning("Please enter a group name.")
            else:
                code, msg = create_group(group_name.strip(), username)
                if code:
                    st.success(msg)
                    st.balloons()
                    st.markdown(f"### Share this code with friends: **`{code}`**")
                else:
                    st.error(msg)

    # ======== Join Group ========
    else:
        st.subheader("🔗 Join a Group")
        search_method = st.radio("Find by:", ["Enter Group Code", "Search by Name"], horizontal=True)

        if search_method == "Enter Group Code":
            input_code = st.text_input("Group Code", placeholder="Enter 5-digit group code...", max_chars=5)
            if st.button("Join Group", type="primary", use_container_width=True):
                if not input_code.strip():
                    st.warning("Please enter a group code.")
                else:
                    ok, msg = join_group(input_code.strip(), username)
                    if ok:
                        st.success(msg)
                        st.session_state.groups_tab = "my_groups"
                        st.rerun()
                    else:
                        st.error(msg)
        else:
            search_keyword = st.text_input("Search by group name", placeholder="Type a keyword...")
            if st.button("Search Groups", type="primary", use_container_width=True):
                if search_keyword.strip():
                    results = search_groups_by_name(search_keyword.strip())
                    if results:
                        for gcode, gname, creator in results:
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.write(f"📋 **{gname}**")
                                st.caption(f"Code: {gcode}")
                            with col2:
                                st.caption(f"Creator: {creator}")
                            with col3:
                                if is_group_member(gcode, username):
                                    st.success("Joined")
                                else:
                                    if st.button("Join", key=f"join_{gcode}", type="primary", use_container_width=True):
                                        ok, msg = join_group(gcode, username)
                                        if ok:
                                            st.success(msg)
                                            st.rerun()
                                        else:
                                            st.error(msg)
                            st.markdown("---")
                    else:
                        st.info("No groups found matching your search.")

# =========================
# Section: Admin Panel
# =========================
elif nav_option == "🔐 Admin Panel":

    st.header("🔐 Admin Panel")

    st.markdown("---")

    # -- 概览卡片 --
    stats = get_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Users", stats["total_users"])
    col2.metric("👀 Total Visits", stats["total_visits"])
    col3.metric("📅 Today", stats["today_visits"])
    col4.metric("👤 Logged-in", stats["logged_visits"])

    st.markdown("---")

    # -- 注册用户列表 --
    st.subheader("👥 Registered Users")

    users = get_all_users()
    if users:
        user_data = []
        for uid, uname, role, created in users:
            user_data.append({
                "ID": uid,
                "Username": uname,
                "Role": "👑 Admin" if role == "admin" else "User",
                "Registered": created
            })
        _interactive_table(
            pd.DataFrame(user_data),
            key="admin_users", page_size=20, height=400
        )
    else:
        st.info("No users registered yet.")

    st.markdown("---")

    # -- 用户活跃度排行 --
    st.subheader("📊 User Activity Ranking")

    visit_counts = get_user_visit_counts()
    if visit_counts:
        activity_data = []
        for uname, cnt in visit_counts:
            activity_data.append({
                "Username": uname,
                "Visits": cnt
            })
        _interactive_table(
            pd.DataFrame(activity_data),
            key="admin_activity", page_size=20, height=400
        )

    st.markdown("---")

    # -- 最近访问记录 --
    st.subheader("📋 Recent Visit Logs")

    logs = get_visit_logs(100)
    if logs:
        log_data = []
        for lid, uname, vtime in logs:
            log_data.append({
                "ID": lid,
                "User": uname or "(Guest)",
                "Time": vtime
            })
        _interactive_table(
            pd.DataFrame(log_data),
            key="admin_logs", page_size=25, height=450
        )
    else:
        st.info("No visit logs yet.")

# =========================
# Sidebar footer
# =========================
st.sidebar.markdown("---")

# -- 访问统计 --
stats = get_stats()
col_a, col_b = st.sidebar.columns(2)
col_a.metric("👥 Users", stats["total_users"])
col_b.metric("👀 Visits", stats["total_visits"])
st.sidebar.caption(f"📅 Today: {stats['today_visits']} visits")

st.sidebar.markdown("---")
st.sidebar.markdown(f"👤 **{st.session_state.username}**")
if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.visit_logged = False
    st.session_state.auth_mode = "login"
    st.rerun()
st.sidebar.caption("Water Music Pavilion v1.0")