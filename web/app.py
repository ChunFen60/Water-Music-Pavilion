import os
import sys
import time
import tempfile

import pandas as pd
import plotly.express as px
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

    # -- 数据表预览 --
    st.subheader("🧾 Full Dataset Preview")
    st.dataframe(df.head(50))

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

    selected_composer = st.sidebar.selectbox(
        "Select Composer",
        composer_list
    )

    composer_df = df[
        df[composer_col].astype(str)
        == selected_composer
    ]

    all_others_df = df[
        df[composer_col].astype(str)
        != selected_composer
    ]

    st.header(
        f"🎼 Composer Style Profile: {selected_composer}"
    )

    # -- 关键风格指标 --
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Pieces", len(composer_df))
    col2.metric(
        "Pitch Range",
        f"{composer_df['pitch_range'].mean():.0f} st"
    )
    col3.metric(
        "Note Density",
        f"{composer_df['note_density'].mean():.1f}/s"
    )
    col4.metric(
        "Tempo",
        f"{composer_df['tempo'].mean():.0f} BPM"
    )
    col5.metric(
        "Melodic Complexity",
        f"{composer_df['melodic_complexity'].mean():.1f}"
    )

    # -- 风格雷达图：五维对比 --
    st.subheader("🎹 Five-Dimensional Style Radar")

    import plotly.graph_objects as go

    dimensions = ["pitch_range", "note_density", "tempo", "rhythm_std", "melodic_complexity"]
    dim_labels = ["Pitch Range", "Note Density", "Tempo", "Rhythm Variation", "Melodic Complexity"]

    # 归一化到 0-1
    norms = {}
    for d in dimensions:
        vmin, vmax = df[d].min(), df[d].max()
        norms[d] = (vmin, vmax)

    composer_vals = []
    global_vals = []
    for d in dimensions:
        vmin, vmax = norms[d]
        c_val = (composer_df[d].mean() - vmin) / (vmax - vmin)
        g_val = (df[d].mean() - vmin) / (vmax - vmin)
        composer_vals.append(round(c_val, 3))
        global_vals.append(round(g_val, 3))

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=composer_vals + [composer_vals[0]],
        theta=dim_labels + [dim_labels[0]],
        fill='toself',
        name=selected_composer,
        line=dict(color='#4F6FDE', width=2),
        fillcolor='rgba(79, 111, 222, 0.25)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=global_vals + [global_vals[0]],
        theta=dim_labels + [dim_labels[0]],
        fill='toself',
        name='Global Average',
        line=dict(color='#aaa', width=1.5, dash='dash'),
        fillcolor='rgba(170, 170, 170, 0.1)'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
        height=420,
        margin=dict(t=40, b=40, l=60, r=60),
        legend=dict(orientation="h", y=-0.1)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # -- 风格散点：该作曲家 vs 其他 --
    st.subheader("🎹 Style Position: Range × Density")

    composer_df_display = composer_df.copy()
    composer_df_display["Group"] = selected_composer
    all_others_display = all_others_df.copy()
    all_others_display["Group"] = "Other Composers"

    combined = pd.concat([composer_df_display, all_others_display])

    fig_pos = px.scatter(
        combined,
        x="pitch_range",
        y="note_density",
        color="Group",
        size="tempo",
        hover_name="title",
        title=f"{selected_composer} vs All Others",
        color_discrete_map={
            selected_composer: "#4F6FDE",
            "Other Composers": "#ccc"
        },
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
        title=f"{selected_composer} — Rhythm × Melody Space",
        color_discrete_map={
            selected_composer: "#4F6FDE",
            "Other Composers": "#ccc"
        },
        labels={
            "rhythm_std": "Rhythm Variation (std)",
            "melodic_complexity": "Melodic Complexity"
        },
        opacity=0.8
    )
    fig_prof.update_layout(height=420)
    st.plotly_chart(fig_prof, use_container_width=True)

    # -- 作品列表 --
    st.subheader("🧾 Works by " + selected_composer)
    st.dataframe(
        composer_df[["title", "emotion", "pitch_range", "note_density", "tempo", "melodic_complexity"]],
        use_container_width=True,
        height=600
    )

# =========================
# Section: MIDI Analysis
# =========================
elif nav_option == "🎵 MIDI Analysis":

    st.header("🎵 MIDI Upload & Analysis")

    if not MIDI_AVAILABLE:
        st.warning("⚠ MIDI analysis is not available in the cloud deployment. Please run the app locally for MIDI features.")
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

                    st.success("✅ MIDI Analysis Complete")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric("Average Pitch", result["avg_pitch"])
                    col2.metric("Average Duration", result["avg_duration"])
                    col3.metric("Average Velocity", result["avg_velocity"])
                    col4.metric("Total Notes", result["total_notes"])

                    st.subheader("🎭 Emotion Prediction")
                    st.info(result["emotion"])

                    st.subheader("🎹 Piano Roll Visualization")

                    notes_df_piano = pd.DataFrame(result["notes_data"])

                    fig_roll = px.scatter(
                        notes_df_piano,
                        x="start",
                        y="pitch",
                        size="duration",
                        color="velocity",
                        title="Interactive Piano Roll",
                        hover_data=[
                            "start", "end", "duration",
                            "pitch", "velocity"
                        ]
                    )
                    fig_roll.update_layout(height=600)
                    st.plotly_chart(fig_roll, use_container_width=True)

                    st.subheader("🎼 Composer Similarity Analysis")

                    similarity_results = []
                    composer_names = (
                        df[composer_col]
                        .dropna()
                        .astype(str)
                        .unique()
                    )

                    for composer in composer_names:

                        comp_df = df[
                            df[composer_col].astype(str)
                            == composer
                        ]

                        if pitch_col is None or duration_col is None:
                            continue

                        comp_pitch = comp_df[pitch_col].mean()
                        comp_duration = comp_df[duration_col].mean()

                        distance = (
                            (result["avg_pitch"] - comp_pitch) ** 2
                            + (result["avg_duration"] - comp_duration) ** 2
                        ) ** 0.5

                        similarity_results.append({
                            "composer": composer,
                            "distance": round(distance, 2)
                        })

                    similarity_df = pd.DataFrame(
                        similarity_results
                    ).sort_values("distance")

                    top_matches = similarity_df.head(5)

                    st.dataframe(top_matches)

                    best_match = top_matches.iloc[0]
                    st.success(
                        f"Most Similar Composer Style: "
                        f"{best_match['composer']}"
                    )

                    if client:

                        prompt = f"""
Analyze this piano music.

Features:
- Average Pitch: {result["avg_pitch"]}
- Average Duration: {result["avg_duration"]}
- Average Velocity: {result["avg_velocity"]}
- Total Notes: {result["total_notes"]}
- Emotion: {result["emotion"]}

Please describe:
1. Musical style
2. Emotional characteristics
3. Possible composer similarity
4. Performance feeling
"""

                        with st.spinner("AI Analyzing Music..."):

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
                                    {"role": "user", "content": prompt}
                                ]
                            )

                            answer = (
                                response
                                .choices[0]
                                .message
                                .content
                            )

                            st.subheader("🤖 AI Music Interpretation")
                            st.success(answer)

            except Exception as e:
                st.error(f"MIDI Analysis Error: {e}")

# =========================
# Section: Emotion Analysis
# =========================
elif nav_option == "🎭 Emotion Analysis":

    st.header("🎭 Emotion Analysis Dashboard")

    # -- 检查 emotion 字段 --
    emotion_col = next(
        (c for c in df.columns if "emotion" in c.lower()),
        None
    )

    if emotion_col:
        st.subheader("😊 Emotion Distribution")

        emotion_counts = (
            df[emotion_col]
            .dropna()
            .astype(str)
            .value_counts()
            .reset_index()
        )
        emotion_counts.columns = ["Emotion", "Count"]

        fig_pie = px.pie(
            emotion_counts,
            names="Emotion",
            values="Count",
            title="Emotion Distribution in Dataset",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # -- Emotion by Composer --
        st.subheader("🎼 Emotion per Composer")

        composer_emotion = (
            df.groupby([composer_col, emotion_col])
            .size()
            .reset_index(name="Count")
        )

        fig_heat = px.density_heatmap(
            composer_emotion,
            x=composer_col,
            y=emotion_col,
            z="Count",
            title="Emotion Distribution by Composer",
            labels={emotion_col: "Emotion"}
        )
        fig_heat.update_layout(height=450)
        st.plotly_chart(fig_heat, use_container_width=True)

    else:
        st.info("No emotion field found in dataset.")

    # -- 音乐特征按情绪分组 --
    st.subheader("🎹 Musical Features by Emotion")

    # 选两个最重要的特征做对比
    col_left, col_right = st.columns(2)

    with col_left:
        fig_range_emo = px.box(
            df,
            x=emotion_col,
            y="pitch_range",
            color=emotion_col,
            title="Pitch Range by Emotion",
            labels={"pitch_range": "Pitch Range (semitones)"}
        )
        fig_range_emo.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_range_emo, use_container_width=True)

    with col_right:
        fig_dens_emo = px.box(
            df,
            x=emotion_col,
            y="note_density",
            color=emotion_col,
            title="Note Density by Emotion",
            labels={"note_density": "Note Density (notes/sec)"}
        )
        fig_dens_emo.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_dens_emo, use_container_width=True)

    # -- 情绪风格空间 --
    st.subheader("🎵 Emotion Style Space: Range × Density")

    fig_emo_space = px.scatter(
        df,
        x="pitch_range",
        y="note_density",
        color=emotion_col,
        size="tempo",
        hover_name="title",
        title="Works Colored by Emotion — Pitch Range vs Note Density",
        labels={
            "pitch_range": "Pitch Range (semitones)",
            "note_density": "Note Density (notes/sec)"
        },
        opacity=0.75
    )
    fig_emo_space.update_layout(height=480)
    st.plotly_chart(fig_emo_space, use_container_width=True)

    # -- Tempo vs Melodic Complexity by Emotion --
    st.subheader("🎼 Tempo × Melodic Complexity by Emotion")

    fig_tm = px.scatter(
        df,
        x="tempo",
        y="melodic_complexity",
        color=emotion_col,
        hover_name="title",
        title="Tempo vs Melodic Complexity Colored by Emotion",
        labels={
            "tempo": "Tempo (BPM)",
            "melodic_complexity": "Melodic Complexity"
        },
        opacity=0.75
    )
    fig_tm.update_layout(height=460)
    st.plotly_chart(fig_tm, use_container_width=True)

    # -- 相关矩阵 --
    st.subheader("🔗 Feature Correlation Matrix")

    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr()

        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            title="Numeric Feature Correlations",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
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
        search_name = st.text_input("Enter username to search", placeholder="Type a username...")
        col1, col2 = st.columns([1, 3])
        with col1:
            search_btn = st.button("Search", type="primary", use_container_width=True)

        if search_btn and search_name.strip():
            if search_name.strip() == username:
                st.warning("That's you!")
            else:
                from web.auth import user_exists
                if user_exists(search_name.strip()):
                    found_user = search_name.strip()
                    st.success(f"User **{found_user}** found!")
                    if are_friends(username, found_user):
                        st.info(f"You are already friends with {found_user}.")
                    else:
                        if st.button(f"➕ Send Friend Request to {found_user}", type="primary"):
                            ok, msg = send_friend_request(username, found_user)
                            if ok:
                                st.success(msg)
                            else:
                                st.warning(msg)
                else:
                    st.error(f"User '{search_name.strip()}' not found.")

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
        st.dataframe(
            user_data,
            use_container_width=True,
            hide_index=True
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
        st.dataframe(
            activity_data,
            use_container_width=True,
            hide_index=True
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
        st.dataframe(
            log_data,
            use_container_width=True,
            hide_index=True
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
