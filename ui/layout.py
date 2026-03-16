from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7f7;
            --sidebar-bg: #e8eeef;
            --surface: #ffffff;
            --surface-soft: #f8fbfb;
            --border: #d6e0e0;
            --border-strong: #c6d3d3;
            --text: #162126;
            --muted: #5d6870;
            --accent: #0f6f78;
            --accent-soft: #d9eff0;
            --shadow: 0 12px 30px rgba(15, 23, 31, 0.05);
        }

        html, body, [class*="css"] {
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: var(--bg);
        }

        .stApp,
        .stApp > div,
        [data-testid="stMain"],
        section.main {
            background: var(--bg) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(244, 247, 247, 0.92);
        }

        [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--border);
        }

        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottom"] {
            background: var(--bg) !important;
            border-top: 1px solid var(--border);
        }

        [data-testid="stChatFloatingInputContainer"],
        [data-testid="stChatInputContainer"] {
            background: var(--bg) !important;
            border-top: 1px solid var(--border);
            box-shadow: none !important;
        }

        div[class*="stBottom"],
        div[class*="stBottomBlockContainer"],
        div[class*="stChatFloatingInputContainer"],
        div[class*="stChatInputContainer"],
        footer {
            background: var(--bg) !important;
            border-top-color: var(--border) !important;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.35rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        h1,
        [data-testid="stMarkdownContainer"] h1 {
            margin: 0;
            font-size: 2.55rem;
            line-height: 1.05;
            letter-spacing: -0.03em;
            color: var(--text) !important;
        }

        h2, h3, h4,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: var(--text) !important;
            letter-spacing: -0.02em;
        }

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] div,
        [data-testid="stCaptionContainer"] {
            color: var(--text);
        }

        .app-header {
            padding: 0.2rem 0 0.6rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 1rem;
        }

        .app-subtitle {
            margin-top: 0.45rem;
            max-width: 760px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
        }

        .eyebrow {
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.45rem;
        }

        .thread-context-bar {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 1rem 1.15rem;
            box-shadow: var(--shadow);
            margin: 0.9rem 0 1.35rem;
        }

        .thread-context-title {
            font-size: 1.1rem;
            font-weight: 600;
            line-height: 1.35;
            color: var(--text);
        }

        .thread-context-meta {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.45rem;
        }

        .scope-chip,
        .meta-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.34rem 0.7rem;
            font-size: 0.82rem;
            line-height: 1;
            white-space: nowrap;
        }

        .scope-chip {
            background: var(--accent-soft);
            color: var(--accent);
            font-weight: 600;
        }

        .meta-chip {
            background: var(--surface-soft);
            color: var(--muted);
            border: 1px solid var(--border);
        }

        .sidebar-brand {
            margin-bottom: 1rem;
        }

        .sidebar-title {
            font-size: 1.15rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--text);
            margin-bottom: 0.15rem;
        }

        .sidebar-copy {
            font-size: 0.9rem;
            line-height: 1.55;
            color: var(--muted);
            margin-bottom: 1rem;
        }

        .sidebar-label {
            font-size: 0.74rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            margin: 1rem 0 0.55rem;
        }

        .sidebar-summary {
            font-size: 0.88rem;
            color: var(--muted);
            margin-top: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .thread-item {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.45rem;
        }

        .thread-item.active {
            background: var(--surface);
            border-color: rgba(15, 111, 120, 0.28);
            box-shadow: inset 3px 0 0 var(--accent);
        }

        .thread-item-title {
            font-size: 0.95rem;
            font-weight: 600;
            line-height: 1.42;
            color: var(--text);
        }

        .thread-item-meta {
            margin-top: 0.28rem;
            font-size: 0.8rem;
            color: var(--muted);
            line-height: 1.4;
        }

        .thread-sep {
            color: #9cadb2;
            padding: 0 0.18rem;
        }

        .active-pill {
            display: inline-flex;
            margin-top: 0.35rem;
            margin-bottom: 0.9rem;
            padding: 0.24rem 0.62rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--accent);
            background: var(--accent-soft);
        }

        .empty-state {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(247, 251, 251, 0.96));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: var(--shadow);
            margin: 1.2rem 0 1rem;
        }

        .empty-state-title {
            font-size: 1.5rem;
            font-weight: 600;
            line-height: 1.2;
            color: var(--text);
        }

        .empty-state-copy {
            margin-top: 0.65rem;
            max-width: 720px;
            font-size: 1rem;
            line-height: 1.7;
            color: var(--muted);
        }

        .sources-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-top: 0.8rem;
            margin-bottom: 0.7rem;
        }

        .sources-title {
            font-size: 1rem;
            font-weight: 600;
            line-height: 1.35;
            color: var(--text);
        }

        .sources-stats {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.4rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border-color: var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
        }

        [data-testid="stChatMessage"] {
            background: transparent;
            padding-top: 0.2rem;
            padding-bottom: 0.2rem;
        }

        [data-testid="stChatMessageContent"],
        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] li,
        [data-testid="stChatMessageContent"] div,
        [data-testid="stChatMessageContent"] span {
            color: var(--text) !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: var(--shadow);
            padding: 0.15rem 0.2rem;
            max-width: 1120px;
            margin: 0 auto;
        }

        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] [data-baseweb="textarea"],
        [data-testid="stChatInput"] [data-baseweb="base-input"],
        [data-testid="stChatInput"] div:has(> textarea[data-testid="stChatInputTextArea"]),
        [data-testid="stChatInput"] div:has(> button[data-testid="stChatInputSubmitButton"]) {
            background: var(--surface) !important;
            border-color: var(--border) !important;
        }

        [data-testid="stChatInput"] textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stTextArea"] textarea {
            background: var(--surface) !important;
            color: var(--text) !important;
            border-radius: 14px !important;
        }

        [data-testid="stChatInputSubmitButton"] {
            background: var(--surface-soft) !important;
            border: 1px solid var(--border) !important;
            color: var(--muted) !important;
            box-shadow: none !important;
        }

        [data-testid="stChatInputSubmitButton"]:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
        }

        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stChatInput"] textarea::placeholder {
            color: #8c989f !important;
        }

        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {
            border-radius: 12px;
            border: 1px solid var(--border-strong);
            background: var(--surface);
            color: var(--text);
            font-weight: 600;
            min-height: 2.7rem;
        }

        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        [data-testid="stButton"] button[kind="primary"] {
            background: var(--accent);
            color: #ffffff;
            border-color: var(--accent);
        }

        [data-testid="stButton"] button[kind="primary"]:hover {
            background: #0b626a;
            color: #ffffff;
        }

        [data-testid="stExpander"] details {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
        }

        [data-testid="stExpander"] summary {
            background: var(--surface) !important;
            color: var(--text) !important;
            border-radius: 16px;
        }

        [data-testid="stExpander"] summary * {
            color: var(--text) !important;
            fill: var(--text) !important;
        }

        [data-testid="stTabs"] button[role="tab"] {
            border-radius: 999px;
            background: transparent;
            color: var(--muted);
            font-weight: 600;
            padding: 0.35rem 0.8rem;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--accent-soft);
            color: var(--accent);
        }

        hr {
            border-color: var(--border);
        }

        @media (max-width: 1100px) {
            .thread-context-bar,
            .sources-header {
                flex-direction: column;
            }

            .thread-context-meta,
            .sources-stats {
                justify-content: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div class="eyebrow">Grounded legal research</div>
            <h1>Indian Legal Advisor</h1>
            <div class="app-subtitle">Ask about sections, articles, procedures, and risks with retrieval-grounded answers and supporting sources.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
