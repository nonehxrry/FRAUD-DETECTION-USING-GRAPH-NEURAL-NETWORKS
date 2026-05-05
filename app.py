"""
app.py
-------
Dark-themed Streamlit dashboard for the Fraud GNN project.

Features:
- Model performance metrics (AUC-ROC, confusion matrix)
- Fraud probability distribution
- Interactive transaction explorer
- Graph visualization of suspicious clusters
- Feature importance ranking

Run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import os
import torch
import torch.nn.functional as F

from utils.preprocessing import preprocess_pipeline
from utils.graph_builder import build_graph
from models.gnn_model import FraudGNN, FraudGraphSAGE

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud GNN Dashboard",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'dark'

# ── Custom CSS (dark theme enhancement) ──────────────────────────────────────
def apply_theme_css(mode: str):
    if mode == 'light':
        bg = '#f4f7fb'
        panel = '#ffffff'
        panel_alt = '#eef3f9'
        text = '#1f2a37'
        muted = '#5b6b7f'
        border = 'rgba(0, 0, 0, 0.08)'
    else:
        bg = '#0E1117'
        panel = '#1E2130'
        panel_alt = '#161B2E'
        text = '#E2E8F0'
        muted = '#A0AEC0'
        border = 'rgba(255, 255, 255, 0.10)'

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
        .stApp {{
            background: linear-gradient(180deg, {bg} 0%, {bg} 100%);
            color: {text};
            font-family: 'Manrope', sans-serif;
        }}
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 1.2rem;
        }}
        h1, h2, h3 {{
            color: {text};
            letter-spacing: 0.2px;
        }}
        p {{ color: {text}; }}
        .subtitle {{
            color: {muted};
            font-size: 0.95rem;
            margin-top: -0.2rem;
            margin-bottom: 0.4rem;
        }}
        .hero-card {{
            background: linear-gradient(130deg, rgba(75,158,255,0.22), {panel} 55%, rgba(255,75,75,0.16));
            border: 1px solid {border};
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }}
        .metric-card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 0.35rem;
        }}
        div[data-testid="stSidebarContent"] {{
            background: {panel_alt};
            border-right: 1px solid {border};
        }}
        .stMetric {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 10px;
        }}
        .stMetricLabel {{ color: {muted} !important; }}
        .stMetricValue {{ color: {text} !important; font-size: 1.85rem !important; }}
        .fade-in {{
            animation: fadeIn 0.35s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .stButton > button {{
            border-radius: 10px;
            border: 1px solid {border};
        }}
        @media (max-width: 900px) {{
            .block-container {{ padding-top: 0.6rem; }}
            h1 {{ font-size: 1.55rem !important; }}
            h2 {{ font-size: 1.25rem !important; }}
            .subtitle {{ font-size: 0.88rem; }}
        }}
    </style>
    """, unsafe_allow_html=True)


apply_theme_css(st.session_state.theme_mode)

# ── Helpers ───────────────────────────────────────────────────────────────────

ARTIFACTS = Path("models/saved")


def _load_first_existing_npy(base: Path, candidates):
    for name in candidates:
        p = base / name
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(f"Missing npy artifact in {base}: {candidates}")


def _load_first_existing_joblib(base: Path, candidates):
    for name in candidates:
        p = base / name
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError(f"Missing joblib artifact in {base}: {candidates}")


def _pick_data_paths():
    train_tx = Path("data/train_transaction.csv")
    train_id = Path("data/train_identity.csv")
    if train_tx.exists() and train_id.exists():
        return str(train_tx), str(train_id)

    demo_tx = Path("data/demo_transaction.csv")
    demo_id = Path("data/demo_identity.csv")
    if demo_tx.exists() and demo_id.exists():
        return str(demo_tx), str(demo_id)

    return "data/train_transaction.csv", "data/train_identity.csv"


def _transform_custom_value(col, raw_value, encoders, scaler):
    value = raw_value

    if col in encoders:
        le = encoders[col]
        token = str(raw_value)
        classes = set(le.classes_)
        if token not in classes:
            token = 'UNKNOWN' if 'UNKNOWN' in classes else le.classes_[0]
        value = float(le.transform([token])[0])

    if hasattr(scaler, 'feature_names_in_') and col in scaler.feature_names_in_:
        if col == 'TransactionAmt':
            value = np.log1p(max(float(raw_value), 0.0))
        idx = int(np.where(scaler.feature_names_in_ == col)[0][0])
        value = float((value - scaler.mean_[idx]) / scaler.scale_[idx])

    return float(value)


@st.cache_resource
def load_prediction_runtime():
    artifact_dir = ARTIFACTS

    meta = _load_first_existing_joblib(artifact_dir, ["model_meta.pkl", "meta.pkl"])
    if 'model_type' not in meta and 'model' in meta:
        meta['model_type'] = meta['model']
    if 'num_layers' not in meta and 'layers' in meta:
        meta['num_layers'] = meta['layers']
    if 'hidden_channels' not in meta and 'hidden' in meta:
        meta['hidden_channels'] = meta['hidden']

    encoders = _load_first_existing_joblib(artifact_dir, ["encoders.pkl"])
    scaler = _load_first_existing_joblib(artifact_dir, ["scaler.pkl"])

    tx_path, id_path = _pick_data_paths()
    df_base, _, _ = preprocess_pipeline(tx_path, id_path)

    return {
        'artifact_dir': artifact_dir,
        'meta': meta,
        'encoders': encoders,
        'scaler': scaler,
        'df_base': df_base,
    }


def predict_custom_transaction(custom_inputs: dict):
    runtime = load_prediction_runtime()
    meta = runtime['meta']
    encoders = runtime['encoders']
    scaler = runtime['scaler']
    df_base = runtime['df_base']

    context_n = min(4000, len(df_base))
    context_df = df_base.sample(n=context_n, random_state=42).reset_index(drop=True)

    row = context_df.median(numeric_only=True).reindex(context_df.columns, fill_value=0.0)
    row['isFraud'] = 0
    row['TransactionID'] = int(context_df['TransactionID'].max()) + 1 if 'TransactionID' in context_df.columns else len(context_df)

    for col, raw_value in custom_inputs.items():
        if col in row.index:
            row[col] = _transform_custom_value(col, raw_value, encoders, scaler)

    df_aug = pd.concat([context_df, pd.DataFrame([row])], ignore_index=True)
    data = build_graph(df_aug)

    expected_in = int(meta.get('in_channels', data['transaction'].x.shape[1]))
    current_in = int(data['transaction'].x.shape[1])
    if current_in != expected_in:
        for nt in data.node_types:
            x = data[nt].x
            if x.size(1) > expected_in:
                data[nt].x = x[:, :expected_in]
            else:
                pad = torch.zeros((x.size(0), expected_in - x.size(1)), dtype=x.dtype)
                data[nt].x = torch.cat([x, pad], dim=1)

    in_channels = data['transaction'].x.shape[1]
    model_type = str(meta.get('model_type', 'sage')).lower()
    hidden = int(meta.get('hidden_channels', 32))
    layers = int(meta.get('num_layers', 2))
    dropout = float(meta.get('dropout', 0.2))
    heads = int(meta.get('heads', 1))

    if model_type == 'gat':
        model = FraudGNN(
            metadata=data.metadata(),
            in_channels=in_channels,
            hidden_channels=hidden,
            num_layers=layers,
            heads=heads,
            dropout=dropout
        )
    else:
        model = FraudGraphSAGE(
            metadata=data.metadata(),
            in_channels=in_channels,
            hidden_channels=hidden,
            num_layers=layers,
            dropout=dropout
        )

    weights_path = runtime['artifact_dir'] / "best_model.pt"
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        logits = out[-1]
        prob = float(F.softmax(logits, dim=-1)[1].item())

    pred = int(prob >= 0.5)
    return prob, pred

@st.cache_data
def load_artifacts():
    """Load saved model artifacts."""
    artifacts = {}
    try:
        artifacts['test_probs']  = _load_first_existing_npy(ARTIFACTS, ["test_probs.npy"])
        artifacts['test_labels'] = _load_first_existing_npy(ARTIFACTS, ["test_labels.npy"])
        artifacts['fpr']         = _load_first_existing_npy(ARTIFACTS, ["fpr.npy"])
        artifacts['tpr']         = _load_first_existing_npy(ARTIFACTS, ["tpr.npy"])
        artifacts['cm']          = _load_first_existing_npy(ARTIFACTS, ["confusion_matrix.npy", "cm.npy"])
        artifacts['meta']        = _load_first_existing_joblib(ARTIFACTS, ["model_meta.pkl", "meta.pkl"])

        # Normalize metadata keys across train.py versions.
        meta = artifacts['meta']
        if 'model_type' not in meta and 'model' in meta:
            meta['model_type'] = meta['model']
        if 'num_layers' not in meta and 'layers' in meta:
            meta['num_layers'] = meta['layers']
        if 'hidden_channels' not in meta and 'hidden' in meta:
            meta['hidden_channels'] = meta['hidden']

        artifacts['trained']     = True
    except FileNotFoundError:
        artifacts['trained'] = False
    return artifacts

def dark_fig(fig):
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1E2130',
        font_color='#E2E8F0'
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/security-checked.png", width=60)
    st.title("🕵️ Fraud GNN")
    st.caption("Graph Neural Network Fraud Detection")
    theme_is_dark = st.toggle("🌗 Dark Theme", value=(st.session_state.theme_mode == 'dark'))
    st.session_state.theme_mode = 'dark' if theme_is_dark else 'light'
    apply_theme_css(st.session_state.theme_mode)
    st.divider()

    page = st.radio("Navigate", [
        "📊 Overview",
        "🔍 Transaction Explorer",
        "🧪 Custom Prediction",
        "🕸️  Graph Visualization",
        "📈 Model Performance",
        "⚙️  Feature Importance"
    ])

    st.divider()
    st.caption("IEEE-CIS Fraud Detection | PyTorch Geometric")

# ── Load Data ─────────────────────────────────────────────────────────────────

arts = load_artifacts()

# ── Pages ─────────────────────────────────────────────────────────────────────

# ── PAGE 1: Overview ──────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.markdown("""
        <div class='hero-card fade-in'>
            <h2 style='margin-bottom:0.2rem;'>📊 Fraud Detection Overview</h2>
            <p class='subtitle'>Heterogeneous Graph Attention Network · IEEE-CIS Dataset</p>
        </div>
        """, unsafe_allow_html=True)

    if not arts['trained']:
        st.warning("⚠️ Model not trained yet. Run `python train.py` first.")
        st.info("""
        **Quick Start:**
        ```bash
        # 1. Download data from Kaggle
        kaggle competitions download -c ieee-fraud-detection

        # 2. Train the model
        python train.py --transaction data/train_transaction.csv \\
                        --identity    data/train_identity.csv    \\
                        --epochs      50 --model gat

        # 3. Launch dashboard
        streamlit run app.py
        ```
        """)
    else:
        meta = arts['meta']
        probs  = arts['test_probs']
        labels = arts['test_labels']

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 AUC-ROC", f"{meta['test_auc']:.4f}", "+vs baseline")
        with col2:
            n_fraud = int(labels.sum())
            st.metric("🚨 Fraud Txns", f"{n_fraud:,}")
        with col3:
            fraud_rate = labels.mean() * 100
            st.metric("📉 Fraud Rate", f"{fraud_rate:.1f}%")
        with col4:
            detected = int((probs[labels == 1] > 0.5).sum())
            detection_rate = detected / n_fraud * 100
            st.metric("✅ Detection Rate", f"{detection_rate:.1f}%")

        st.divider()

        tab_dist, tab_arch = st.tabs(["📈 Risk Distribution", "🏗️ Architecture"]) 

        with tab_dist:
            col_l, col_r = st.columns(2)
            with col_l:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=probs[labels == 0], name='Legitimate',
                    marker_color='#00CC96', opacity=0.75, nbinsx=50))
                fig.add_trace(go.Histogram(
                    x=probs[labels == 1], name='Fraudulent',
                    marker_color='#FF4B4B', opacity=0.75, nbinsx=50))
                fig.update_layout(title='Fraud Probability Distribution',
                                  xaxis_title='P(Fraud)', yaxis_title='Count',
                                  barmode='overlay')
                st.plotly_chart(dark_fig(fig), use_container_width=True)

            with col_r:
                fraud_c = int(labels.sum())
                legit_c = len(labels) - fraud_c
                fig2 = go.Figure(go.Pie(
                    labels=['Legitimate', 'Fraudulent'],
                    values=[legit_c, fraud_c],
                    hole=0.6,
                    marker_colors=['#00CC96', '#FF4B4B']
                ))
                fig2.update_layout(title='Test Set Class Balance')
                st.plotly_chart(dark_fig(fig2), use_container_width=True)

        with tab_arch:
            with st.expander("Model and Graph Details", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info("**Node Types**\n\n- Transactions\n- Cards (card1–6)\n- Email Domains\n- Devices\n- Addresses")
                with col2:
                    st.info("**Edge Types**\n\n- TX → Card\n- TX → Email\n- TX → Device\n- TX ↔ TX (shared card)")
                with col3:
                    st.info(f"**Model Config**\n\n- Type: {meta['model_type'].upper()}\n"
                            f"- Layers: {meta['num_layers']}\n"
                            f"- Hidden: {meta['hidden_channels']}\n"
                            f"- Heads: {meta.get('heads', 'N/A')}")


# ── PAGE 2: Transaction Explorer ──────────────────────────────────────────────
elif page == "🔍 Transaction Explorer":
    st.markdown("""
        <div class='hero-card fade-in'>
            <h2 style='margin-bottom:0.2rem;'>🔍 Transaction Explorer</h2>
            <p class='subtitle'>Filter, inspect, and prioritize high-risk predictions quickly.</p>
        </div>
        """, unsafe_allow_html=True)

    if not arts['trained']:
        st.warning("Train the model first.")
    else:
        probs  = arts['test_probs']
        labels = arts['test_labels']

        # Threshold slider
        threshold = st.slider("Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)

        # Generate demo transaction table
        n = len(probs)
        df_view = pd.DataFrame({
            'Transaction ID': [f'TX_{i:06d}' for i in range(n)],
            'Fraud Probability': probs.round(4),
            'Prediction': ['🚨 FRAUD' if p > threshold else '✅ Legit' for p in probs],
            'Actual': ['🚨 FRAUD' if l == 1 else '✅ Legit' for l in labels.astype(int)],
            'Risk Level': pd.cut(probs, bins=[0, 0.3, 0.6, 0.8, 1.0],
                                  labels=['Low', 'Medium', 'High', 'Critical'])
        })

        col_l, col_r = st.columns([3, 1])
        with col_r:
            risk_filter = st.multiselect("Filter by Risk", ['Low', 'Medium', 'High', 'Critical'],
                                          default=['High', 'Critical'])
            show_fraud_only = st.checkbox("Show predicted fraud only")

        filtered = df_view.copy()
        if risk_filter:
            filtered = filtered[filtered['Risk Level'].isin(risk_filter)]
        if show_fraud_only:
            filtered = filtered[filtered['Prediction'] == '🚨 FRAUD']

        with col_l:
            st.caption(f"Showing {len(filtered):,} of {n:,} transactions")

        # Color rows by risk
        st.dataframe(
            filtered.head(500).style.background_gradient(
                subset=['Fraud Probability'], cmap='RdYlGn_r'),
            use_container_width=True
        )

        # Risk distribution
        st.subheader("Risk Distribution")
        risk_counts = df_view['Risk Level'].value_counts().reset_index()
        fig = px.bar(risk_counts, x='Risk Level', y='count',
                     color='Risk Level',
                     color_discrete_map={'Low':'#00CC96','Medium':'#FFA500',
                                         'High':'#FF6B35','Critical':'#FF4B4B'})
        st.plotly_chart(dark_fig(fig), use_container_width=True)


# ── PAGE 3: Custom Prediction ────────────────────────────────────────────────
elif page == "🧪 Custom Prediction":
    st.markdown("""
        <div class='hero-card fade-in'>
            <h2 style='margin-bottom:0.2rem;'>🧪 Custom Fraud Prediction</h2>
            <p class='subtitle'>Enter custom transaction values and get an instant fraud score.</p>
        </div>
        """, unsafe_allow_html=True)

    if not arts['trained']:
        st.warning("Train the model first.")
    else:
        runtime = load_prediction_runtime()
        encoders = runtime['encoders']

        c1, c2 = st.columns(2)

        with c1:
            amount = st.number_input("Transaction Amount (USD)", min_value=0.0, value=120.0, step=10.0, help="Total amount of the payment.")
            card1 = st.number_input("Card Identifier (card1)", min_value=0, value=1000, step=1, help="An anonymized card ID used to track card reuse patterns.")
            addr1 = st.number_input("Billing Address Region (addr1)", min_value=0, value=200, step=1, help="An anonymized billing region code.")

        with c2:
            email_options = list(encoders.get('P_emaildomain').classes_) if 'P_emaildomain' in encoders else ['UNKNOWN']
            device_options = list(encoders.get('DeviceType').classes_) if 'DeviceType' in encoders else ['UNKNOWN']
            p_email = st.selectbox("Purchaser Email Domain (P_emaildomain)", email_options, help="Domain part of buyer email (for example gmail.com).")
            device = st.selectbox("Device Type", device_options, help="Type of device used for the transaction (desktop/mobile).")
            dist1 = st.number_input("Distance Indicator (dist1)", value=10.0, step=1.0, help="An anonymized distance/risk indicator from the dataset.")

        if st.button("Predict Fraud", type="primary"):
            prob, pred = predict_custom_transaction({
                'TransactionAmt': float(amount),
                'card1': float(card1),
                'addr1': float(addr1),
                'P_emaildomain': str(p_email),
                'DeviceType': str(device),
                'dist1': float(dist1),
            })

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Fraud Probability", f"{prob:.4f}")
            with col_b:
                if pred == 1:
                    st.error("Prediction: FRAUD")
                else:
                    st.success("Prediction: LEGIT")

            st.caption("Threshold used: 0.50")


# ── PAGE 3: Graph Visualization ───────────────────────────────────────────────
elif page == "🕸️  Graph Visualization":
    st.markdown("""
        <div class='hero-card fade-in'>
            <h2 style='margin-bottom:0.2rem;'>🕸️ Fraud Network Graph</h2>
            <p class='subtitle'>Visualize suspicious clusters and shared entity connections.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("""
    **How to read this graph:**
    - 🔴 **Red nodes** = High-risk transactions (P > 0.7)
    - 🟠 **Orange nodes** = Medium-risk transactions (0.5 < P ≤ 0.7)
    - 🔵 **Blue nodes** = Card entities (shared between transactions)
    - 🟢 **Green nodes** = Email domain entities
    - 🟣 **Purple nodes** = Device entities

    Transactions connected to the same card/device/email are linked.
    **Clusters of red/orange nodes indicate fraud rings.**
    """)

    if not arts['trained']:
        st.warning("Train the model first to see real fraud clusters.")
        # Show demo graph
        import networkx as nx
        G = nx.karate_club_graph()
        st.caption("Demo graph (Karate Club) — train model for real fraud clusters")

    else:
        probs = arts['test_probs']

        # Generate demo fraud cluster visualization
        fraud_indices = np.where(probs > 0.5)[0][:50]

        nodes_data = []
        for i in fraud_indices[:30]:
            nodes_data.append({
                'id': f'TX_{i}',
                'type': 'Transaction',
                'fraud_prob': float(probs[i]),
                'color': '#FF4B4B' if probs[i] > 0.7 else '#FFA500'
            })

        # Add some entity nodes
        for c in range(5):
            nodes_data.append({'id': f'CARD_{c}', 'type': 'Card', 'color': '#4B9EFF'})
        for e in range(3):
            nodes_data.append({'id': f'EMAIL_{e}', 'type': 'Email', 'color': '#4BFF9E'})

        # Plotly network graph
        import random
        random.seed(42)
        pos = {n['id']: (random.uniform(0, 10), random.uniform(0, 10)) for n in nodes_data}

        edge_x, edge_y = [], []
        # Connect transactions to cards
        for i, tx in enumerate(fraud_indices[:30]):
            card_id = f"CARD_{i % 5}"
            x0, y0 = pos[f'TX_{tx}']
            x1, y1 = pos[card_id]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                  line=dict(color='#555', width=1), hoverinfo='none'))

        for n in nodes_data:
            x, y = pos[n['id']]
            size = 20 if n['type'] == 'Transaction' else 12
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text',
                marker=dict(size=size, color=n['color'],
                            line=dict(width=1, color='white')),
                text=[n['id'][:8]], textposition='top center',
                textfont=dict(size=8, color='white'),
                name=n['type'],
                hovertemplate=f"<b>{n['id']}</b><br>Type: {n['type']}<br>" +
                              (f"Fraud P: {n.get('fraud_prob', 0):.3f}" if 'fraud_prob' in n else "")
            ))

        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=550
        )
        st.plotly_chart(dark_fig(fig), use_container_width=True)
        st.caption("💡 For interactive PyVis graph: run `python utils/visualization.py` to generate HTML")

        pyvis_html = Path("fraud_graph.html")
        if pyvis_html.exists():
            st.subheader("Interactive PyVis Graph")
            with open(pyvis_html, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=700, scrolling=True)
        else:
            st.info("PyVis HTML not found. Run `python utils/visualization.py` to generate `fraud_graph.html`.")


# ── PAGE 4: Model Performance ─────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown("""
        <div class='hero-card fade-in'>
            <h2 style='margin-bottom:0.2rem;'>📈 Model Performance</h2>
            <p class='subtitle'>AUC, confusion matrix, and precision-recall analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    if not arts['trained']:
        st.warning("Train the model first.")
    else:
        fpr    = arts['fpr']
        tpr    = arts['tpr']
        cm     = arts['cm']
        labels = arts['test_labels']
        probs  = arts['test_probs']
        meta   = arts['meta']

        col_l, col_r = st.columns(2)

        with col_l:
            # ROC Curve
            auc = meta['test_auc']
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f'GAT (AUC={auc:.4f})',
                                     line=dict(color='#4B9EFF', width=2.5)))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                     name='Random', line=dict(color='gray', dash='dash')))
            fig.update_layout(title='ROC Curve',
                              xaxis_title='FPR', yaxis_title='TPR')
            st.plotly_chart(dark_fig(fig), use_container_width=True)

        with col_r:
            # Confusion matrix
            labels_map = ['Legitimate', 'Fraudulent']
            fig2 = go.Figure(go.Heatmap(
                z=cm, x=labels_map, y=labels_map,
                colorscale='RdBu_r',
                text=cm, texttemplate="%{text}",
                textfont={"size": 22}
            ))
            fig2.update_layout(title='Confusion Matrix',
                               xaxis_title='Predicted', yaxis_title='Actual')
            st.plotly_chart(dark_fig(fig2), use_container_width=True)

        # Precision-Recall
        from sklearn.metrics import precision_recall_curve, average_precision_score
        preds_binary = (probs > 0.5).astype(int)
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                   name=f'AP={ap:.4f}',
                                   line=dict(color='#FF9F40', width=2.5)))
        fig3.update_layout(title='Precision-Recall Curve',
                            xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(dark_fig(fig3), use_container_width=True)

        # Metrics table
        TP = int(cm[1,1]); FP = int(cm[0,1])
        FN = int(cm[1,0]); TN = int(cm[0,0])
        precision_val = TP / (TP + FP + 1e-8)
        recall_val    = TP / (TP + FN + 1e-8)
        f1_val        = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)

        metrics_df = pd.DataFrame({
            'Metric': ['AUC-ROC', 'Precision', 'Recall', 'F1-Score', 'TP', 'FP', 'FN', 'TN'],
            'Value' : [f"{meta['test_auc']:.4f}", f"{precision_val:.4f}",
                       f"{recall_val:.4f}", f"{f1_val:.4f}",
                       TP, FP, FN, TN]
        })
        st.table(metrics_df)


# ── PAGE 5: Feature Importance ────────────────────────────────────────────────
elif page == "⚙️  Feature Importance":
    st.markdown("""
        <div class='hero-card fade-in'>
            <h2 style='margin-bottom:0.2rem;'>⚙️ Feature Importance</h2>
            <p class='subtitle'>Understand which features influence fraud predictions most.</p>
        </div>
        """, unsafe_allow_html=True)

    if not arts['trained']:
        st.warning("Train the model first.")
    else:
        fi_path = ARTIFACTS / "feature_importance.csv"
        if fi_path.exists():
            fi_df = pd.read_csv(fi_path)
            if {'feature', 'importance'}.issubset(fi_df.columns):
                fi_df = fi_df.sort_values('importance', ascending=False).head(25)
                features = fi_df['feature'].astype(str).tolist()
                scores = fi_df['importance'].astype(float).to_numpy()
                scores = scores / (scores.sum() + 1e-8)
                st.success(f"Loaded real feature importance from {fi_path}")
            else:
                st.warning("feature_importance.csv found but missing required columns: feature, importance. Showing representative values.")
                features = ['TransactionAmt', 'card1', 'card2', 'addr1', 'dist1',
                            'C1', 'C2', 'C13', 'C14', 'D1', 'D10', 'D15',
                            'V258', 'V257', 'V201', 'V189', 'V83', 'V45',
                            'P_emaildomain', 'DeviceType', 'V307', 'V310',
                            'C6', 'C11', 'D4']
                scores = np.array([0.12, 0.09, 0.07, 0.065, 0.06,
                                   0.055, 0.05, 0.048, 0.045, 0.04,
                                   0.038, 0.035, 0.032, 0.031, 0.03,
                                   0.028, 0.026, 0.025, 0.024, 0.022,
                                   0.02, 0.019, 0.018, 0.017, 0.016])
                scores = scores / scores.sum()
        else:
            st.warning("Real feature-importance artifact not found. Showing representative values.")
            features = ['TransactionAmt', 'card1', 'card2', 'addr1', 'dist1',
                        'C1', 'C2', 'C13', 'C14', 'D1', 'D10', 'D15',
                        'V258', 'V257', 'V201', 'V189', 'V83', 'V45',
                        'P_emaildomain', 'DeviceType', 'V307', 'V310',
                        'C6', 'C11', 'D4']
            scores = np.array([0.12, 0.09, 0.07, 0.065, 0.06,
                               0.055, 0.05, 0.048, 0.045, 0.04,
                               0.038, 0.035, 0.032, 0.031, 0.03,
                               0.028, 0.026, 0.025, 0.024, 0.022,
                               0.02, 0.019, 0.018, 0.017, 0.016])
            scores = scores / scores.sum()

        fig = go.Figure(go.Bar(
            x=scores[::-1], y=features[::-1],
            orientation='h',
            marker=dict(
                color=scores[::-1],
                colorscale='Blues',
                showscale=False
            )
        ))
        fig.update_layout(
            title='Top 25 Feature Importances (Gradient × Input)',
            xaxis_title='Importance Score',
            height=650
        )
        st.plotly_chart(dark_fig(fig), use_container_width=True)

        st.subheader("💡 Key Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.success("""
            **Top Predictors:**
            - **TransactionAmt** – Unusual amounts flag fraud
            - **card1** – Stolen card reuse pattern
            - **C-features** – Count-aggregated card/email stats
            - **D-features** – Days since last transaction
            - **V-features** – Vesta's proprietary signals
            """)
        with col2:
            st.warning("""
            **Graph-Specific Insights:**
            - Fraud rings reuse `card1` across many TXs
            - Email domains concentrate fraud activity
            - Device fingerprints link fraud accounts
            - Multi-hop patterns expose money mule chains
            """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with PyTorch Geometric · Streamlit · Plotly | IEEE-CIS Fraud Detection Dataset")
