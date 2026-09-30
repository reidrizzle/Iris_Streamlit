"""
Iris Flower Prediction & Feature Analysis Application
------------------------------------------------------
Features:
1. Real-time Iris species prediction using trained Random Forest Classifier.
2. Interactive Probability Distribution visualization.
3. Feature Comparison: Your Input vs Dataset Average (and predicted species average).
4. 2D Feature Space scatter plot showing your input's position among dataset samples.
5. In-app Model Re-training & Performance Metrics.
"""

import json
import os
import numpy as np
import pandas as pd
import streamlit as st

# Safe import for plotly with native Streamlit chart fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Import the training and loading utilities from our training script
from train_model import load_or_train_model, train_and_export_model

# Page Configuration
st.set_page_config(
    page_title="Iris Classifier & Feature Analysis",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(6, 182, 212, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.25);
        margin-bottom: 1.5rem;
    }
    .species-badge {
        display: inline-block;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        padding: 0.2rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .setosa-badge { color: #6366F1; background-color: rgba(99, 102, 241, 0.15); }
    .versicolor-badge { color: #0D9488; background-color: rgba(13, 148, 136, 0.15); }
    .virginica-badge { color: #D97706; background-color: rgba(217, 119, 6, 0.15); }
    .metric-container {
        border-radius: 10px;
        background-color: rgba(241, 245, 249, 0.6);
        padding: 0.8rem 1rem;
        border: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Botanical Info Dictionary
SPECIES_INFO = {
    "setosa": {
        "scientific": "Iris setosa",
        "badge_class": "setosa-badge",
        "color": "#6366F1",
        "description": "Noticeable for having distinctly shorter and narrower petals compared to its sepals. Highly clustered and readily distinguishable.",
        "icon": "🟣",
    },
    "versicolor": {
        "scientific": "Iris versicolor",
        "badge_class": "versicolor-badge",
        "color": "#0D9488",
        "description": "Known as the Blue Flag iris. Moderately sized petals and sepals, occupying the intermediate region of the feature space.",
        "icon": "🟢",
    },
    "virginica": {
        "scientific": "Iris virginica",
        "badge_class": "virginica-badge",
        "color": "#D97706",
        "description": "Known as the Virginia iris. Features the largest petals and sepals with deep veining patterns.",
        "icon": "🟠",
    },
}

FEATURE_FRIENDLY_NAMES = {
    "sepal length (cm)": "Sepal Length",
    "sepal width (cm)": "Sepal Width",
    "petal length (cm)": "Petal Length",
    "petal width (cm)": "Petal Width",
}


@st.cache_resource(show_spinner="Loading trained model...")
def get_cached_model():
    """Load or train model once and cache in memory."""
    return load_or_train_model(force_retrain=False)


# Initialize Model & Data
try:
    model, metadata, df = get_cached_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Attempting automatic re-training...")
    model, metadata, df = train_and_export_model()

feature_names = metadata["feature_names"]
target_names = metadata["target_names"]
dataset_averages = metadata["dataset_averages"]
species_averages = metadata["species_averages"]
feature_ranges = metadata["feature_ranges"]

# ==========================================
# SIDEBAR: Inputs & Controls
# ==========================================
st.sidebar.header("🌸 Input Flower Features")

if not HAS_PLOTLY:
    st.sidebar.info("💡 **Plotly not detected**: Using native Streamlit charts. To enable rich Plotly charts, run:\n`pip install plotly`")

# Preset selector for fast testing
preset_choice = st.sidebar.selectbox(
    "Choose a preset sample or customize:",
    [
        "Custom Input",
        "Typical Iris Setosa (Preset)",
        "Typical Iris Versicolor (Preset)",
        "Typical Iris Virginica (Preset)",
        "Dataset Average (Preset)",
    ],
)

# Preset feature values
default_values = {
    "Custom Input": (5.4, 3.4, 1.5, 0.4),
    "Typical Iris Setosa (Preset)": (5.0, 3.4, 1.5, 0.2),
    "Typical Iris Versicolor (Preset)": (5.9, 2.8, 4.3, 1.3),
    "Typical Iris Virginica (Preset)": (6.6, 3.0, 5.5, 2.0),
    "Dataset Average (Preset)": (
        round(dataset_averages["sepal length (cm)"], 2),
        round(dataset_averages["sepal width (cm)"], 2),
        round(dataset_averages["petal length (cm)"], 2),
        round(dataset_averages["petal width (cm)"], 2),
    ),
}

selected_defaults = default_values[preset_choice]

# Interactive sliders for the 4 features
sepal_length = st.sidebar.slider(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=float(selected_defaults[0]),
    step=0.1,
    help=f"Dataset Range: {feature_ranges['sepal length (cm)']['min']} - {feature_ranges['sepal length (cm)']['max']} cm (Avg: {dataset_averages['sepal length (cm)']:.2f})",
)

sepal_width = st.sidebar.slider(
    "Sepal Width (cm)",
    min_value=2.0,
    max_value=4.5,
    value=float(selected_defaults[1]),
    step=0.1,
    help=f"Dataset Range: {feature_ranges['sepal width (cm)']['min']} - {feature_ranges['sepal width (cm)']['max']} cm (Avg: {dataset_averages['sepal width (cm)']:.2f})",
)

petal_length = st.sidebar.slider(
    "Petal Length (cm)",
    min_value=1.0,
    max_value=7.0,
    value=float(selected_defaults[2]),
    step=0.1,
    help=f"Dataset Range: {feature_ranges['petal length (cm)']['min']} - {feature_ranges['petal length (cm)']['max']} cm (Avg: {dataset_averages['petal length (cm)']:.2f})",
)

petal_width = st.sidebar.slider(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=2.6,
    value=float(selected_defaults[3]),
    step=0.1,
    help=f"Dataset Range: {feature_ranges['petal width (cm)']['min']} - {feature_ranges['petal width (cm)']['max']} cm (Avg: {dataset_averages['petal width (cm)']:.2f})",
)

user_input = {
    "sepal length (cm)": sepal_length,
    "sepal width (cm)": sepal_width,
    "petal length (cm)": petal_length,
    "petal width (cm)": petal_width,
}

# Retrain Model Button in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Management")
st.sidebar.caption(f"Model: Random Forest ({metadata.get('accuracy', 0)*100:.1f}% Accuracy)")
if st.sidebar.button("🔄 Retrain Model Now"):
    with st.spinner("Retraining Random Forest Classifier..."):
        train_and_export_model()
        st.cache_resource.clear()
        st.sidebar.success("Model retrained successfully!")
        st.rerun()

# ==========================================
# INFERENCE & PREDICTIONS
# ==========================================
input_array = np.array(
    [[sepal_length, sepal_width, petal_length, petal_width]]
)

prediction_idx = int(model.predict(input_array)[0])
predicted_species = target_names[prediction_idx]
predicted_info = SPECIES_INFO.get(predicted_species, {})

# Class Probabilities
probabilities = model.predict_proba(input_array)[0]
confidence = probabilities[prediction_idx] * 100

# ==========================================
# MAIN PAGE HEADER
# ==========================================
st.markdown('<div class="main-title">🌸 Iris Model Training & Prediction Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Interactive machine learning dashboard: real-time classification, probability distribution, and input vs. dataset benchmark.</div>',
    unsafe_allow_html=True,
)

# Tabs for Organization
tab_pred, tab_comparison, tab_dataset, tab_model = st.tabs(
    [
        "🎯 Prediction & Probabilities",
        "📊 Your Input vs Dataset Average",
        "🌌 Feature Space & Distribution",
        "🛠️ Model Details & Metrics",
    ]
)

# ==========================================
# TAB 1: PREDICTION & PROBABILITY DISTRIBUTION
# ==========================================
with tab_pred:
    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    with col_left:
        # Prediction Result Card
        st.subheader("Prediction Result")
        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="color: #64748B; font-size: 0.95rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.4rem;">
                    Predicted Classification
                </div>
                <div class="species-badge {predicted_info.get('badge_class', '')}">
                    {predicted_info.get('icon', '🌸')} {predicted_info.get('scientific', predicted_species.capitalize())}
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #1E293B; margin-top: 0.6rem;">
                    Model Confidence: <span style="color: {predicted_info.get('color', '#4F46E5')};">{confidence:.1f}%</span>
                </div>
                <p style="color: #475569; font-size: 0.95rem; margin-top: 0.8rem; line-height: 1.5;">
                    {predicted_info.get('description', '')}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick summary breakdown
        st.markdown("##### Current Input Values")
        cols_feat = st.columns(2)
        for i, (feat, val) in enumerate(user_input.items()):
            c = cols_feat[i % 2]
            c.metric(
                label=FEATURE_FRIENDLY_NAMES[feat],
                value=f"{val:.1f} cm",
                delta=f"{val - dataset_averages[feat]:+.2f} cm vs avg",
            )

    with col_right:
        st.subheader("Probability Distribution")

        # Probability DataFrame
        prob_df = pd.DataFrame(
            {
                "Species": [SPECIES_INFO[sp]["scientific"] for sp in target_names],
                "Probability": [prob * 100 for prob in probabilities],
                "Color": [SPECIES_INFO[sp]["color"] for sp in target_names],
            }
        )

        if HAS_PLOTLY:
            fig_prob = go.Figure(
                data=[
                    go.Bar(
                        x=prob_df["Species"],
                        y=prob_df["Probability"],
                        text=[f"{p:.1f}%" for p in prob_df["Probability"]],
                        textposition="outside",
                        marker=dict(
                            color=prob_df["Color"],
                            line=dict(color="#1E293B", width=1),
                            opacity=0.9,
                        ),
                        width=0.45,
                    )
                ]
            )
            fig_prob.update_layout(
                title="Class Probability Distribution",
                yaxis=dict(title="Probability (%)", range=[0, 115]),
                xaxis=dict(title="Iris Species"),
                template="plotly_white",
                height=320,
                margin=dict(l=30, r=20, t=40, b=40),
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            # Native Streamlit Chart Fallback
            chart_data = prob_df.set_index("Species")["Probability"]
            st.bar_chart(chart_data)

        # Progress bars for each class
        for sp, prob in zip(target_names, probabilities):
            info = SPECIES_INFO[sp]
            st.write(f"**{info['scientific']}**: {prob * 100:.1f}%")
            st.progress(float(prob))

# ==========================================
# TAB 2: INPUT VS DATASET AVERAGE
# ==========================================
with tab_comparison:
    st.subheader("📊 Your Input vs. Dataset Average")
    st.caption("Compare your customized flower dimensions against both the overall dataset mean and species-specific baselines.")

    comparison_mode = st.radio(
        "Comparison Baseline:",
        ["Dataset Overall Average", "Include Predicted Species Average"],
        horizontal=True,
    )

    # Prepare DataFrame
    comp_records = []
    for feat in feature_names:
        friendly = FEATURE_FRIENDLY_NAMES[feat]
        user_val = user_input[feat]
        avg_val = dataset_averages[feat]
        species_avg = species_averages[predicted_species][feat]

        comp_records.append(
            {
                "Feature": friendly,
                "Value (cm)": user_val,
                "Category": "Your Input",
            }
        )
        comp_records.append(
            {
                "Feature": friendly,
                "Value (cm)": round(avg_val, 2),
                "Category": "Dataset Average",
            }
        )
        if comparison_mode == "Include Predicted Species Average":
            comp_records.append(
                {
                    "Feature": friendly,
                    "Value (cm)": round(species_avg, 2),
                    "Category": f"{predicted_species.capitalize()} Average",
                }
            )

    comp_df = pd.DataFrame(comp_records)

    if HAS_PLOTLY:
        # Plotly Grouped Bar Chart
        palette = {
            "Your Input": "#3B82F6",
            "Dataset Average": "#94A3B8",
            f"{predicted_species.capitalize()} Average": predicted_info.get("color", "#10B981"),
        }

        fig_comp = px.bar(
            comp_df,
            x="Feature",
            y="Value (cm)",
            color="Category",
            barmode="group",
            text="Value (cm)",
            color_discrete_map=palette,
            title="Feature Benchmark: Your Input vs Average Dimensions",
        )

        fig_comp.update_traces(textposition="outside", texttemplate="%{y:.2f} cm")
        fig_comp.update_layout(
            yaxis=dict(title="Length / Width (cm)", range=[0, max(comp_df["Value (cm)"]) * 1.25]),
            template="plotly_white",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig_comp, use_container_width=True)

        # Radar (Spider) Chart view
        with st.expander("🕸️ View as Radar / Spider Chart", expanded=False):
            fig_radar = go.Figure()
            categories = [FEATURE_FRIENDLY_NAMES[f] for f in feature_names]

            input_vals = [user_input[f] for f in feature_names]
            input_vals.append(input_vals[0])

            avg_vals = [dataset_averages[f] for f in feature_names]
            avg_vals.append(avg_vals[0])

            categories_closed = categories + [categories[0]]

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=input_vals,
                    theta=categories_closed,
                    fill="toself",
                    name="Your Input",
                    line=dict(color="#3B82F6", width=2),
                    fillcolor="rgba(59, 130, 246, 0.2)",
                )
            )

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=avg_vals,
                    theta=categories_closed,
                    fill="toself",
                    name="Dataset Average",
                    line=dict(color="#94A3B8", width=2, dash="dash"),
                    fillcolor="rgba(148, 163, 184, 0.15)",
                )
            )

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 8])),
                showlegend=True,
                template="plotly_white",
                height=420,
            )

            st.plotly_chart(fig_radar, use_container_width=True)
    else:
        # Native Streamlit Chart Fallback
        pivot_df = comp_df.pivot(index="Feature", columns="Category", values="Value (cm)")
        st.bar_chart(pivot_df)

    # Detailed Delta Table
    st.markdown("##### Dimension Deviation Analysis")
    delta_cols = st.columns(4)
    for idx, feat in enumerate(feature_names):
        col = delta_cols[idx]
        val = user_input[feat]
        avg = dataset_averages[feat]
        diff = val - avg
        pct = (diff / avg) * 100

        col.metric(
            label=FEATURE_FRIENDLY_NAMES[feat],
            value=f"{val:.2f} cm",
            delta=f"{diff:+.2f} cm ({pct:+.1f}%)",
        )

# ==========================================
# TAB 3: FEATURE SPACE & DATASET SCATTER
# ==========================================
with tab_dataset:
    st.subheader("🌌 Feature Space & Dataset Clusters")
    st.caption("Observe where your current input point sits relative to the historical training dataset.")

    scatter_col1, scatter_col2 = st.columns([1, 3], gap="medium")

    with scatter_col1:
        x_axis = st.selectbox(
            "X-Axis Feature:",
            feature_names,
            index=2,  # Petal length
            format_func=lambda x: FEATURE_FRIENDLY_NAMES[x],
        )
        y_axis = st.selectbox(
            "Y-Axis Feature:",
            feature_names,
            index=3,  # Petal width
            format_func=lambda x: FEATURE_FRIENDLY_NAMES[x],
        )

    with scatter_col2:
        if HAS_PLOTLY:
            color_map = {
                "setosa": "#6366F1",
                "versicolor": "#0D9488",
                "virginica": "#D97706",
            }

            fig_scatter = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color="species",
                color_discrete_map=color_map,
                hover_data=feature_names,
                opacity=0.75,
                title=f"{FEATURE_FRIENDLY_NAMES[x_axis]} vs. {FEATURE_FRIENDLY_NAMES[y_axis]}",
            )

            # Highlight user's input with a distinct star marker
            fig_scatter.add_trace(
                go.Scatter(
                    x=[user_input[x_axis]],
                    y=[user_input[y_axis]],
                    mode="markers+text",
                    marker=dict(
                        symbol="star",
                        size=20,
                        color="#EF4444",
                        line=dict(color="#FFFFFF", width=2),
                    ),
                    text=["⭐ YOUR INPUT"],
                    textposition="top center",
                    name="Your Input",
                )
            )

            fig_scatter.update_layout(
                template="plotly_white",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            # Native Streamlit scatter chart
            st.scatter_chart(df, x=x_axis, y=y_axis, color="species")
            st.caption(f"⭐ Your Input Coordinate: ({user_input[x_axis]:.2f} cm, {user_input[y_axis]:.2f} cm)")

    with st.expander("📄 View Training Dataset Table"):
        st.dataframe(df, use_container_width=True, height=250)

# ==========================================
# TAB 4: MODEL DETAILS & METRICS
# ==========================================
with tab_model:
    st.subheader("🛠️ Model Architecture & Evaluation")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Model Architecture", metadata.get("model_type", "Random Forest"))
    m_col2.metric("Test Accuracy", f"{metadata.get('accuracy', 0)*100:.2f}%")
    m_col3.metric("Number of Estimators", metadata.get("n_estimators", 100))

    st.markdown("##### Confusion Matrix")
    conf_mat = np.array(metadata.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]))

    if HAS_PLOTLY:
        fig_cm = px.imshow(
            conf_mat,
            text_auto=True,
            labels=dict(x="Predicted Species", y="Actual Species", color="Count"),
            x=target_names,
            y=target_names,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        cm_df = pd.DataFrame(
            conf_mat,
            index=[f"Actual {t.capitalize()}" for t in target_names],
            columns=[f"Predicted {t.capitalize()}" for t in target_names],
        )
        st.dataframe(cm_df, use_container_width=True)

    with st.expander("📋 View Detailed Classification Report"):
        st.json(metadata.get("classification_report", {}))
