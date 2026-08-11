
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(
    page_title="Milan Urban Agriculture Suitability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌱 Urban Agriculture Suitability Dashboard")
st.markdown("### Milan, Italy (Zones 4 & 9)")
st.markdown("**Model:** CatBoost | **Accuracy:** 99.96% | **Dataset:** 364,540 grid points")

# Load model and predictions
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

@st.cache_data
def load_predictions():
    df1 = pd.read_csv("part1.zip")
    df2 = pd.read_csv("part2.zip")
    return pd.concat([df1, df2], ignore_index=True)

try:
    model = load_model()
    predictions_df = load_predictions()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

if data_loaded:
    # Sidebar
    st.sidebar.header("Model Performance Metrics")
    
    accuracy = (predictions_df['Actual_Class'] == predictions_df['Predicted_Class']).mean()
    st.sidebar.metric("Overall Accuracy", f"{accuracy:.2%}")
    
    avg_confidence = predictions_df['Confidence'].mean()
    st.sidebar.metric("Average Confidence", f"{avg_confidence:.4f}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Class Distribution")
    
    class_counts = predictions_df['Predicted_Label'].value_counts()
    for label, count in class_counts.items():
        percentage = (count / len(predictions_df)) * 100
        st.sidebar.text(f"{label}: {count:,} ({percentage:.1f}%)")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Sample Selection")
    
    # Sample selector
    sample_idx = st.sidebar.selectbox(
        "Select Sample Row Index",
        options=predictions_df.index.tolist(),
        index=0
    )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🗺️ Spatial Map", "📊 Sample Analysis", "📈 Model Insights"])
    
    with tab1:
        st.header("Geospatial Distribution of Suitability Classes")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_class = st.multiselect(
                "Show Classes",
                options=['N (Not Suitable)', 'S2 (Moderately Suitable)', 'S1 (Highly Suitable)'],
                default=['N (Not Suitable)', 'S2 (Moderately Suitable)', 'S1 (Highly Suitable)']
            )
        with col2:
            sample_size = st.slider("Sample Size for Display", 1000, 50000, 10000, step=1000)
        
        # Filter data
        map_data = predictions_df[predictions_df['Predicted_Label'].isin(show_class)]
        if len(map_data) > sample_size:
            map_data = map_data.sample(n=sample_size, random_state=42)
        
        # Color map
        color_map = {
            'S1 (Highly Suitable)': '#2E7D32',
            'S2 (Moderately Suitable)': '#FBC02D',
            'N (Not Suitable)': '#C62828'
        }
        
        # Create map
        if 'latitude' in map_data.columns and 'longitude' in map_data.columns:
            fig = px.scatter_mapbox(
                map_data,
                lat='latitude',
                lon='longitude',
                color='Predicted_Label',
                color_discrete_map=color_map,
                zoom=11,
                height=600,
                mapbox_style="carto-positron",
                hover_data=['Confidence', 'Actual_Label', 'NDVI', 'soil_moisture'],
                title=f"Predicted Suitability Map ({len(map_data):,} points)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Geospatial coordinates not available in dataset.")
    
    with tab2:
        st.header(f"Detailed Analysis for Sample #{sample_idx}")
        
        row_data = predictions_df.loc[sample_idx]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Actual Class", row_data['Actual_Label'])
        with col2:
            st.metric("Predicted Class", row_data['Predicted_Label'])
        with col3:
            st.metric("Confidence", f"{row_data['Confidence']:.4f}")
        
        # Feature values
        st.subheader("Feature Values")
        feature_cols = ['LST', 'NDVI', 'NDBI', 'soil_moisture', 'Slope', 
                       'UHI_Mitigation_Index', 'Agri_Potential_Index']
        
        feature_df = pd.DataFrame({
            'Feature': feature_cols,
            'Value': [row_data[col] for col in feature_cols]
        })
        st.dataframe(feature_df, use_container_width=True)
        
        # Probability bars
        st.subheader("Class Probabilities")
        prob_data = {
            'N (Not Suitable)': row_data['Prob_Class_0_N'],
            'S2 (Moderately Suitable)': row_data['Prob_Class_1_S2'],
            'S1 (Highly Suitable)': row_data['Prob_Class_2_S1']
        }
        
        fig_prob = go.Figure()
        for class_name, prob in prob_data.items():
            color = '#2E7D32' if 'S1' in class_name else '#FBC02D' if 'S2' in class_name else '#C62828'
            fig_prob.add_trace(go.Bar(
                x=[class_name],
                y=[prob],
                marker_color=color,
                text=[f"{prob:.4f}"],
                textposition='auto'
            ))
        fig_prob.update_layout(
            yaxis_title="Probability",
            yaxis_range=[0, 1],
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
        # SHAP and LIME images (if available)
        st.subheader("Explainability")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**SHAP Waterfall Plot**")
            shap_path = f"figures/shap_waterfall_row_{sample_idx}_class_{int(row_data['Predicted_Class'])}.png"
            if os.path.exists(shap_path):
                st.image(shap_path)
            else:
                st.info("SHAP plot not available for this sample")
        
        with col2:
            st.markdown("**LIME Explanation**")
            lime_path = f"figures/lime_explanation_row_{sample_idx}_class_{int(row_data['Predicted_Class'])}.png"
            if os.path.exists(lime_path):
                st.image(lime_path)
            else:
                st.info("LIME plot not available for this sample")
    
    with tab3:
        st.header("Model Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Importance")
            feature_imp = pd.DataFrame({
                'Feature': ['NDVI', 'soil_moisture', 'Agri_Potential_Index', 
                           'LST', 'UHI_Mitigation_Index', 'Slope', 'NDBI'],
                'Importance': [59.13, 28.92, 4.48, 2.55, 2.24, 2.10, 0.58]
            })
            fig_imp = px.bar(
                feature_imp.sort_values('Importance', ascending=True),
                x='Importance',
                y='Feature',
                orientation='h',
                title="CatBoost Feature Importance (%)"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            st.subheader("Confusion Matrix")
            cm_data = np.array([[67295, 0, 0], [44, 246455, 108], [0, 6, 50632]])
            fig_cm = px.imshow(
                cm_data,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['N', 'S2', 'S1'],
                y=['N', 'S2', 'S1'],
                text_auto=True,
                title="Confusion Matrix",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    
    # Download button
    st.markdown("---")
    st.header("Download Data")
    csv = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions CSV (100 MB)",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
    
    st.markdown("""
    **Note:** This dashboard visualizes the results of the Urban Agriculture Suitability 
    analysis for Milan Zones 4 & 9. The CatBoost model achieved 99.96% accuracy using 
    a Multi-Criteria Decision Making (MCDM) approach balancing F1-Score, ROC-AUC, 
    inference speed, and model size.
    """)

else:
    st.error("Failed to load model or predictions. Please check if files exist.")
