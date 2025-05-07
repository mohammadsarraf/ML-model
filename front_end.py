import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(
    page_title="NBA MVP Prediction Model",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .model-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        border-left: 5px solid #1E3A8A;
    }
    .model-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    .model-description {
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #EFF6FF;
        border-radius: 5px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Set page title
st.markdown("<h1 class='main-header'>NBA MVP Prediction Model 🏀</h1>", unsafe_allow_html=True)

# Load prediction results
@st.cache_data
def load_data():
    output_rank = pd.read_csv("outputrank.csv")
    return {
        "outputrank.csv": output_rank
    }

prediction_data = load_data()

# Performance metrics for models
model_metrics = {
    "SVM": {"RMSE": 0.148, "R²": 0.592, "Pros": ["Good for high-dimensional data", "Effective when classes are separable", "Works well with clear margins of separation"], "Cons": ["Sensitive to feature scaling", "Less efficient on large datasets", "May overfit with wrong parameters"]},
    "Elastic Net": {"RMSE": 0.161, "R²": 0.505, "Pros": ["Combines L1 and L2 regularization", "Handles correlated features well", "Performs automatic feature selection"], "Cons": ["Requires tuning of multiple hyperparameters", "May struggle with highly nonlinear relationships", "Less interpretable than simple linear models"]},
    "Random Forest": {"RMSE": 0.142, "R²": 0.608, "Pros": ["Handles non-linear relationships", "Resistant to overfitting", "Provides feature importance metrics"], "Cons": ["Less interpretable than single decision trees", "Computationally intensive for large datasets", "May overfit noisy datasets"]},
    "AdaBoost": {"RMSE": 0.142, "R²": 0.601, "Pros": ["Automatically identifies weak spots in previous models", "Less prone to overfitting than other boosting methods", "Simple to implement"], "Cons": ["Sensitive to noisy data and outliers", "Sequential processing limits parallelization", "Can create complex models"]},
    "Gradient Boosting": {"RMSE": 0.141, "R²": 0.605, "Pros": ["Generally outperforms random forests", "Flexible for different loss functions", "Handles complex non-linear relationships"], "Cons": ["More hyperparameters to tune", "Computationally intensive", "Prone to overfitting without proper tuning"]},
    "LGBM": {"RMSE": 0.136, "R²": 0.642, "Pros": ["Faster training speed and higher efficiency", "Lower memory usage", "Better accuracy than many other boosting methods"], "Cons": ["More complex to tune properly", "Relatively newer algorithm with less documentation", "May require more careful parameter tuning"]}
}

# Display tabs for different views
tab1, tab2, tab3 = st.tabs(["MVP Predictions", "Model Details", "About"])

with tab1:
    st.markdown("<h2 class='section-header'>MVP Prediction Results</h2>", unsafe_allow_html=True)
    
    # Select data source
    data_source = st.selectbox(
        "Select Prediction Data Source", 
        ["Model 1"],
        index=0
    )
    
    # Extract the file name from the selection
    file_name = "outputrank.csv"
    
    # Get the selected data
    selected_data = prediction_data[file_name]
    
    # Select season
    seasons = sorted(selected_data["Season"].unique(), reverse=True)
    selected_season = st.selectbox("Select Season", seasons)
    
    # Filter data for selected season
    season_data = selected_data[selected_data["Season"] == selected_season]
    
    # Display top 3 MVP candidates for selected season
    st.subheader(f"Top MVP Candidates for {selected_season} Season")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual Results")
        actual_data = season_data[["MVP Rank Real", "MVP Share Real"]].head(3)
        st.dataframe(actual_data, hide_index=True)
    
    with col2:
        # Let user select model
        models = ["SVM", "Elastic Net", "Random Forest", "AdaBoost", "Gradient Boosting", "LGBM"]
        selected_model = st.selectbox("Select Model", models)
        
        # Display model predictions
        st.subheader(f"{selected_model} Model Predictions")
        model_cols = [f"MVP Rank {selected_model}", f"MVP Share {selected_model}"]
        model_data = season_data[model_cols].head(3)
        st.dataframe(model_data, hide_index=True)
    
    # Show full comparison table
    with st.expander("View Full Comparison"):
        st.dataframe(season_data, hide_index=True)
    
    # File source explanation
    st.markdown("---")
    st.subheader("Data Source Information")
    if file_name == "mvppred2.csv":
        st.info("**mvppred2.csv**: Direct output from the prediction model notebook, saved with `rank_final.to_csv('mvppred2.csv', index=False)`")
    elif file_name == "mvppred.csv":
        st.info("**mvppred.csv**: Alternative version of the prediction results")
    else:
        st.info("**outputrank.csv**: Alternative version of the prediction results")

with tab2:
    st.markdown("<h2 class='section-header'>Model Details</h2>", unsafe_allow_html=True)
    
    # Create a plot of model performance metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(model_metrics.keys())
    rmse_values = [metric["RMSE"] for metric in model_metrics.values()]
    r2_values = [metric["R²"] for metric in model_metrics.values()]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, rmse_values, width, label='RMSE (lower is better)', color='#4472C4')
    rects2 = ax.bar(x + width/2, r2_values, width, label='R² (higher is better)', color='#ED7D31')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    fig.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    
    # Information about the metrics
    with st.expander("About These Metrics"):
        st.markdown("""
        - **RMSE (Root Mean Square Error)**: Measures the average magnitude of errors in predictions. Lower values indicate better performance.
        - **R² (Coefficient of Determination)**: Represents the proportion of variance in the dependent variable that is predictable from the independent variables. Higher values (closer to 1) indicate better performance.
        
        LGBM performs the best overall with the lowest RMSE (0.136) and highest R² (0.642).
        """)
    
    # Display model details in native Streamlit components
    st.markdown("<h3 class='section-header'>Model Descriptions</h3>", unsafe_allow_html=True)
    
    # Create expanders for each model
    for model, metrics in model_metrics.items():
        with st.expander(f"{model} Model", expanded=True if model == "LGBM" else False):
            st.markdown(f"<div class='model-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3 class='model-header'>{model}</h3>", unsafe_allow_html=True)
            
            if model == "SVM":
                st.markdown("<p class='model-description'>Support Vector Machine (SVM) is a supervised learning algorithm that finds the optimal hyperplane to separate data points of different classes. For regression tasks like MVP prediction, it attempts to find a hyperplane that best fits the data within a certain margin of error.</p>", unsafe_allow_html=True)
            elif model == "Elastic Net":
                st.markdown("<p class='model-description'>Elastic Net is a linear regression model with both L1 and L2 regularization penalties. It's particularly useful when dealing with highly correlated predictors in the NBA stats, handling the multicollinearity problem effectively while performing feature selection.</p>", unsafe_allow_html=True)
            elif model == "Random Forest":
                st.markdown("<p class='model-description'>Random Forest is an ensemble learning method that builds multiple decision trees during training and outputs the average prediction of the individual trees. This model handles the non-linear relationships in NBA statistics well and provides insight into feature importance.</p>", unsafe_allow_html=True)
            elif model == "AdaBoost":
                st.markdown("<p class='model-description'>Adaptive Boosting (AdaBoost) is a boosting technique that combines multiple weak learners into a strong learner. It works by training models sequentially, with each new model focusing on correctly predicting the instances that previous models misclassified.</p>", unsafe_allow_html=True)
            elif model == "Gradient Boosting":
                st.markdown("<p class='model-description'>Gradient Boosting builds an ensemble of weak prediction models, typically decision trees, in a stage-wise fashion. Each new model is trained to minimize the loss function gradient, making it particularly powerful for prediction tasks with complex relationships.</p>", unsafe_allow_html=True)
            elif model == "LGBM":
                st.markdown("<p class='model-description'>Light Gradient Boosting Machine (LGBM) is a gradient boosting framework that uses tree-based learning algorithms. It's designed for distributed and efficient training with faster speed and lower memory usage, making it ideal for large NBA datasets with many features.</p>", unsafe_allow_html=True)
            
            # Performance metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE (Lower is better)", f"{metrics['RMSE']:.3f}")
            with col2:
                st.metric("R² (Higher is better)", f"{metrics['R²']:.3f}")
            
            # Pros and cons
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Pros")
                for pro in metrics["Pros"]:
                    st.markdown(f"✅ {pro}")
            
            with col2:
                st.markdown("##### Cons")
                for con in metrics["Cons"]:
                    st.markdown(f"⚠️ {con}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Usage in NBA context
    st.markdown("<h3 class='section-header'>Application to NBA MVP Prediction</h3>", unsafe_allow_html=True)
    st.markdown("""
    These models are applied to NBA MVP prediction by:
    
    1. **Training on historical data**: Each model learns patterns from past MVP voting results and player statistics
    2. **Feature importance**: Models identify which statistics most heavily influence MVP voting
    3. **Prediction**: Models generate MVP share predictions for current players based on their stats
    
    The best performing model is **LGBM** with the highest R² value (0.642) and lowest error rate (RMSE: 0.136),
    suggesting it best captures the complex relationships between player statistics and MVP voting.
    """)

with tab3:
    st.markdown("<h2 class='section-header'>About the Project</h2>", unsafe_allow_html=True)
    st.write("""
    This application displays the results of NBA MVP prediction models. The models were trained on historical NBA data
    to predict Most Valuable Player award winners and voting shares.
    
    The predictions are based on various statistics including player performance metrics, team success,
    and historical voting patterns.
    
    The model uses saved results from a Jupyter notebook, allowing users to view predictions without
    needing to rerun the computations.
    """)
    
    st.subheader("How to Use")
    st.write("""
    1. Select a prediction data source (mvppred2.csv is the direct output from the notebook)
    2. Select a season from the dropdown menu
    3. Choose a prediction model to compare with actual results
    4. Explore the detailed predictions in the expanded view
    """)
    
    st.subheader("Available Data Files")
    st.markdown("""
    - **mvppred2.csv**: Direct output from the prediction model notebook
    - **mvppred.csv**: Alternative version or run of the prediction model
    - **outputrank.csv**: Another alternative version of the prediction results
    """)