# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None

def predict_disease(model, scaler, age, gender, fever, cough, headache, bp, sugar, cholesterol, history):
    input_data = np.array([[age, gender, fever, cough, headache, bp, sugar, cholesterol, history]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return "Heart Disease" if prediction == 1 else "Flu"

def risk_level(sugar, cholesterol):
    if sugar > 180 or cholesterol > 240:
        return "High Risk", "risk-high"
    elif sugar > 140:
        return "Medium Risk", "risk-medium"
    else:
        return "Low Risk", "risk-low"

# Main header
st.markdown("""
<div class="main-header">
    <h1>🩺 Disease Prediction System</h1>
    <p>Advanced Medical Diagnosis Assistant using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966325.png", width=100)
    st.title("Navigation")
    page = st.radio("", ["🏥 Prediction", "📊 Analytics", "ℹ️ About"])
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.info("""
    - **BP**: Normal 120/80
    - **Sugar**: Normal < 100 mg/dL
    - **Cholesterol**: Normal < 200 mg/dL
    """)

# Main content based on navigation
if page == "🏥 Prediction":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Personal Information")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"])
        gender_val = 1 if gender == "Male" else 0
        
        st.markdown("### 🩺 Medical History")
        history = st.number_input("Medical History Score", min_value=0, max_value=100, value=0, 
                                   help="0-100 scale for medical history severity")
    
    with col2:
        st.markdown("### 🤒 Symptoms")
        fever = st.radio("Fever", ["No", "Yes"], horizontal=True)
        fever_val = 1 if fever == "Yes" else 0
        
        cough = st.radio("Cough", ["No", "Yes"], horizontal=True)
        cough_val = 1 if cough == "Yes" else 0
        
        headache = st.radio("Headache", ["No", "Yes"], horizontal=True)
        headache_val = 1 if headache == "Yes" else 0
    
    with col3:
        st.markdown("### 📊 Vital Signs")
        bp = st.slider("Blood Pressure (Systolic)", 80, 200, 120, 5)
        sugar = st.slider("Blood Sugar (mg/dL)", 50, 400, 100, 5)
        cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 180, 10)
    
    # Prediction button
    st.markdown("---")
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    with col_button2:
        predict_btn = st.button("🔍 PREDICT DISEASE", use_container_width=True)
    
    if predict_btn:
        model, scaler = load_models()
        if model and scaler:
            disease = predict_disease(model, scaler, age, gender_val, fever_val, 
                                      cough_val, headache_val, bp, sugar, 
                                      cholesterol, history)
            risk, risk_class = risk_level(sugar, cholesterol)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📋 Diagnosis Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>🩺 Predicted Disease</h2>
                    <h1 style="font-size: 3rem;">{disease}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                st.markdown(f"""
                <div class="prediction-card {risk_class}">
                    <h2>⚠️ Risk Assessment</h2>
                    <h1 style="font-size: 3rem;">{risk}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed metrics
            st.markdown("### 📈 Detailed Analysis")
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            sugar_status = "🟢 Normal" if sugar < 100 else "🟡 Pre-diabetic" if sugar < 126 else "🔴 Diabetic"
            bp_status = "🟢 Normal" if bp < 120 else "🟡 Elevated" if bp < 130 else "🔴 Hypertension"
            chol_status = "🟢 Desirable" if cholesterol < 200 else "🟡 Borderline" if cholesterol < 240 else "🔴 High"
            
            with metric_col1:
                st.metric("Blood Sugar", f"{sugar} mg/dL", sugar_status)
            with metric_col2:
                st.metric("Blood Pressure", f"{bp} mmHg", bp_status)
            with metric_col3:
                st.metric("Cholesterol", f"{cholesterol} mg/dL", chol_status)
            
            # Recommendations
            st.markdown("### 💊 Recommendations")
            if disease == "Heart Disease":
                st.warning("""
                **Heart Disease Management:**
                - Consult a cardiologist immediately
                - Maintain a heart-healthy diet low in saturated fats
                - Regular exercise (30 mins daily, as advised by doctor)
                - Monitor blood pressure and cholesterol regularly
                - Take prescribed medications on time
                """)
            else:
                st.info("""
                **Flu Management:**
                - Get plenty of rest and stay hydrated
                - Take over-the-counter fever reducers if needed
                - Consult a doctor if symptoms persist beyond 5 days
                - Practice good hygiene to prevent spread
                - Consider flu vaccine for future prevention
                """)
            
            # Risk-based recommendations
            if risk == "High Risk":
                st.error("⚠️ **High Risk Alert:** Immediate medical consultation recommended!")

elif page == "📊 Analytics":
    st.markdown("## 📊 Disease Analytics Dashboard")
    
    # Sample data for demonstration (in real scenario, load from database)
    data = {
        'Age': [25, 45, 35, 50, 23, 38, 42, 55, 48, 32],
        'Disease': ['Flu', 'Heart Disease', 'Flu', 'Heart Disease', 'Flu', 
                    'Heart Disease', 'Heart Disease', 'Flu', 'Heart Disease', 'Flu'],
        'Sugar': [85, 160, 120, 180, 90, 150, 170, 110, 155, 95],
        'Cholesterol': [180, 220, 200, 240, 170, 230, 250, 190, 215, 185],
        'BP': [120, 140, 130, 150, 110, 135, 145, 125, 138, 118]
    }
    df = pd.DataFrame(data)
    
    # Disease distribution
    col1, col2 = st.columns(2)
    
    with col1:
        disease_counts = df['Disease'].value_counts()
        fig1 = go.Figure(data=[go.Pie(labels=disease_counts.index, values=disease_counts.values, 
                                      marker=dict(colors=['#667eea', '#764ba2']))])
        fig1.update_layout(title="Disease Distribution", height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Box(x=df['Disease'], y=df['Age'], name='Age',
                              marker_color=['#667eea', '#764ba2']))
        fig2.update_layout(title="Age Distribution by Disease", height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Risk factors heatmap
    st.markdown("### 🔥 Risk Factors Correlation")
    correlation = df[['Sugar', 'Cholesterol', 'BP']].corr()
    fig3 = px.imshow(correlation, text_auto=True, aspect="auto",
                     color_continuous_scale='RdBu', title="Vital Signs Correlation")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Scatter plot
    st.markdown("### 📈 Sugar vs Cholesterol Analysis")
    fig4 = px.scatter(df, x='Sugar', y='Cholesterol', color='Disease',
                      size='BP', hover_data=['Age'],
                      title="Sugar vs Cholesterol by Disease")
    st.plotly_chart(fig4, use_container_width=True)

else:
    st.markdown("## ℹ️ About This System")
    
    col_about1, col_about2 = st.columns([1, 2])
    
    with col_about1:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063719.png", width=200)
    
    with col_about2:
        st.markdown("""
        ### Disease Prediction System using Machine Learning
        
        This advanced medical diagnosis assistant uses a **Logistic Regression** model to predict 
        diseases based on patient symptoms and vital signs.
        
        #### Features:
        - ✅ Real-time disease prediction
        - ✅ Risk level assessment
        - ✅ Interactive data analytics
        - ✅ Personalized recommendations
        - ✅ Medical history tracking
        
        #### How It Works:
        1. Enter patient information and symptoms
        2. Input vital signs (BP, Sugar, Cholesterol)
        3. System processes data using ML model
        4. Get instant diagnosis and risk assessment
        
        #### Technology Stack:
        - **Frontend**: Streamlit
        - **ML Model**: Logistic Regression
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly
        """)
    
    st.markdown("---")
    st.markdown("### 📞 Emergency Contacts")
    
    emergency_col1, emergency_col2, emergency_col3 = st.columns(3)
    
    with emergency_col1:
        st.info("🚑 **Emergency**\n\nCall: 911")
    with emergency_col2:
        st.info("🏥 **Hospital**\n\nLocate nearest hospital")
    with emergency_col3:
        st.info("💊 **Poison Control**\n\nCall: 1-800-222-1222")
    
    st.markdown("---")
    st.markdown("*Disclaimer: This system is for informational purposes only. Always consult a healthcare professional for medical advice.*")

# Footer
st.markdown("---")
st.markdown("""
<center>
    <p style="color: gray;">© 2024 Disease Prediction System | Powered by Machine Learning</p>
</center>
""", unsafe_allow_html=True)