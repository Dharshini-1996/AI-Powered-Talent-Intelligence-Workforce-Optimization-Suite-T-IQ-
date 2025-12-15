import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION & PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="T-IQ | Talent Intelligence Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# DATA GENERATION (Simulating Database)
# ==========================================
@st.cache_data

def load_data():
    df = pd.read_csv("E:/Final_Project_AI-Powered Talent Intelligence & Workforce/Dataset/IBM HR Analytics Attrition_Revised.csv")
    
    # 1. Map Satisfaction strings to Numbers (Your existing code)
    satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
    df['JobSatisfaction'] = df['JobSatisfaction'].map(satisfaction_map).fillna(2)
    
    # 2. --- NEW FIX ---
    # Create a clean Numeric column for ML (1 for Yes, 0 for No)
    df['Attrition_Numeric'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Create a clean Label column for Visuals (Ensures it's not NaN)
    df['Attrition_Label'] = df['Attrition'] 
    
    return df

@st.cache_resource
def train_model(df):
    """Trains a quick RF model for the live prediction demo."""
    # Simple preprocessing
    df_encoded = df.copy()
    df_encoded['Department_Code'] = df_encoded['Department'].astype('category').cat.codes
    
    features = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'Department_Code']
    X = df_encoded[features]
    y = df_encoded['Attrition']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

df = load_data()
model = train_model(df)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
st.sidebar.title("T-IQ Suite")
st.sidebar.subheader("AI-Powered Workforce Optimization")

page = st.sidebar.radio("Navigate", [
    "Dashboard Overview", 
    "Workforce EDA", 
    "Attrition Predictor (ML)", 
    "Resume Matcher (NLP)",
    "HR Assistant (Chatbot)"
])

st.sidebar.markdown("---")
st.sidebar.info("Project: T-IQ Suite\nDeliverable: Interactive Dashboard")

# ==========================================
# PAGE 1: DASHBOARD OVERVIEW
# ==========================================
if page == "Dashboard Overview":
    st.title("📊 Executive Command Center")
    st.markdown("Real-time workforce insights and AI-driven metrics.")
    
    # Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_emp = len(df)
    ##attrition_rate = (df['Attrition'].sum() / total_emp) * 100
    attrition_rate = (df[df['Attrition'] == 'Yes'].shape[0] / total_emp) * 100
    avg_satisfaction = df['JobSatisfaction'].mean()
    avg_income = df['MonthlyIncome'].mean()
    
    with col1:
        st.metric("Total Employees", f"{total_emp}", "+12 hired this month")
    with col2:
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%", "-2% vs target", delta_color="inverse")
    with col3:
        st.metric("Avg Job Satisfaction", f"{avg_satisfaction:.1f}/4.0", "Stable")
    with col4:
        st.metric("Avg Monthly Income", f"${avg_income:,.0f}", "+5% YoY")

    st.markdown("---")
    
    # Row 2: Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Employee Distribution by Department")
        fig_dept = px.pie(df, names='Department', title='Headcount Share', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_dept, use_container_width=True)
        
    with c2:
        st.subheader("Attrition by Department")
        # FIX: Filter for the string 'Yes' instead of the integer 1
        attrition_by_dept = df[df['Attrition'] == 'Yes'].groupby('Department').size().reset_index(name='Count')
        
        # print(attrition_by_dept) # This will now show data
        fig_att = px.bar(attrition_by_dept, x='Department', y='Count', color='Department', title='High Risk Areas')
        st.plotly_chart(fig_att, use_container_width=True)

# ==========================================
# PAGE 2: WORKFORCE EDA
# ==========================================
elif page == "Workforce EDA":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Deep dive into demographics, salary, and satisfaction metrics.")
    
    tab1, tab2 = st.tabs(["Income Analysis", "Satisfaction & Tenure"])
    
    with tab1:
        st.subheader("Income Distribution vs. Attrition")
        fig_inc = px.histogram(df, x="MonthlyIncome", color="Attrition_Label", 
                               marginal="box", nbins=30, title="Does Salary Impact Churn?",
                               color_discrete_map={"Yes": "red", "No": "blue"})
        st.plotly_chart(fig_inc, use_container_width=True)
        
        st.caption("Observation: Higher attrition density observed in lower income brackets (Synthetic Data).")

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_box = px.box(df, x="JobRole", y="YearsAtCompany", color="JobRole", title="Tenure by Job Role")
            st.plotly_chart(fig_box, use_container_width=True)
        with col_b:
            # Heatmap approximation
            corr = df[['Age', 'MonthlyIncome', 'JobSatisfaction', 'YearsAtCompany']].corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# PAGE 3: ATTRITION PREDICTOR (ML)
# ==========================================
elif page == "Attrition Predictor (ML)":
    st.title("🤖 AI Attrition Forecaster")
    st.markdown("Predict the likelihood of an employee leaving based on key metrics.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Employee Profile")
        with st.form("prediction_form"):
            st.header("Department")
            dept_input = st.selectbox("Department:", ['Sales', 'R&D', 'HR'])
            st.header("Age")
            age_input = st.slider("Age:", 18, 65, 30)
            st.header("Monthly Income")
            income_input = st.number_input("Monthly Income ($)", 2000, 25000, 5000)
            st.header("Years at Company")
            tenure_input = st.slider("Years at Company", 0, 40, 5)
            st.header("Job Satisfaction")
            satisfaction_input = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
            
            submit_btn = st.form_submit_button("Analyze Risk")
    
    with col2:
        st.subheader("Prediction Results")
        if submit_btn:
            # Encode input to match training data
            dept_map = {'Sales': 3, 'R&D': 2, 'HR': 1, 'IT': 4, 'Finance': 0} # Simplified mapping
            dept_code = dept_map.get(dept_input, 0)
            
            input_data = np.array([[age_input, income_input, tenure_input, satisfaction_input, dept_code]])
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            # Display Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                title = {'text': "Attrition Probability (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red" if probability > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 100], 'color': "lightpink"}],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if probability > 0.5:
                ##st.error("⚠️ HIGH RISK: This employee is likely to leave. Recommended Action: Schedule 1:1 retention meeting.")
                st.markdown('<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">"⚠️ HIGH RISK: This employee is likely to leave. Recommended Action: Schedule 1:1 retention meeting."</p>', unsafe_allow_html=True)

            else:
                ##st.success("✅ LOW RISK: Employee appears stable.")
                st.markdown('<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">"✅ LOW RISK: Employee appears stable."</p>', unsafe_allow_html=True)

        else:
            st.info("Adjust the parameters on the left and click 'Analyze Risk' to see AI predictions.")

# ==========================================
# PAGE 4: RESUME MATCHER (NLP)
# ==========================================
elif page == "Resume Matcher (NLP)":
    st.title("📄 Smart Resume Screening")
    st.markdown("Rank candidates against job descriptions using TF-IDF & Cosine Similarity.")
    
    # 1. Inputs
    c1, c2 = st.columns(2)
    with c1:
        job_desc = st.text_area("Paste Job Description:", 
                                "Looking for a Data Scientist with Python, SQL, and Machine Learning experience. Must know PyTorch.")
    
    with c2:
        # Hardcoded synthetic resumes for demo
        candidate_data = {
            "Alice (Data Scientist)": "Expert in Python, Pandas, Scikit-Learn and SQL. 5 years in Data Science.",
            "Bob (HR Manager)": " experienced in recruitment, payroll, and employee engagement. Good communication.",
            "Charlie (Junior Dev)": "Fresh graduate knowing Python basics and SQL. Learning PyTorch.",
            "Diana (AI Engineer)": "Deep Learning specialist. Expert in PyTorch, Transformers, NLP, and Python."
        }
        st.write("Candidates in Pool:", len(candidate_data))
        st.json(list(candidate_data.keys()), expanded=False)

    # 2. Match Logic
    if st.button("Run AI Matching"):
        names = list(candidate_data.keys())
        resumes = list(candidate_data.values())
        
        # Vectorize
        vectorizer = TfidfVectorizer()
        corpus = resumes + [job_desc]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Calculate Sim
        job_vec = tfidf_matrix[-1]
        resume_vecs = tfidf_matrix[:-1]
        scores = cosine_similarity(resume_vecs, job_vec).flatten()
        
        # Create Result DF
        results = pd.DataFrame({'Candidate': names, 'Match Score': scores * 100})
        results = results.sort_values(by='Match Score', ascending=False)
        
        # 3. Visuals
        st.subheader("Ranking Results")
        
        col_res, col_chart = st.columns([1, 2])
        
        with col_res:
            st.dataframe(results.style.background_gradient(cmap="Greens", subset=["Match Score"]), hide_index=True)
            
        with col_chart:
            fig_match = px.bar(results, x='Match Score', y='Candidate', orientation='h', 
                               text_auto='.2f', title="Candidate Relevance Score", color='Match Score')
            st.plotly_chart(fig_match, use_container_width=True)

# ==========================================
# PAGE 5: HR CHATBOT
# ==========================================
elif page == "HR Assistant (Chatbot)":
    st.title("💬 T-IQ Assistant")
    st.markdown("Ask questions about policies, attrition data, or hiring status.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your HR Assistant. Ask me about the attrition rate or hiring status."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your query here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Simple Rule-based Logic (Mocking LLM)
        response = ""
        prompt_lower = prompt.lower()
        
        if "attrition" in prompt_lower:
            rate = (df['Attrition'].sum() / len(df)) * 100
            response = f"Current attrition rate is **{rate:.1f}%**. This is slightly above our target of 15%."
        elif "hire" in prompt_lower or "hiring" in prompt_lower:
            response = "We have **12 open positions** in the R&D and Sales departments. Resume screening is 60% complete."
        elif "policy" in prompt_lower:
            response = "Please refer to the employee handbook section 4.2 for leave policies. Generally, employees are entitled to 20 days PTO."
        else:
            response = "I'm currently a demo bot. I can answer questions about 'attrition', 'hiring', or 'policy'."

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.markdown("© 2024 T-IQ Suite Prototype | Generated for Educational Purposes")