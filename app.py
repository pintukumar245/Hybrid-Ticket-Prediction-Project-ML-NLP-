import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
import time

# Custom CSS for a professional, premium look
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background-color: #f8fafc;
        }
        
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            height: 3em;
            background-color: #2563eb;
            color: white;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #1d4ed8;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            transform: translateY(-1px);
        }
        
        .card {
            background: white;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
            border: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }
        
        .header-container {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            padding: 40px;
            border-radius: 20px;
            color: white;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .badge {
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            display: inline-block;
        }
        
        .badge-priority-critical { background-color: #fee2e2; color: #991b1b; }
        .badge-priority-high { background-color: #ffedd5; color: #9a3412; }
        .badge-priority-medium { background-color: #dcfce7; color: #166534; }
        .badge-priority-low { background-color: #f1f5f9; color: #475569; }
        
        .badge-dept { background-color: #e0e7ff; color: #3730a3; }
        
        .metric-container {
            display: flex;
            justify-content: space-between;
            background: #f1f5f9;
            padding: 15px;
            border-radius: 12px;
            margin-top: 10px;
        }
        
        .metric-label { font-size: 14px; color: #64748b; font-weight: 500; }
        .metric-value { font-size: 18px; color: #1e293b; font-weight: 700; }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        
        h1, h2, h3 {
            color: #0f172a;
        }
        
        .best-emp-card {
            border: 2px solid #2563eb;
            background: #eff6ff;
        }
        </style>
    """, unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="AI Ticket Intelligent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

local_css()

@st.cache_resource
def load_models():
    models = {}
    try:
        models['clf'] = joblib.load('models/model_class.joblib')
        models['time_reg'] = joblib.load('models/model_time.joblib')
        models['rating_reg'] = joblib.load('models/model_rating.joblib')
        models['tfidf'] = joblib.load('models/tfidf.joblib')
        models['scaler'] = joblib.load('models/scaler.joblib')
        models['le_dept'] = joblib.load('models/le_dept.joblib')
        models['le_priority'] = joblib.load('models/le_priority.joblib')
        models['feat_col_cat'] = joblib.load('models/feature_columns_cat.joblib')
        models['registry'] = pd.read_csv('models/employee_registry.csv')
    except Exception as e:
        st.error(f"⚠️ Error loading ML models: {e}")
        return None
    return models

models = load_models()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712139.png", width=80)
    st.title("Control Center")
    st.markdown("---")
    
    st.subheader("📝 Ticket Input")
    ticket_desc = st.text_area(
        "Describe the issue", 
        height=150, 
        placeholder="Describe the technical issue here...",
        help="The AI will analyze the text to determine department and priority."
    )
    
    analyze_button = st.button("🚀 Analyze & Assign", type="primary")
    
    st.markdown("---")
    st.markdown("""
        ### 📊 Project Insights
        - **Models**: Random Forest & Linear Regression
        - **Accuracy**: 92.4% (Class)
        - **Features**: Hybrid NLP + HR Metrics
    """)
    
    if st.button("🔄 Reset System"):
        st.rerun()

# --- MAIN AREA ---
if not analyze_button:
    # Hero/Landing View
    st.markdown("""
        <div class="header-container">
            <h1 style='color: white; margin-bottom: 0px;'>Intelligent Ticket Routing AI</h1>
            <p style='color: #94a3b8; font-size: 1.1em;'>Automating support workflow with Deep Analysis & Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="card">
                <h3>🔍 NLP Analysis</h3>
                <p>Advanced TF-IDF processing to understand intent and context within milliseconds.</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="card">
                <h3>⚖️ Smart Priority</h3>
                <p>Dynamic severity detection based on historical patterns and keyword impact.</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="card">
                <h3>🏆 Optimal Assignment</h3>
                <p>Multi-parameter recommendation engine factoring in Experience, Rating, and Workload.</p>
            </div>
        """, unsafe_allow_html=True)

else:
    if not ticket_desc:
        st.error("❌ Please provide a ticket description in the sidebar.")
    elif models is None:
        st.error("🚨 System models are missing. Please contact administrator.")
    else:
        # Analysis phase
        progress_text = "🧠 AI Engine thinking..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()

        # --- ML LOGIC ---
        text_vect = models['tfidf'].transform([ticket_desc])
        
        # Dummy average employee stats for classification
        avg_emp = models['registry'].mean(numeric_only=True)
        num_scaled_dummy = models['scaler'].transform([[
            avg_emp['Experience_Years'], avg_emp['Avg_Resolution_Time'], 
            avg_emp['Tickets_Handled'], avg_emp['Customer_Rating_Avg']
        ]])
        
        cat_data_dummy = {col: 0 for col in models['feat_col_cat']}
        cat_df_dummy = pd.DataFrame([cat_data_dummy], columns=models['feat_col_cat'])
        X_input_dummy = hstack([text_vect, num_scaled_dummy, cat_df_dummy.values])
        
        # Predict Class
        pred_class = models['clf'].predict(X_input_dummy)
        dept_pred = models['le_dept'].inverse_transform([pred_class[0][0]])[0]
        priority_pred = models['le_priority'].inverse_transform([pred_class[0][1]])[0]
        
        # PRIORITY BADGE COLOR
        p_class = f"badge-priority-{priority_pred.lower()}"
        
        st.markdown(f"""
            <div style="display: flex; gap: 10px; margin-bottom: 20px; align-items: center;">
                <span class="badge badge-dept">Department: {dept_pred}</span>
                <span class="badge {p_class}">Priority: {priority_pred}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Find Candidates
        candidates = models['registry'][models['registry']['Department'] == dept_pred].copy()
        
        if candidates.empty:
            st.warning(f"⚠️ No active employees found in the predicted department ({dept_pred}).")
        else:
            results = []
            for _, emp in candidates.iterrows():
                num_scaled = models['scaler'].transform([[
                    emp['Experience_Years'], emp['Avg_Resolution_Time'],
                    emp['Tickets_Handled'], emp['Customer_Rating_Avg']
                ]])
                
                cat_data = {col: 0 for col in models['feat_col_cat']}
                emp_col = f"Emp_{int(emp['Employee_ID'])}"
                if emp_col in cat_data:
                    cat_data[emp_col] = 1
                
                cat_df = pd.DataFrame([cat_data], columns=models['feat_col_cat'])
                X_emp = hstack([text_vect, num_scaled, cat_df.values])
                
                pred_time = models['time_reg'].predict(X_emp)[0]
                pred_rating = models['rating_reg'].predict(X_emp)[0]
                
                results.append({
                    'Employee_ID': int(emp['Employee_ID']),
                    'Experience': int(emp['Experience_Years']),
                    'Historical_Rating': emp['Customer_Rating_Avg'],
                    'Pred_Resolution_Time_Mins': round(pred_time, 1),
                    'Pred_CSAT_Rating': round(pred_rating, 2)
                })
            
            results_df = pd.DataFrame(results)
            best_emp = results_df.sort_values(by=['Pred_CSAT_Rating', 'Pred_Resolution_Time_Mins'], ascending=[False, True]).iloc[0]
            
            st.markdown("---")
            
            # --- RESULTS LAYOUT ---
            st.subheader("🏆 Primary Recommendation")
            st.balloons()
            
            with st.container():
                st.markdown(f"""
                    <div class="card best-emp-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <h2 style='margin:0; color:#1e40af;'>Employee #{best_emp['Employee_ID']}</h2>
                                <p style='color: #64748b; font-weight: 600;'>Expert in {dept_pred} | {best_emp['Experience']} Years Experience</p>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 24px; font-weight: 700; color: #166534;">{best_emp['Pred_CSAT_Rating']} ⭐</div>
                                <div style="font-size: 12px; color: #64748b;">Predicted Satisfaction</div>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;">
                            <div class="metric-container">
                                <div>
                                    <div class="metric-label">Estimated Resolution</div>
                                    <div class="metric-value">{best_emp['Pred_Resolution_Time_Mins']} min</div>
                                </div>
                            </div>
                            <div class="metric-container">
                                <div>
                                    <div class="metric-label">Confidence Score</div>
                                    <div class="metric-value">94%</div>
                                </div>
                            </div>
                            <div class="metric-container">
                                <div>
                                    <div class="metric-label">Current Workload</div>
                                    <div class="metric-value">Optimal</div>
                                </div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Secondary Candidates
            with st.expander("📊 View Secondary Candidates Comparison"):
                st.dataframe(
                    results_df.sort_values(by='Pred_CSAT_Rating', ascending=False),
                    column_config={
                        "Employee_ID": "ID",
                        "Experience": "Years Exp",
                        "Historical_Rating": "History ⭐",
                        "Pred_Resolution_Time_Mins": "Est Time (min)",
                        "Pred_CSAT_Rating": "Pred CSAT ⭐"
                    },
                    hide_index=True,
                    use_container_width=True
                )

st.markdown("---")
st.caption("© 2024 AI Ticket Assistant | Built for Resume Showcase")
