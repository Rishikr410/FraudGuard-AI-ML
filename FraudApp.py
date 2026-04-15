import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import VarianceThreshold

# --- Page Configuration ---
st.set_page_config(page_title="FraudGuard AI", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Professional Dashboard Look ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'data' not in st.session_state: st.session_state.data = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'step' not in st.session_state: st.session_state.step = 1

def move_to(step_idx):
    st.session_state.step = step_idx
    st.rerun()

# --- Sidebar Navigation ---
st.sidebar.title("🛡️ FraudGuard AI")
steps = [
    "1. Dataset Upload", "2. Preprocessing", "3. EDA", 
    "4. Outlier Detection", "5. Clustering", "6. Feature Selection",
    "7. Model Insight", "8. Dashboard", "9. Final Result"
]
selected_step = st.sidebar.radio("Pipeline Navigation", steps, index=st.session_state.step - 1)
current_step = steps.index(selected_step) + 1

# --- MODULE 1: Dataset Upload ---
if current_step == 1:
    st.header("📤 Dataset Upload")
    uploaded_file = st.file_uploader("Upload Transactional CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        
        c1, c2 = st.columns(2)
        c1.metric("Total Records", df.shape[0])
        c2.metric("Total Features", df.shape[1])
        
        st.write("### Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("Proceed to Preprocessing"): move_to(2)

# --- MODULE 2: Preprocessing ---
elif current_step == 2:
    if st.session_state.data is not None:
        st.header("⚙️ Data Preprocessing")
        df = st.session_state.data.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            impute_strat = st.selectbox("Missing Value Strategy", ["mean", "median", "most_frequent"])
            encode_strat = st.selectbox("Encoding Type", ["Label Encoding", "One-Hot Encoding"])
        
        if st.button("Run Preprocessing Pipeline"):
            # 1. Handle Missing Values
            num_cols = df.select_dtypes(include=[np.number]).columns
            cat_cols = df.select_dtypes(exclude=[np.number]).columns
            
            imputer = SimpleImputer(strategy=impute_strat)
            if len(num_cols) > 0:
                df[num_cols] = imputer.fit_transform(df[num_cols])
            
            # 2. Encoding
            if encode_strat == "Label Encoding":
                for col in cat_cols:
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            else:
                df = pd.get_dummies(df, columns=cat_cols)
            
            # 3. Scaling
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            st.session_state.processed_data = df_scaled
            st.success("Data Scaled and Cleaned Successfully!")
            st.dataframe(df_scaled.head())

        if st.session_state.processed_data is not None:
            if st.button("Next: Exploratory Data Analysis"): move_to(3)
    else: st.warning("Please upload a dataset first.")

# --- MODULE 3: EDA ---
elif current_step == 3:
    st.header("📊 Exploratory Data Analysis")
    df = st.session_state.processed_data
    
    tab1, tab2 = st.tabs(["Correlation Heatmap", "Distributions"])
    with tab1:
        fig_corr = px.imshow(df.corr(), color_continuous_scale='RdBu_r', title="Feature Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)
    with tab2:
        feat = st.selectbox("Select Feature to view Distribution", df.columns)
        fig_dist = px.histogram(df, x=feat, marginal="violin", color_discrete_sequence=['#ff4b4b'])
        st.plotly_chart(fig_dist, use_container_width=True)
    
    if st.button("Next: Outlier Detection"): move_to(4)

# --- MODULE 4: Outlier Detection ---
elif current_step == 4:
    st.header("🕵️ Anomaly Detection (Fraud Core)")
    df = st.session_state.processed_data.copy()
    
    # CRITICAL FIX: Ensure no duplicate columns on rerun
    if 'Is_Outlier' in df.columns:
        df = df.drop(columns=['Is_Outlier'])
    
    method = st.selectbox("Choose Detection Algorithm", ["Isolation Forest", "IQR Method"])
    
    if method == "Isolation Forest":
        contam = st.slider("Contamination (Expected Fraud %)", 0.01, 0.2, 0.05)
        model = IsolationForest(contamination=contam, random_state=42)
        df['Is_Outlier'] = [1 if x == -1 else 0 for x in model.fit_predict(df)]
    else:
        Q1, Q3 = df.quantile(0.25), df.quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
        df['Is_Outlier'] = outlier_mask.astype(int)
    
    fraud_pct = (df['Is_Outlier'].sum() / len(df)) * 100
    st.metric("Fraud Detection Rate", f"{fraud_pct:.2f}%")
    
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color=df['Is_Outlier'].astype(str), 
                     color_discrete_map={'0':'#636EFA', '1':'#EF553B'}, 
                     title="Anomalies Highlighted (Red = Potential Fraud)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.session_state.processed_data = df
    if st.button("Next: Clustering"): move_to(5)

# --- MODULE 5: Clustering ---
elif current_step == 5:
    st.header("🧩 Clustering Analysis")
    df = st.session_state.processed_data.copy()
    
    # CRITICAL FIX: Ensure no duplicate columns on rerun
    if 'Cluster' in df.columns:
        df = df.drop(columns=['Cluster'])
        
    feat_cols = [c for c in df.columns if c != 'Is_Outlier']
    
    algo = st.selectbox("Clustering Method", ["K-Means", "DBSCAN"])
    if algo == "K-Means":
        k = st.slider("Number of Clusters", 2, 8, 3)
        df['Cluster'] = KMeans(n_clusters=k, random_state=42).fit_predict(df[feat_cols])
    else:
        df['Cluster'] = DBSCAN(eps=0.5).fit_predict(df[feat_cols])
    
    pca = PCA(n_components=2).fit_transform(df[feat_cols])
    pca_df = pd.DataFrame(pca, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = df['Cluster'].astype(str)
    
    st.plotly_chart(px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title="PCA Cluster Map"), use_container_width=True)
    st.session_state.processed_data = df
    if st.button("Next: Feature Selection"): move_to(6)

# --- MODULE 6: Feature Selection ---
elif current_step == 6:
    st.header("✂️ Feature Selection")
    df = st.session_state.processed_data
    var_thresh = st.slider("Variance Threshold (Filter low variance)", 0.0, 1.0, 0.01)
    
    # Keeping labels safe
    cols_to_keep = ['Is_Outlier', 'Cluster']
    labels_present = [c for c in cols_to_keep if c in df.columns]
    features_only = df.drop(columns=labels_present)
    
    selector = VarianceThreshold(threshold=var_thresh)
    selector.fit(features_only)
    
    selected_feat_names = features_only.columns[selector.get_support()].tolist()
    st.session_state.processed_data = df[selected_feat_names + labels_present]
    
    st.write(f"Selected {len(selected_feat_names)} features.")
    st.dataframe(st.session_state.processed_data.head())
    
    if st.button("Next: Supervised Insight"): move_to(7)

# --- MODULE 7: Model Insight ---
elif current_step == 7:
    st.header("🤖 Supervised Model Insight")
    df = st.session_state.processed_data
    target = st.selectbox("Select Target Label (if exists)", ["None"] + list(df.columns))
    
    if target != "None":
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier().fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2f}")
        c3.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.2f}")
    else:
        st.info("No label selected. You can skip to the dashboard.")
    
    if st.button("Next: Dashboard"): move_to(8)

# --- MODULE 8: Dashboard ---
elif current_step == 8:
    st.header("📈 Visualization Dashboard")
    df = st.session_state.processed_data
    
    c1, c2 = st.columns(2)
    with c1:
        if 'Is_Outlier' in df.columns:
            counts = df['Is_Outlier'].value_counts().reset_index()
            counts.columns = ['Status', 'Count']
            counts['Status'] = counts['Status'].astype(str).replace({'0': 'Normal', '1': 'Fraud'})
            st.plotly_chart(px.pie(counts, values='Count', names='Status', hole=0.4, title="Fraud Proportion"), use_container_width=True)
    with c2:
        if 'Cluster' in df.columns:
            st.plotly_chart(px.histogram(df, x='Cluster', title="Volume by Cluster"), use_container_width=True)
            
    if st.button("Next: Final Result"): move_to(9)

# --- MODULE 9: Final Result ---
elif current_step == 9:
    st.header("🏁 Final System Audit")
    df = st.session_state.processed_data
    
    if 'Is_Outlier' in df.columns:
        fraud_count = int(df['Is_Outlier'].sum())
        fraud_pct = (fraud_count / len(df)) * 100
        
        if fraud_count > 0:
            st.error(f"### 🚨 FRAUD DETECTED")
            st.metric("Fraud Rate", f"{fraud_pct:.2f}%")
            st.warning(f"Total of {fraud_count} records were flagged as highly suspicious.")
            st.subheader("🚩 Suspicious Transactions Detail")
            st.dataframe(df[df['Is_Outlier'] == 1], use_container_width=True)
        else:
            st.success("### ✅ NO FRAUD DETECTED")
            st.balloons()
            st.write("All transactions are categorized as normal.")
    
    st.markdown("---")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Flagged Report", data=csv, file_name="fraud_audit_report.csv")
    
    if st.button("🔄 Restart New Analysis"):
        st.session_state.clear()
        st.rerun()