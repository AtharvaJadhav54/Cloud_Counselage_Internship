import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('PS2_Dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'PS2_Dataset.csv' not found. Please make sure the dataset is in the correct directory.")
        return None

df = load_data()

@st.cache_resource
def train_model(df):
    df.dropna(subset=['Suggested Job Role'], inplace=True)
    
    # Define features (X) and target (y)
    X = df.drop('Suggested Job Role', axis=1)
    y = df['Suggested Job Role']

    # --- Data Cleaning and Preparation ---
    # Clean categorical columns to have consistent, manageable categories
    def clean_column(series, mapping, default):
        def find_match(value):
            for keyword, clean_category in mapping.items():
                if keyword in str(value).lower():
                    return clean_category
            return default
        return series.apply(find_match)

    X['Interested subjects'] = clean_column(X['Interested subjects'], {'programming': 'Programming', 'data': 'Data Science', 'network': 'Networking', 'software': 'Software Engineering', 'cloud': 'Cloud Computing', 'security': 'Cybersecurity'}, 'Programming')
    X['interested career area '] = clean_column(X['interested career area '], {'system': 'System Development', 'business': 'Business Analysis', 'security': 'Security', 'data': 'Data Engineering', 'web': 'Web Development'}, 'System Development')
    X['Type of company want to settle in?'] = clean_column(X['Type of company want to settle in?'], {'product': 'Product Based', 'service': 'Service Based', 'startup': 'Startup', 'cloud': 'Cloud Services', 'finance': 'Finance'}, 'Product Based')
    X['Interested Type of Books'] = clean_column(X['Interested Type of Books'], {'technical': 'Technical', 'sci': 'Science Fiction', 'bio': 'Biographies', 'self': 'Self-help', 'journal': 'Journals'}, 'Technical')

    # Use One-Hot Encoding for categorical features
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object', 'bool']).columns, drop_first=True)
    
    # Encode the target variable
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Align columns after splitting to handle potential missing columns from one-hot encoding
    train_cols = X_train.columns
    test_cols = X_test.columns
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0
    X_test = X_test[train_cols] # Ensure order is the same

    # --- Model Training ---
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred_test = pipeline.predict(X_test)
    report = classification_report(le_y.inverse_transform(y_test), le_y.inverse_transform(y_pred_test), output_dict=True, zero_division=0)
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({'feature': train_cols, 'importance': importances})
    
    return pipeline, le_y, report, X.columns, df, importance_df

if df is not None:
    model, label_encoder_y, report, trained_features, original_df, importance_df = train_model(df.copy())

def get_course_recommendations(predicted_role):
    course_map = {
        "Applications Developer": ["Advanced Java & Web Development", "MERN Stack Development", "Python with Django"],
        "Web Developer": ["Full Stack Web Development (MERN)", "JavaScript for Beginners", "Advanced CSS and Sass"],
        "Software Developer": ["Data Structures & Algorithms in Python", "Object-Oriented Programming in C++", "Software Engineering Principles"],
        "Software Engineer": ["Advanced Algorithms", "System Design and Architecture", "Cloud Computing with AWS"],
        "Business Process Analyst": ["Business Analysis Fundamentals", "Agile & Scrum Master Certification", "SQL for Data Analysis"],
        "Database Administrator": ["Oracle Database Administration", "SQL Server for Beginners", "NoSQL Databases: MongoDB & Cassandra"]
    }
    return course_map.get(predicted_role, ["Generic Software Development Course", "Explore courses on Coursera or Udemy"])

st.set_page_config(page_title="Career Path Prediction System", layout="wide")
st.title("🎓 Career Path Prediction System")
st.write("This tool uses a machine learning model to predict a suitable career path based on your skills and interests.")

numerical_ui_features = ['Logical quotient rating', 'hackathons', 'coding skills rating', 'public speaking points']
categorical_ui_features = {
    'Interested subjects': ['Programming', 'Data Science', 'Networking', 'Software Engineering', 'Cloud Computing', 'Cybersecurity'],
    'interested career area ': ['System Development', 'Business Analysis', 'Security', 'Data Engineering', 'Web Development'],
    'Type of company want to settle in?': ['Product Based', 'Service Based', 'Startup', 'Cloud Services', 'Finance'],
    'Taken inputs from seniors or elders': ['yes', 'no'],
    'Interested Type of Books': ['Technical', 'Science Fiction', 'Biographies', 'Self-help', 'Journals'],
    'Management or Technical': ['Management', 'Technical'],
    'hard/smart worker': ['Smart Worker', 'Hard Worker'],
    'worked in teams ever?': ['yes', 'no'],
    'Introvert': ['yes', 'no'],
    'self-learning capability?': ['yes', 'no'],
    'Extra-courses did': ['yes', 'no'],
    'certifications': ['information security', 'shell programming', 'r programming', 'distro making', 'machine learning', 'full stack'],
    'workshops': ['testing', 'database security', 'game development', 'data science', 'hacking', 'cloud computing'],
    'reading and writing skills': ['poor', 'medium', 'excellent'],
    'memory capability score': ['poor', 'medium', 'excellent']
}

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Enter Your Details")
    user_inputs = {}
    for feature in numerical_ui_features:
        user_inputs[feature] = st.radio(f"Rate your '{feature}' (1=Low, 5=High)", list(range(1, 6)), horizontal=True)
    
    for feature, options in categorical_ui_features.items():
        user_inputs[feature] = st.selectbox(f"Select your '{feature}'", options)

    if st.button("✨ Predict My Career Path", use_container_width=True):
        input_df = pd.DataFrame([user_inputs])
        
        # One-Hot Encode the user input to match the training data format
        input_processed = pd.get_dummies(input_df)
        
        # Align columns with the model's training data
        input_aligned = input_processed.reindex(columns=trained_features, fill_value=0)
        
        prediction_encoded = model.predict(input_aligned)
        predicted_role = label_encoder_y.inverse_transform(prediction_encoded)[0]
        courses = get_course_recommendations(predicted_role)

        st.success(f"### Predicted Career Path: **{predicted_role}**")
        st.subheader("Recommended Courses to Excel:")
        for course in courses:
            st.markdown(f"- **{course}**")

    st.header("🧠 Model Insights")
    st.subheader("Key Factors in Prediction")
    fig_imp = px.bar(importance_df.nlargest(10, 'importance').sort_values('importance', ascending=True), 
                     x='importance', y='feature', orientation='h', 
                     title='Top 10 Most Important Features', 
                     color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Skill & Attribute Correlations")
    # Show correlation only for original numerical features for clarity
    corr = original_df[numerical_ui_features].corr()
    fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
    fig_corr.update_layout(title='Correlation Heatmap of Numerical Skills')
    st.plotly_chart(fig_corr, use_container_width=True)

with col2:
    st.header("Model Performance")
    st.subheader("Performance Breakdown (on Test Data)")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df[['precision', 'recall', 'f1-score', 'support']].style.format("{:.2f}"))

    st.header("Exploratory Data Analysis")
    fig1 = px.histogram(original_df, x='Suggested Job Role', title='Distribution of Career Roles', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(original_df, x='Suggested Job Role', y='coding skills rating', title='Coding Skills by Career Role', color='Suggested Job Role')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Average Skill Profile by Top Career Paths")
    top_roles = original_df['Suggested Job Role'].value_counts().nlargest(4).index
    skill_columns = ['coding skills rating', 'hackathons', 'public speaking points', 'self-learning capability?']
    
    df_radar = original_df[original_df['Suggested Job Role'].isin(top_roles)].copy()
    df_radar['self-learning capability?'] = df_radar['self-learning capability?'].apply(lambda x: 1 if x == 'yes' else 0) * 5
    radar_data = df_radar.groupby('Suggested Job Role')[skill_columns].mean().reset_index()
    
    fig_radar = go.Figure()
    for i, row in radar_data.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=row[skill_columns].values,
            theta=skill_columns,
            fill='toself',
            name=row['Suggested Job Role']
        ))
    fig_radar.update_layout(title='Skill Profiles for Top 4 Careers')
    st.plotly_chart(fig_radar, use_container_width=True)
