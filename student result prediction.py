pip install PyPDF2

# Install all necessary libraries for the Streamlit application
!pip install streamlit scikit-learn pandas numpy matplotlib seaborn


import PyPDF2

pdf_path = '/content/student ai project.pdf'

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
    except Exception as e:
        text = f"Error reading PDF: {e}"
    return text

pdf_content = extract_text_from_pdf(pdf_path)

print("--- Extracted PDF Content (first 1000 characters) ---")
print(pdf_content[:1000])
print("\n--- End of preview ---")


import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Number of synthetic student records
num_students = 500

# Generate synthetic data for features based on ranges described in the PDF
hours_study = np.random.randint(1, 13, num_students)  # 1 to 12 hours
attendance = np.random.randint(40, 101, num_students) # 40 to 100%
previous_marks = np.random.randint(30, 101, num_students) # 30 to 100
assignments_done = np.random.randint(0, 11, num_students) # 0 to 10
sleep_hours = np.random.randint(4, 11, num_students) # 4 to 10 hours

# Create a 'realistic' formula for marks to simulate Pass/Fail
# This formula aims to create a distribution where higher feature values lead to higher marks
# and consequently, a higher chance of passing.
simulated_marks = (
    (hours_study * 5) +
    (attendance * 0.5) +
    (previous_marks * 0.7) +
    (assignments_done * 4) +
    (sleep_hours * 2)
)

# Normalize simulated_marks to be within a reasonable range, e.g., 0-100
simulated_marks = np.interp(simulated_marks, (simulated_marks.min(), simulated_marks.max()), (30, 100))

# Determine Result: 0 = Fail (marks < 50), 1 = Pass (marks >= 50)
result = (simulated_marks >= 50).astype(int)

# Create DataFrame
data = {
    'Hours_Study': hours_study,
    'Attendance': attendance,
    'Previous_Marks': previous_marks,
    'Assignments_Done': assignments_done,
    'Sleep_Hours': sleep_hours,
    'Result': result # Target variable
}
df = pd.DataFrame(data)

print("Synthetic dataset generated successfully. Displaying first 5 rows:")
display(df.head())
print("\nDistribution of Result (Pass/Fail):")
display(df['Result'].value_counts())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define features (X) and target (y)
X = df[['Hours_Study', 'Attendance', 'Previous_Marks', 'Assignments_Done', 'Sleep_Hours']]
y = df['Result']

# Split data into training and testing sets (80/20 split as per PDF)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and fit StandardScaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data split and scaled successfully.")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")


from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model with specified parameters
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model using the scaled training data
model.fit(X_train_scaled, y_train)

print("Logistic Regression model trained successfully.")


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Make predictions on the scaled test set
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Example input for a new student
# Values should be within the ranges used for synthetic data generation
# Hours_Study (1-12), Attendance (40-100), Previous_Marks (30-100),
# Assignments_Done (0-10), Sleep_Hours (4-10)

new_student_data = {
    'Hours_Study': 9,
    'Attendance': 85,
    'Previous_Marks': 75,
    'Assignments_Done': 8,
    'Sleep_Hours': 7
}

# Convert to DataFrame for consistent scaling
new_student_df = pd.DataFrame([new_student_data])

# Scale the new student's data using the *same scaler* fitted on the training data
new_student_scaled = scaler.transform(new_student_df)

# Make a prediction (0 for Fail, 1 for Pass)
prediction = model.predict(new_student_scaled)[0]

# Get prediction probabilities (e.g., [prob_fail, prob_pass])
probabilities = model.predict_proba(new_student_scaled)[0]

predicted_outcome = 'Pass' if prediction == 1 else 'Fail'
confidence = probabilities[prediction] * 100

print("\n--- New Student Prediction ---")
print(f"Input Data: {new_student_data}")
print(f"Predicted Outcome: {predicted_outcome}")
print(f"Confidence: {confidence:.2f}%")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get feature names from the original DataFrame
feature_names = X.columns

# Get the coefficients from the trained Logistic Regression model
coefficients = model.coef_[0]

# Create a DataFrame for coefficients and their absolute values
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})

# Sort by absolute coefficient value to see most important features
feature_importance = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

print("Feature Coefficients and Importance:")
display(feature_importance)

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Abs_Coefficient', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance (Absolute Coefficients of Logistic Regression)')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.show()


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Student Result Prediction System")

# --- 1. Data Generation and Model Training (as in previous cells) ---
@st.cache_resource # Cache the data generation and model training for efficiency
def load_model_and_data():
    np.random.seed(42)
    num_students = 500

    hours_study = np.random.randint(1, 13, num_students)
    attendance = np.random.randint(40, 101, num_students)
    previous_marks = np.random.randint(30, 101, num_students)
    assignments_done = np.random.randint(0, 11, num_students)
    sleep_hours = np.random.randint(4, 11, num_students)

    simulated_marks = (
        (hours_study * 5) +
        (attendance * 0.5) +
        (previous_marks * 0.7) +
        (assignments_done * 4) +
        (sleep_hours * 2)
    )
    simulated_marks = np.interp(simulated_marks, (simulated_marks.min(), simulated_marks.max()), (30, 100))
    result = (simulated_marks >= 50).astype(int)

    data = {
        'Hours_Study': hours_study,
        'Attendance': attendance,
        'Previous_Marks': previous_marks,
        'Assignments_Done': assignments_done,
        'Sleep_Hours': sleep_hours,
        'Result': result
    }
    df = pd.DataFrame(data)

    X = df[['Hours_Study', 'Attendance', 'Previous_Marks', 'Assignments_Done', 'Sleep_Hours']]
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    return df, X, y, X_test, y_test, model, scaler

df, X, y, X_test, y_test, model, scaler = load_model_and_data()

# --- Streamlit UI ---
st.title("Student Result Prediction System")
st.markdown("A Machine Learning web application to predict student pass/fail results.")

tab1, tab2, tab3 = st.tabs(["Predict Result", "Data Analysis", "Model Performance"])

with tab1:
    st.header("Predict Student Result")
    st.markdown("Enter the student's academic and lifestyle parameters to get a prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        hours_study = st.slider("Daily Study Hours (1-12)", 1, 12, 5)
        attendance = st.slider("Attendance Percentage (40-100%)", 40, 100, 75)

    with col2:
        previous_marks = st.slider("Previous Exam Marks (30-100)", 30, 100, 60)
        assignments_done = st.slider("Assignments Completed (0-10)", 0, 10, 5)

    with col3:
        sleep_hours = st.slider("Average Daily Sleep Hours (4-10)", 4, 10, 7)

    input_data = pd.DataFrame([[hours_study, attendance, previous_marks, assignments_done, sleep_hours]],
                              columns=X.columns)

    if st.button("Predict Result"):
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]

        predicted_outcome = 'PASS' if prediction == 1 else 'FAIL'
        confidence = probabilities[prediction] * 100

        if predicted_outcome == 'PASS':
            st.success(f"Predicted Result: **{predicted_outcome}**")
        else:
            st.error(f"Predicted Result: **{predicted_outcome}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Input Summary")
        st.table(input_data)

with tab2:
    st.header("Data Analysis")
    st.markdown("Visualizations and summary statistics of the synthetic student dataset.")

    st.subheader("Summary Statistics")
    total_students = len(df)
    pass_count = df['Result'].sum()
    fail_count = total_students - pass_count
    pass_rate = (pass_count / total_students) * 100

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Total Students", total_students)
    metrics_col2.metric("Pass Rate", f"{pass_rate:.2f}%")
    metrics_col3.metric("Fail Count", fail_count)

    st.subheader("Visualizations")

    # Study Hours vs Marks
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Hours_Study', y='Previous_Marks', hue='Result', data=df, palette='viridis', s=100, ax=ax1)
    ax1.set_title('Study Hours vs Previous Marks (Colored by Result)')
    ax1.set_xlabel('Daily Study Hours')
    ax1.set_ylabel('Previous Exam Marks')
    ax1.legend(title='Result', labels=['Fail', 'Pass'])
    st.pyplot(fig1)

    # Attendance vs Marks
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Attendance', y='Previous_Marks', hue='Result', data=df, palette='magma', s=100, ax=ax2)
    ax2.set_title('Attendance vs Previous Marks (Colored by Result)')
    ax2.set_xlabel('Attendance Percentage')
    ax2.set_ylabel('Previous Exam Marks')
    ax2.legend(title='Result', labels=['Fail', 'Pass'])
    st.pyplot(fig2)

    # Pass/Fail Pie Chart
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    df['Result'].map({0: 'Fail', 1: 'Pass'}).value_counts().plot.pie(
        autopct='%1.1f%%', startangle=90, colors=['salmon', 'lightgreen'],
        wedgeprops=dict(width=0.3), ax=ax3
    )
    ax3.set_title('Distribution of Student Results (Pass/Fail)')
    ax3.set_ylabel('')
    st.pyplot(fig3)

    # Marks Distribution Histogram
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Previous_Marks'], bins=20, kde=True, color='skyblue', ax=ax4)
    ax4.set_title('Distribution of Previous Exam Marks')
    ax4.set_xlabel('Previous Exam Marks')
    ax4.set_ylabel('Number of Students')
    st.pyplot(fig4)

with tab3:
    st.header("Model Performance")
    st.markdown("Key metrics and visualizations to assess the Logistic Regression model's effectiveness.")

    y_pred = model.predict(scaler.transform(X_test))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
    col_metrics1.metric("Accuracy", f"{accuracy:.4f}")
    col_metrics2.metric("Precision", f"{precision:.4f}")
    col_metrics3.metric("Recall", f"{recall:.4f}")
    col_metrics4.metric("F1 Score", f"{f1:.4f}")

    st.markdown(
        "_**Accuracy:** Percentage of students correctly classified (Pass or Fail)._\n"
        "_**Precision:** Of all students predicted as PASS, how many actually passed?_\n"
        "_**Recall:** Of all students who actually passed, how many were correctly identified?_\n"
        "_**F1 Score:** Harmonic mean of Precision and Recall, balancing both metrics._"
    )

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'], ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    st.pyplot(fig_cm)

    st.subheader("Feature Importance")
    feature_names = X.columns
    coefficients = model.coef_[0]
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    })
    feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)

    fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Abs_Coefficient', y='Feature', data=feature_importance_df, palette='viridis', ax=ax_fi)
    ax_fi.set_title('Feature Importance (Absolute Coefficients of Logistic Regression)')
    ax_fi.set_xlabel('Absolute Coefficient Value')
    ax_fi.set_ylabel('Feature')
    st.pyplot(fig_fi)


import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style('whitegrid')

# 1. Study Hours vs Marks (scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Hours_Study', y='Previous_Marks', hue='Result', data=df, palette='viridis', s=100)
plt.title('Study Hours vs Previous Marks (Colored by Result)')
plt.xlabel('Daily Study Hours')
plt.ylabel('Previous Exam Marks')
plt.legend(title='Result', labels=['Fail', 'Pass'])
plt.show()

# 2. Attendance vs Marks (scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Attendance', y='Previous_Marks', hue='Result', data=df, palette='magma', s=100)
plt.title('Attendance vs Previous Marks (Colored by Result)')
plt.xlabel('Attendance Percentage')
plt.ylabel('Previous Exam Marks')
plt.legend(title='Result', labels=['Fail', 'Pass'])
plt.show()

# 3. Pass/Fail pie chart
plt.figure(figsize=(7, 7))
df['Result'].map({0: 'Fail', 1: 'Pass'}).value_counts().plot.pie(
    autopct='%1.1f%%', startangle=90, colors=['salmon', 'lightgreen'],
    wedgeprops=dict(width=0.3)
)
plt.title('Distribution of Student Results (Pass/Fail)')
plt.ylabel('') # Hide the default 'Result' label
plt.show()

# 4. Marks distribution histogram (using Previous_Marks as a proxy for overall 'marks')
plt.figure(figsize=(8, 6))
sns.histplot(df['Previous_Marks'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Previous Exam Marks')
plt.xlabel('Previous Exam Marks')
plt.ylabel('Number of Students')
plt.show()

# Also show summary statistics (total students, pass count, fail count, pass rate)
print("\n--- Summary Statistics ---")
total_students = len(df)
pass_count = df['Result'].sum()
fail_count = total_students - pass_count
pass_rate = (pass_count / total_students) * 100

print(f"Total Students: {total_students}")
print(f"Pass Count: {pass_count}")
print(f"Fail Count: {fail_count}")
print(f"Pass Rate: {pass_rate:.2f}%")

