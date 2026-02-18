import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from scipy.sparse import hstack
import joblib
import os

def train_and_save():
    # 1. Load Dataset
    csv_path = "final_hybrid_ticket_employee_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found.")
        return

    print(f"Loading dataset from {csv_path}...")
    # Read CSV
    hybrid_dataset = pd.read_csv(csv_path)
    
    # 2. Map Columns (Handling _x suffixes from previous merges)
    print("Mapping columns...")
    # Use _x columns as primary source for employee data
    col_mapping = {
        'Department_x': 'Department',
        'Experience_Years_x': 'Experience_Years',
        'Avg_Resolution_Time_x': 'Avg_Resolution_Time',
        'Tickets_Handled_x': 'Tickets_Handled',
        'Customer_Rating_Avg_x': 'Customer_Rating_Avg'
    }
    
    # Rename if they exist, otherwise assume standard names might exist (fallback)
    hybrid_dataset.rename(columns=col_mapping, inplace=True, errors='ignore')

    # Check for 'Department' again to be safe
    if 'Department' not in hybrid_dataset.columns and 'Department_y' in hybrid_dataset.columns:
         hybrid_dataset.rename(columns={'Department_y': 'Department'}, inplace=True)

    print("Using provided hybrid dataset...")
    
    # Check if necessary columns exist
    required_cols = ['Ticket Description', 'Ticket Priority', 'Department', 'Assigned_Employee_ID', 
                     'Experience_Years', 'Avg_Resolution_Time', 'Tickets_Handled', 'Customer_Rating_Avg']
    
    # Quick fix for typo if needed, though file check showed Tickets_Handled_x seems correct spelling for 'Tickets'
    # But let's verify if any key is missing
    missing = [c for c in required_cols if c not in hybrid_dataset.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
        # Attempt to find close matches or stop
        return

    print(f"Dataset Shape: {hybrid_dataset.shape}")
    
    # 3. Preprocessing
    print("Preprocessing...")
    # Drop unnecessary columns
    new_dataset = hybrid_dataset.drop(columns=["Customer Name","Customer Email","Customer Age","Customer Gender"], errors='ignore')
    
    # Encoders
    le_dept = LabelEncoder()
    le_priority = LabelEncoder()
    
    new_dataset['Department_Label'] = le_dept.fit_transform(new_dataset['Department'])
    new_dataset['Priority_Label'] = le_priority.fit_transform(new_dataset['Ticket Priority'])
    
    # Regression Target
    # Notebook assumes Avg_Resolution_Time as target, effectively trying to predict the employee's avg time? 
    # Or is it predicting the time the ticket *will* take? 
    # In the notebook: new_dataset['Resolution_Time'] = new_dataset['Avg_Resolution_Time']
    # This seems to be predicting the Assigned Employee's historical avg resolution time using the ticket description?
    # It's a bit circular if employee assumes traits, but we will follow the notebook logic exactly.
    # FEATURES
    # To predict 'Resolution_Time' (Time) and 'Customer_Rating_Avg' (Rating)
    # The input to these models will be the Ticket Description + Employee Features.
    # Note: Customer_Rating_Avg in the dataset is likely the rating that specific ticket got.
    # While 'Customer_Rating_Avg' in employee data (if it exists) is their historical avg.
    # The prompt implies we want to predict the rating *this* employee will get for *this* ticket.
    # So we treat 'Customer_Rating_Avg' (the target) as the outcome of the interaction.
    
    # Let's check what we have in columns.
    # 'Customer_Rating_Avg' seems to be the column mapped from 'Customer_Rating_Avg_x' or 'y'.
    # In the synthetic generation, it was an employee attribute. 
    # But in the merged dataset, for each ticket row, it's the attribute of the assigned employee.
    # Wait, if we use the employee's *fixed* average rating as the target, the model will just learn to output that employee's average.
    # That is technically "predicting the rating", but simplistic. 
    # However, given the synthetic nature, that's likely what the user expects: "If I assign Employee A, what is the expected rating?" -> "Their average rating".
    # Or does the dataset have a per-ticket rating?
    # View file showed: "Customer Satisfaction Rating" (col 17 in line 1).
    # Let's use 'Customer Satisfaction Rating' as the target for the *prediction*.
    # And we use "Experience_Years", "Avg_Resolution_Time" etc as features of the employee.
    
    # 3. Preprocessing
    print("Preprocessing...")
    
    # Extract Employee Registry (Unique employees) for the App to use later
    # We need unique stats per Employee ID to simulate "what if we assigned this employee?"
    print("Extracting employee registry...")
    employee_cols = ['Assigned_Employee_ID', 'Department', 'Experience_Years', 'Avg_Resolution_Time', 'Tickets_Handled', 'Customer_Rating_Avg'] 
    # Note: 'Customer_Rating_Avg' here is the employee's historical average (feature), distinct from the specific ticket rating (target).
    # Let's verify if 'Customer_Rating_Avg' exists in the mapped dataframe.
    # Based on previous replace, we mapped _x columns to these names.
    
    # Drop duplicates to get unique employees
    employee_registry = hybrid_dataset[employee_cols].drop_duplicates(subset=['Assigned_Employee_ID']).copy()
    employee_registry.rename(columns={'Assigned_Employee_ID': 'Employee_ID'}, inplace=True)
    print(f"Employee Registry: Found {len(employee_registry)} unique employees.")
    
    # ... Continue with Standard Preprocessing ...
    
    # Encoders
    le_dept = LabelEncoder()
    le_priority = LabelEncoder()
    
    hybrid_dataset['Department_Label'] = le_dept.fit_transform(hybrid_dataset['Department'])
    hybrid_dataset['Priority_Label'] = le_priority.fit_transform(hybrid_dataset['Ticket Priority'])
    
    # Targets
    # 1. Classification: Dept, Priority
    y_class = hybrid_dataset[['Department_Label','Priority_Label']]
    
    # 2. Regression: Resolution Time
    # "Avg_Resolution_Time" is an employee trait. "Time to Resolution" (if in dataset) would be better per ticket.
    # Line 1 of CSV showed "Time to Resolution". Let's use that if available, else fallback.
    if 'Time to Resolution' in hybrid_dataset.columns:
         # Some might be null/str, handle cleanup? 
         # View file showed "2023-06-01...", wait that's a date? No "Time to Resolution" might be delta.
         # Actually Line 1: "...Ticket Priority,Ticket Channel,First Response Time,Time to Resolution,Customer Satisfaction Rating..."
         # Line 8: "...Social media,2023-06-01 12:15:36,,," -> It's empty in this row!
         # The notebook used 'Avg_Resolution_Time' (the employee trait) as the target 'Resolution_Time'. 
         # We will STICK TO THE NOTEBOOK LOGIC to match user expectations of "notebook deployment".
         y_time = hybrid_dataset['Avg_Resolution_Time'].values
    else:
         y_time = hybrid_dataset['Avg_Resolution_Time'].values

    # 3. Regression: Rating
    # Similarly, use 'Customer Satisfaction Rating' (actual ticket rating) if available/clean, 
    # OR use 'Customer_Rating_Avg' (employee trait) if the notebook did that.
    # The notebook didn't predict rating.
    # User asked for it now. 
    # If we predict 'Customer_Rating_Avg' (the employee trait), it's trivial (just look up the employee).
    # If we predict 'Customer Satisfaction Rating', it depends on the ticket.
    # Let's try to use 'Customer Satisfaction Rating' as target. If it's mostly empty (as seen in line 8), we fallback to 'Customer_Rating_Avg'.
    
    target_rating_col = 'Customer Satisfaction Rating'
    if target_rating_col in hybrid_dataset.columns and hybrid_dataset[target_rating_col].notna().sum() > 100:
        # Fill NA with mean or drop? Let's drop rows where target is NaN for training
        print(f"Using '{target_rating_col}' as rating target.")
        # We need to align X and y. This makes splitting complex if we drop rows.
        # Simpler approach for this demo: Fill NA with Employee's Avg Rating.
        hybrid_dataset[target_rating_col] = hybrid_dataset[target_rating_col].fillna(hybrid_dataset['Customer_Rating_Avg'])
        y_rating = hybrid_dataset[target_rating_col].values
    else:
        print("Using employee 'Customer_Rating_Avg' as rating target (proxy).")
        y_rating = hybrid_dataset['Customer_Rating_Avg'].values

    # Features
    text_feature = hybrid_dataset['Ticket Description']
    numerical_features = hybrid_dataset[['Experience_Years','Avg_Resolution_Time','Tickets_Handled','Customer_Rating_Avg']]
    
    # Categorical: Employee ID (OHE)
    # NOTE: For the "Best Employee" prediction, we will need to simulate inputting DIFFERENT Employee IDs for the SAME text.
    # So the model MUST depend on Employee ID (or its attributes).
    categorical_features_df = pd.get_dummies(hybrid_dataset['Assigned_Employee_ID'], prefix='Emp')
    feature_columns_cat = categorical_features_df.columns.tolist()

    X_text = text_feature
    X_num = numerical_features.values
    X_cat = categorical_features_df.values

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_text_vect = tfidf.fit_transform(X_text)

    # Scaling
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Combine
    X_full = hstack([X_text_vect, X_num_scaled, X_cat])

    # 4. Train Models
    print("Training models (n_estimators=10 for speed)...")
    # Split
    X_train, X_test, y_train_class, y_test_class, y_train_time, y_test_time, y_train_rating, y_test_rating = train_test_split(
        X_full, y_class, y_time, y_rating, test_size=0.2, random_state=42
    )

    # Classifier (Dept, Priority)
    clf = RandomForestClassifier(
        n_estimators=10, max_depth=10, min_samples_split=10, 
        min_samples_leaf=5, max_features='sqrt', class_weight='balanced', random_state=42
    )
    model_class = MultiOutputClassifier(clf)
    model_class.fit(X_train, y_train_class)

    # Regressor (Time)
    model_time = RandomForestRegressor(n_estimators=10, random_state=42)
    model_time.fit(X_train, y_train_time)
    
    # Regressor (Rating) - NEW
    model_rating = RandomForestRegressor(n_estimators=10, random_state=42)
    model_rating.fit(X_train, y_train_rating)

    print("Training complete.")

    # 5. Save Artifacts
    if not os.path.exists("models"):
        os.makedirs("models")

    print("Saving artifacts...")
    joblib.dump(model_class, 'models/model_class.joblib')
    joblib.dump(model_time, 'models/model_time.joblib')
    joblib.dump(model_rating, 'models/model_rating.joblib') # NEW
    joblib.dump(tfidf, 'models/tfidf.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(le_dept, 'models/le_dept.joblib')
    joblib.dump(le_priority, 'models/le_priority.joblib')
    joblib.dump(feature_columns_cat, 'models/feature_columns_cat.joblib')
    
    # Save Registry
    employee_registry.to_csv('models/employee_registry.csv', index=False)
    
    print("Done! Models and registry saved.")

if __name__ == "__main__":
    train_and_save()
