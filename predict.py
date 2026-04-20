import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

"""
Load the model before the normalization in order to be able
to accept the input from the form.
"""
model = tf.keras.models.load_model('./model/depression_model.h5')

raw_df = pd.read_csv('./dataset/dataset.csv')
raw_df = raw_df.drop(columns=['id', 'City', 'Work Pressure', 'Profession', 'Job Satisfaction', 'Depression'])

numerical_cols   = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 'Work/Study Hours', 'Financial Stress']
categorical_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Family History of Mental Illness', 'Have you ever had suicidal thoughts ?']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
])
preprocessor.fit(raw_df)

"""
Helper function to prevent typing errors.
"""
def ask_index(prompt, options):
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"    {i}. {option}")
    while True:
        try:
            index = int(input("  Enter number: ").strip())
            if 1 <= index <= len(options):
                return options[index - 1]
            else:
                print(f"  Invalid input. Please enter a number between 1 and {len(options)}.\n")
        except ValueError:
            print("  Please enter a valid number.\n")

"""
Helper function to prevent typing errors.
Only works with ranges
"""
def ask_float(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(f"\n{prompt} ({min_val}–{max_val}): ").strip())
            if not (min_val <= value <= max_val):
                print(f"  Please enter a value between {min_val} and {max_val}.\n")
            else:
                return value
        except ValueError:
            print("  Please enter a valid number.\n")

print("\n" + "="*55)
print("         STUDENT DEPRESSION SCREENING TOOL")
print("="*55)
print("Please answer honestly. All fields are required.\n")

print("── Demographics ──────────────────────────────────────")
gender = ask_index("  Asigned sex at birth:", ["Male", "Female"])

age = ask_float("  Age", 1, 100)

degree_options = [
    "Class 12",
    "BSc       (Bachelor of Science)",
    "B.Tech    (Bachelor of Technology)",
    "B.Com     (Bachelor of Commerce)",
    "BCA       (Bachelor of Computer Applications)",
    "BBA       (Bachelor of Business Administration)",
    "BA        (Bachelor of Arts)",
    "B.Ed      (Bachelor of Education)",
    "B.Pharm   (Bachelor of Pharmacy)",
    "BHM       (Bachelor of Hotel Management)",
    "BE        (Bachelor of Engineering)",
    "LLB       (Bachelor of Law)",
    "MSc       (Master of Science)",
    "M.Tech    (Master of Technology)",
    "M.Com     (Master of Commerce)",
    "MCA       (Master of Computer Applications)",
    "MBA       (Master of Business Administration)",
    "MA        (Master of Arts)",
    "M.Ed      (Master of Education)",
    "M.Pharm   (Master of Pharmacy)",
    "MHM       (Master of Hotel Management)",
    "ME        (Master of Engineering)",
    "LLM       (Master of Law)",
    "MBBS      (Bachelor of Medicine and Surgery)",
    "MD        (Doctor of Medicine)",
    "PhD       (Doctor of Philosophy)",
    "Others"
]
degree_raw = ask_index("  Degree:", degree_options)
degree = degree_raw.split()[0]

print("\n── Academic ──────────────────────────────────────────")
cgpa               = ask_float("  CGPA [Can be decimal]", 0, 10)
academic_pressure  = ask_float("  Academic Pressure", 0, 5)
study_satisfaction = ask_float("  Study Satisfaction", 0, 5)
work_study_hours   = ask_float("  Work/Study Hours per day", 0, 24)

print("\n── Lifestyle ─────────────────────────────────────────")
sleep_duration = ask_index("  Sleep Duration:", [
    "Less than 5 hours",
    "5-6 hours",
    "7-8 hours",
    "More than 8 hours"
])

dietary_habits = ask_index("  Dietary Habits:", [
    "Healthy",
    "Moderate",
    "Unhealthy"
])

financial_stress = ask_float("  Financial Stress", 0, 5)

print("\n── Mental Health History ─────────────────────────────")
family_history    = ask_index("  Family History of Mental Illness:", ["Yes", "No"])
suicidal_thoughts = ask_index("  Have you ever had suicidal thoughts?", ["Yes", "No"])

"""
Builder for the dataframe with the information from the form.
"""
input_data = pd.DataFrame([{
    'Gender':                                gender,
    'Age':                                   age,
    'Academic Pressure':                     academic_pressure,
    'CGPA':                                  cgpa,
    'Study Satisfaction':                    study_satisfaction,
    'Sleep Duration':                        sleep_duration,
    'Dietary Habits':                        dietary_habits,
    'Degree':                                degree,
    'Work/Study Hours':                      work_study_hours,
    'Financial Stress':                      financial_stress,
    'Family History of Mental Illness':      family_history,
    'Have you ever had suicidal thoughts ?': suicidal_thoughts
}])

input_processed = preprocessor.transform(input_data)
probability     = model.predict(input_processed, verbose=0)[0][0]
prediction      = int(probability > 0.35)

print("\n" + "="*55)
print("                     RESULT")
print("="*55)

"""
Generation of a progress bar
"""
bar_filled = int(probability * 30)
bar        = "█" * bar_filled + "░" * (30 - bar_filled)
print(f"\n  Probability: [{bar}] {probability*100:.1f}%\n")

if probability >= 0.65:
    print("     HIGH RISK | Strong signs of depression detected.")
    print("     Please reach out to a mental health professional")
    print("     or a trusted person for a mental evaluation as soon")
    print("     as possible.")
elif probability >= 0.35:
    print("     MODERATE RISK | Some signs of depression detected.")
    print("     Consider speaking with a counselor or doctor.")
else:
    print("     LOW RISK | No strong signs of depression detected.")
    print("     Keep maintaining healthy habits and routines.")

print("\n" + "="*55)
print("  DISCLAIMER")
print("  This is NOT a medical diagnosis.")
print("  This tool is for screening purposes only.")
print("  Always consult a qualified professional.")
print("="*55)