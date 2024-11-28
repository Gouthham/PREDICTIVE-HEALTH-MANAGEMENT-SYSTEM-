import streamlit as st
import joblib
import pandas as pd

# Add custom background color and styling
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"]{
    background-color: #273746;
    }

    [data-testid="stHeader"] {
        background-color: #273746; /* Match app background */
        padding: 0; /* Remove padding for better alignment */
    }


    [data-testid="stMainMenu"] {
    background-color: #7f8c8d;
    border-radius: 8px;
    color: #000000;
    }
    .stButton>button {
        background-color:#7f8c8d ;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #515a5a  ;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fdfefe;
    }
      p, label, .stMarkdown {
        color: #f8f9f9; /* Light gray for normal text */
    }

    .stMultiSelect div[data-baseweb="select"] {
        background-color: #ffffff; /* White background for multiselect dropdown */
        color: #000000; /* Black text for dropdown options */
    }

    .stMultiSelect div[data-baseweb="select"]:hover {
        background-color: #f0f0f0; /* Light gray hover for dropdown */
    }

    .subheader {
        color: #fdfefe; /* White color for subheaders */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model
model = joblib.load('trained_random_forest_model.joblib')

# Symptoms list
symptoms=['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_ofurine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis']


# Dropdown for selecting symptoms
st.title("Disease Prediction and Care Assistant")
st.write("Select the symptoms you are experiencing to predict the possible disease and get related recommendations.")
selected_symptoms = st.multiselect("Select Symptoms", symptoms)

# Prepare user input
user_input = {symptom: 1 if symptom in selected_symptoms else 0 for symptom in symptoms}
input_df = pd.DataFrame([user_input]).reindex(columns=model.feature_names_in_, fill_value=0)

# Load metadata files
description = pd.read_csv('1. DESCRIPTION.csv')
medication = pd.read_csv('3. MEDICATION.csv')
precaution = pd.read_csv('4. Numbered_Precautions.csv')
diet = pd.read_csv('5. DIET.csv')

# Load the label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Prediction and output
# Prediction and output
if st.button("Predict Disease"):
    if not selected_symptoms:  # Check if no symptoms are selected
        st.warning("Please select at least one symptom for prediction.")  # Display warning message
    else:
        # Make prediction
        predicted_disease_encoded = model.predict(input_df)
        disease = label_encoder.inverse_transform(predicted_disease_encoded)[0]

        # Display the predicted disease
        st.subheader(f"Predicted Disease: {disease}")

        # Description
        description_info = description[description['DISEASES'] == disease]['DESCRIPTION'].values
        if description_info.size > 0:
            st.write(f"**Description:** {description_info[0]}")

        # Medication
        medication_info = medication[medication['DISEASES'] == disease]['MEDICATION'].values
        if medication_info.size > 0:
            st.write(f"**Recommended Medication:** {medication_info[0]}")

        # Precaution
        precaution_info = precaution[precaution['DISEASES'] == disease]['PRECAUTION'].values
        if precaution_info.size > 0:
            st.write(f"**Precautions:** {precaution_info[0]}")

        # Diet
        diet_info = diet[diet['DISEASES'] == disease]['DIET'].values
        if diet_info.size > 0:
            st.write(f"**Recommended Diet:** {diet_info[0]}")
