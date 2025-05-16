import joblib
import streamlit as st
from utils import predict_early_risk, predict_genetic, predict_clinical
import math
import os

# --------------------------
# Page Config
# --------------------------

st.set_page_config(page_title="Gastric Cancer Detection", layout="centered", page_icon=":hospital:")

# --------------------------
# Session State Navigation
# --------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()

# --------------------------
# Sidebar Navigation
# --------------------------

st.sidebar.title("Navigation")
if st.sidebar.button("🏠 Home"):
    go_to("Home")
if st.sidebar.button("📑 Model Selection"):
    go_to("Model Selection")
st.sidebar.markdown("""---""")
st.sidebar.markdown("### 👨‍💻 Team Members")
st.sidebar.markdown("""
* Youssef Kotb
* Ahmed Emad
* Sara Yasser
* Basma Khalil
* Shahd Mohamed
""")

# --------------------------
# HOME PAGE
# --------------------------

if st.session_state.page == "Home":
    st.title("Gastric Cancer Detection System")
    st.subheader("Early Detection. Smarter Decisions. Better Outcomes.")
    st.markdown("""
    Our AI-powered system leverages clinical data to support medical professionals in identifying gastric cancer risk with improved accuracy and speed.
    """)

    st.markdown("### Why Early Detection Matters")
    st.markdown("""
    Gastric cancer remains one of the leading causes of cancer-related deaths worldwide. Early detection significantly improves survival rates, but current screening methods are:
    
    * **Expensive** - Many patients cannot afford comprehensive screening
    * **Not widely available** - Especially in low-resource settings
    * **Invasive** - Current methods often require uncomfortable procedures
    """)

    st.markdown("""Click below to begin:""")
    if st.button("🚀 Try it Now", type="primary"):
        go_to("Model Selection")
    st.info("ℹ️ This tool is for educational purposes only and should not replace professional medical advice.")

    st.markdown("""---""")
    st.markdown("""## About the Project""")
    st.markdown("""
    This project introduces a machine learning-based model trained on clinical and lifestyle features to assess the risk of gastric cancer in patients. By analyzing non-invasive indicators, the model aims to assist healthcare providers in making quicker, data-driven decisions.
    
    **Key Features:**
    - Three-stage risk assessment pipeline
    - Trained on real-world, anonymized data
    - Uses robust algorithms like AdaBoost and Random Forests
    - Designed for both clinical and research use
    
    This system is part of our **graduation project** for the **Data Science and Machine Learning course** under the [Digital Egypt Pioneers Initiative (DEPI)](https://depi.gov.eg/).
    """)

    st.markdown("""---""")
    st.markdown("### How It Works")
    st.markdown("""
    We use three models at different stages of assessment:
    
    **1. Early Screening Model**  
    Uses lifestyle factors like age, gender, diet, and family history to provide initial risk assessment.
    
    **2. Genetic Tests Model**  
    Analyzes genetic markers to confirm cancer risk at the molecular level with higher precision.
    
    **3. Clinical Tests Model**  
    Uses lab data and imaging results for definitive risk evaluation in clinical settings.
    """)

    #for Troubleshooting
    st.write("Current working directory:", os.getcwd())
    # /mount/src/gastric-cancer-detection-system

    #img_path = os.path.join(os.path.dirname(__file__), "/images/flow.png")
    img_path = 'deployment/images/flow.png'
    st.write(img_path)

    st.image(img_path, use_container_width=True)
    st.caption("Figure 1: Our three-stage assessment workflow")

    st.markdown("""---""")
    st.markdown("""## Who Can Benefit From This System?""")
    st.markdown("""
    * **Healthcare Providers**: For preliminary patient screening and triage
    * **Researchers**: Analyzing patterns or testing clinical hypotheses
    * **Public Health Officials**: Population-level risk assessment
    * **Individuals**: Understanding personal risk factors
    """)

    st.markdown("""---""")
    st.markdown("""## Important Disclaimer""")
    st.markdown("""
    :warning: **This tool is for educational and research purposes only.**  
    It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider before making health-related decisions.
    """)

    st.markdown("""---""")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Ready to assess your risk?")
    with col2:
        if st.button("🧪 Let's Get Started", type="primary"):
            go_to("Model Selection")

    # Footer
    st.markdown("""---""")
    st.markdown("""
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Made with ❤️ by the DEPI Data Science team<br>
        Special thanks to our instructor <strong>Mahmoud Bustami</strong> and the <strong>Ministry of Communications and Information Technology (MCIT)</strong>, <strong>Eyouth</strong>, with <strong>Digital Egypt Pioneers Initiative (DEPI)</strong> for their incredible support.<br><br>
        🔗 <a href="https://depi.gov.eg/" target="_blank">DEPI</a> |
        <a href="https://mcit.gov.eg/" target="_blank">MCIT</a> |
        <a href="https://eyouth.com/" target="_blank">Eyouth</a> |
        <a href="https://github.com/your-project-link" target="_blank">GitHub</a> |
        <a href="https://your-linkedin.com" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

    left_col, center_col, right_col = st.columns([1, 2, 1])
    with left_col:
        img_path = os.path.join(os.path.dirname(__file__), "deployment/images/DEPI logo.webp")
        st.image(img_path, use_container_width=True)
    with right_col:
        img_path = os.path.join(os.path.dirname(__file__), "deployment/images/MCIT.webp")
        st.image(img_path, use_container_width=True)

# --------------------------
# MODEL SELECTION PAGE
# --------------------------

elif st.session_state.page == "Model Selection":
    st.title("Choose a Screening Model")
    st.markdown("""
    ### 🧭 Model Selection Guide
    
    Select one of our three assessment models below. Each is designed for different stages of risk evaluation:
    
    - **🩺 Early Screening Model**  
    Best for initial assessment using basic health and lifestyle factors
    
    - **🧬 Genetic Tests Model**  
    For patients with available genetic testing results
    
    - **🧫 Clinical Tests Model**  
    For patients with existing clinical test results
    
    > 💡 **Professional Tip:** Begin with the Early Screening Model unless you have specific test results available.
    """)

    st.markdown("""---""")
    st.markdown("### Select Your Assessment Model")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        
        img_path = os.path.join(os.path.dirname(__file__), "deployment/images/healthy.webp")
        st.image(img_path, use_container_width=True)

        st.markdown("#### 🩺 Early Screening")
        st.markdown("""
        * Age, gender, ethnicity
        * Lifestyle factors
        * Family history
        """)
        if st.button("Select Early Screening", key="early_btn"):
            go_to("Early Screening")

    with col2:

        img_path = os.path.join(os.path.dirname(__file__), "deployment/images/genetic.webp")
        st.image(img_path, use_container_width=True)

        st.markdown("#### 🧬 Genetic Tests")
        st.markdown("""
        * miRNA accession
        * Target gene symbols
        * Reaction algorithm data
        """)
        if st.button("Select Genetic Tests", key="genetic_btn"):
            go_to("Genetic Tests")

    with col3:

        img_path = os.path.join(os.path.dirname(__file__), "deployment/images/doctor.webp")
        st.image(img_path, use_container_width=True)

        st.markdown("#### 🧫 Clinical Tests")
        st.markdown("""
        * H. pylori status
        * Endoscopic images
        * Biopsy results
        * CT scan results
        """)
        if st.button("Select Clinical Tests", key="clinical_btn"):
            go_to("Clinical Tests")

# --------------------------
# EARLY SCREENING MODEL PAGE
# --------------------------

elif st.session_state.page == "Early Screening":
    st.title("🩺 Early Screening Assessment")
    st.markdown("""
    This initial screening evaluates your basic risk factors for gastric cancer.  
    Complete the form below to receive your preliminary risk assessment.
    """)

    with st.expander("ℹ️ About This Assessment"):
        st.markdown("""
        The Early Screening Model analyzes:
        - **Demographic factors**: Age, gender, ethnicity
        - **Lifestyle factors**: Smoking, alcohol, diet
        - **Medical history**: Family history, existing conditions
        - **Environmental factors**: Geographic location, H. pylori infection
        
        *Note: This is not a diagnostic tool but rather a preliminary screening.*
        """)

    with st.form("early_screening_form"):
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 20, 120, help="Enter your current age")
            gender = st.selectbox("Gender", ["Male", "Female"])
            ethnicity = st.selectbox("Ethnicity", ["Ethnicity_A", "Ethnicity_B", "Ethnicity_C"])
        with col2:
            geographical_location = st.selectbox("Geographical Location", ["California", "Other"])
            family_history = st.selectbox("Family History of Gastric Cancer?", ["Yes", "No"])
        
        st.markdown("### Lifestyle Factors")
        col1, col2 = st.columns(2)
        with col1:
            smoking_habits = st.selectbox("Do you smoke?", ["Yes", "No"])
            alcohol_consumption = st.selectbox("Do you drink alcohol?", ["Yes", "No"])
        with col2:
            dietary_habits = st.selectbox("Dietary Habits", ["Low_Salt", "High_Salt"])
            helicobacter_pylori_infection = st.selectbox("Helicobacter Pylori Infection?", ["Yes", "No"])
        
        st.markdown("### Medical History")
        existing_conditions = st.selectbox("Existing Conditions?", ["No_condition", "Diabetes", "Chronic Gastritis"])
        
        submitted = st.form_submit_button("🔍 Assess My Risk", type="primary")
    
    if submitted:
        # Convert inputs to model format
        gender = 1 if gender == "Male" else 0
        geographical_location = 1 if geographical_location == "Other" else 0
        family_history = 1 if family_history == "Yes" else 0
        smoking_habits = 1 if smoking_habits == "Yes" else 0
        alcohol_consumption = 1 if alcohol_consumption == "Yes" else 0
        helicobacter_pylori_infection = 1 if helicobacter_pylori_infection == "Yes" else 0
        dietary_habits = 1 if dietary_habits == "Low_Salt" else 0

        input_data = {
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "geographical_location": geographical_location,
            "family_history": family_history,
            "smoking_habits": smoking_habits,
            "alcohol_consumption": alcohol_consumption,
            "helicobacter_pylori_infection": helicobacter_pylori_infection,
            "dietary_habits": dietary_habits,
            "existing_conditions": existing_conditions
        }

        result = predict_early_risk(input_data)
        
        if result:  # High risk
            st.error("""
            ## 🔴 High Risk Detected
            
            Our assessment indicates you may be at elevated risk for gastric cancer based on the factors provided.
            """)
            
            st.markdown("""
            ### Recommended Actions:
            1. **Consult a gastroenterologist** for professional evaluation
            2. **Consider clinical testing** for definitive assessment
            3. **Monitor symptoms** like persistent indigestion or abdominal pain
            """)
            
            if st.button("➡️ Proceed to Clinical Tests", type="primary"):
                go_to("Clinical Tests")
            
            st.markdown("""
            ### Risk Factors Identified:
            - **Modifiable Factors**:  
              {}{}{}
            - **Non-Modifiable Factors**:  
              {}
            """.format(
                "• Smoking\n" if smoking_habits else "",
                "• Alcohol consumption\n" if alcohol_consumption else "",
                "• Dietary habits\n" if dietary_habits else "",
                "• Family history\n" if family_history else "• Age"
            ))
            
        else:  # Low risk
            st.success("""
            ## 🟢 Low Risk Identified
            
            Our assessment indicates you currently have low risk factors for gastric cancer.
            """)
            
            st.markdown("""
            ### Maintenance Recommendations:
            1. **Continue healthy habits**: Maintain balanced diet and exercise
            2. **Consider genetic testing** for deeper insights (especially if family history exists)
            3. **Regular checkups**: Annual physical exams are recommended
            """)
            
            if st.button("➡️ Explore Genetic Testing", type="primary"):
                go_to("Genetic Tests")
            
            st.markdown("""
            ### Preventive Measures:
            - Reduce processed food consumption
            - Increase vegetable and fruit intake
            - Regular exercise (150 mins/week)
            - Limit alcohol consumption
            """)

# --------------------------
# GENETIC TESTS MODEL PAGE
# --------------------------

elif st.session_state.page == "Genetic Tests":
    st.title("🧬 Genetic Risk Assessment")
    st.markdown("""
    This advanced analysis evaluates genetic markers associated with gastric cancer risk.  
    Please provide available genetic testing results below.
    """)

    with st.expander("ℹ️ About Genetic Testing"):
        st.markdown("""
        Genetic testing examines specific biomarkers that may indicate:
        - Increased susceptibility to gastric cancer
        - Molecular pathways affected
        - Potential response to treatments
        
        *Note: Genetic risk factors don't guarantee cancer development but indicate elevated risk.*
        """)

    with st.form("genetic_test_form"):
        st.markdown("### MicroRNA and Gene Data")
        col1, col2 = st.columns(2)
        with col1:
            mature_mirna_acc = st.selectbox("miRNA Accession", ["MIR123", "MIR234", "MIR345"])
        with col2:
            target_symbol = st.selectbox("Target Gene Symbol", ["CDH1", "KRAS", "TP53"])

        st.markdown("### Reaction Algorithm Data")
        col1, col2 = st.columns(2)
        with col1:
            diana_microt = st.number_input("DIANA-microT", format="%.2f")
            elmmo = st.number_input("ElMMo", format="%.2f")
            microcosm = st.number_input("MicroCosm", format="%.2f")
            miranda = st.number_input("miRanda", format="%.2f")
        with col2:
            mirdb = st.number_input("miRDB", format="%.2f")
            pictar = st.number_input("PicTar", format="%.2f")
            pita = st.number_input("PITA", format="%.2f")
            targetscan = st.number_input("TargetScan", format="%.2f")

        submitted = st.form_submit_button("🧪 Analyze Genetic Risk", type="primary")

    if submitted:
        input_data = {
            "diana_microt": diana_microt,
            "elmmo": elmmo,
            "microcosm": microcosm,
            "miranda": miranda,
            "mirdb": mirdb,
            "pictar": pictar,
            "pita": pita,
            "targetscan": targetscan,
            "mature_mirna_acc": mature_mirna_acc,
            "target_symbol": target_symbol
        }

        result = predict_genetic(input_data)
        
        if result:  # High risk
            st.error("""
            ## 🔴 Elevated Genetic Risk Detected
            
            Your genetic profile shows markers associated with increased gastric cancer risk.
            """)
            
            st.markdown("""
            ### Recommended Actions:
            1. **Consult a genetic counselor** for detailed interpretation
            2. **Enhanced screening** may be recommended
            3. **Discuss preventive options** with your healthcare provider
            """)
            
            if st.button("➡️ Proceed to Clinical Evaluation", type="primary"):
                go_to("Clinical Tests")
            
            st.markdown("""
            ### About Your Genetic Markers:
            - **{}** miRNA: Associated with {} pathways
            - **{}** gene: Plays role in {}
            """.format(
                mature_mirna_acc,
                "cell proliferation" if mature_mirna_acc == "MIR123" else "apoptosis regulation",
                target_symbol,
                "cell adhesion" if target_symbol == "CDH1" else "tumor suppression"
            ))
            
        else:  # Low risk
            st.success("""
            ## 🟢 No Significant Genetic Risk Identified
            
            Your genetic markers don't indicate elevated risk for gastric cancer.
            """)
            
            st.markdown("""
            ### Maintenance Recommendations:
            1. **Continue regular health screenings**
            2. **Monitor family history** for changes
            3. **Maintain healthy lifestyle** to minimize other risk factors
            """)
            
            st.markdown("""
            ### Genetic Insights:
            - The analyzed miRNA-gene interactions show normal patterns
            - No pathogenic variants detected in the assessed markers
            - Continued monitoring recommended as genetic research evolves
            """)

# --------------------------
# CLINICAL TESTS MODEL PAGE
# --------------------------

elif st.session_state.page == "Clinical Tests":
    st.title("🧫 Clinical Evaluation")
    st.markdown("""
    This definitive assessment evaluates clinical test results for gastric cancer detection.  
    Please provide available medical test results below.
    """)

    with st.expander("ℹ️ About Clinical Testing"):
        st.markdown("""
        Clinical tests provide the most accurate risk assessment by examining:
        - **H. pylori infection status**: Major risk factor
        - **Endoscopic findings**: Visual examination of stomach lining
        - **Biopsy results**: Tissue analysis
        - **Imaging results**: CT scan findings
        
        *Note: These results should be interpreted by a qualified medical professional.*
        """)

    with st.form("clinical_test_form"):
        st.markdown("### Clinical Test Results")
        col1, col2 = st.columns(2)
        with col1:
            helicobacter_pylori_infection = st.selectbox("Helicobacter pylori infection", ["Yes", "No"])
            endoscopic_images = st.selectbox("Endoscopic images result", ["Positive", "Negative"])
        with col2:
            biopsy_results = st.selectbox("Biopsy results", ["Positive", "Negative"])
            ct_scan = st.selectbox("CT scan results", ["Positive", "Negative"])

        submitted = st.form_submit_button("🏥 Evaluate Clinical Results", type="primary")

    if submitted:
        helicobacter_pylori_infection = 1 if helicobacter_pylori_infection == "Yes" else 0
        input_data = {
            "helicobacter_pylori_infection": helicobacter_pylori_infection,
            "endoscopic_images": endoscopic_images,
            "biopsy_results": biopsy_results,
            "ct_scan": ct_scan
        }

        result = predict_clinical(input_data)
        
        if result:  # High risk
            st.error("""
            ## 🔴 Clinical Findings Suggest Gastric Cancer Risk
            
            Your test results indicate possible gastric abnormalities requiring attention.
            """)
            
            st.markdown("""
            ### Urgent Next Steps:
            1. **Immediate consultation** with gastroenterologist
            2. **Possible follow-up testing** may be needed
            3. **Treatment planning** if malignancy confirmed
            """)
            
            st.markdown("""
            ### Clinical Insights:
            - **H. pylori status**: {}
            - **Endoscopic findings**: {}
            - **Biopsy results**: {}
            - **Imaging results**: {}
            """.format(
                "Positive (Major risk factor)" if helicobacter_pylori_infection else "Negative",
                endoscopic_images,
                biopsy_results,
                ct_scan
            ))
            
        else:  # Low risk
            st.success("""
            ## 🟢 No Clinical Evidence of Gastric Cancer
            
            Your test results don't indicate current gastric cancer risk.
            """)
            
            st.markdown("""
            ### Recommended Follow-up:
            1. **Routine monitoring** as advised by your physician
            2. **Address any H. pylori infection** if present
            3. **Maintain preventive care** regimen
            """)
            
            st.markdown("""
            ### Clinical Summary:
            - All tests within normal ranges
            - No malignant findings detected
            - Continue regular health maintenance
            """)