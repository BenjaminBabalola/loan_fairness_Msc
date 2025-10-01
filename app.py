import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# Set page config
st.set_page_config(
    page_title="FairLoan AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load all necessary artifacts
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        # Load models
        artifacts['baseline_model'] = joblib.load('models/xgboost_baseline.pkl')
        artifacts['fairness_pipeline'] = joblib.load('models/xgboost_fairness_pipeline.pkl')
        
        # Load preprocessing objects
        artifacts['scaler'] = joblib.load('models/scaler.pkl')
        artifacts['label_encoder'] = joblib.load('models/label_encoder.pkl')
        artifacts['feature_names'] = joblib.load('models/feature_names_clean.pkl')
        
        # Load SHAP explainer (pre-computed for fairness model)
        #artifacts['explainer'] = shap.TreeExplainer(artifacts['fairness_pipeline']['model'])
        
        # Debug: Check what features the scaler was trained on
        if hasattr(artifacts['scaler'], 'feature_names_in_'):
            artifacts['scaler_features'] = artifacts['scaler'].feature_names_in_
        else:
            # For older scikit-learn versions, we need to infer
            artifacts['scaler_features'] = ['duration', 'credit_amount', 'installment_commitment', 
                                          'residence_since', 'age', 'existing_credits', 'num_dependents']
        
        return artifacts
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

# Preprocess input data
def preprocess_input(input_dict, feature_names, scaler, scaler_features):
    # Create a DataFrame with all features initialized to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Map numerical features
    numerical_mapping = {
        'duration': 'duration',
        'credit_amount': 'credit_amount',
        'installment_commitment': 'installment_commitment',
        'residence_since': 'residence_since',
        'age': 'age',
        'existing_credits': 'existing_credits',
        'num_dependents': 'num_dependents'
    }
    
    for frontend_name, backend_name in numerical_mapping.items():
        if frontend_name in input_dict:
            input_df[backend_name] = input_dict[frontend_name]
    
    # Map categorical features (one-hot encoding) - using cleaned feature names
    categorical_mapping = {
        'checking_status': {
            '<0 DM': 'checking_status__0',                    # Map to checking_status__0
            '0 <= ... < 200 DM': 'checking_status__=200',     # Map to checking_status__=200  
            '>= 200 DM': 'checking_status__=200',             # Map to checking_status__=200
            'no checking account': 'checking_status_no checking'  # Map to checking_status_no checking
        },
        'credit_history': {
            'no credits/all paid': 'credit_history_no credits/all paid',
            'all paid': 'credit_history_all paid',
            'existing paid': 'credit_history_existing paid',
            'delayed previously': 'credit_history_delayed previously',
            'critical/other existing credit': 'credit_history_critical/other existing credit'
        },
        'purpose': {
            'new car': 'purpose_new car',
            'used car': 'purpose_used car',
            'furniture/equipment': 'purpose_furniture/equipment',
            'radio/tv': 'purpose_radio/tv',
            'domestic appliances': 'purpose_domestic appliances',
            'repairs': 'purpose_repairs',
            'education': 'purpose_education',
            'vacation': 'purpose_vacation',
            'retraining': 'purpose_retraining',
            'business': 'purpose_business',
            'other': 'purpose_other'
        },
        'savings_status': {
            '<100': 'savings_status__100',
            '100<=X<500': 'savings_status_100__X_500',
            '500<=X<1000': 'savings_status_500__X_1000',
            '>=1000': 'savings_status__=1000',
            'no known savings': 'savings_status_no known savings'
        },
        'employment': {
            'unemployed': 'employment_unemployed',
            '<1': 'employment__1',
            '1<=X<4': 'employment_1__X_4',
            '4<=X<7': 'employment_4__X_7',
            '>=7': 'employment__=7'
        },
        'personal_status': {
            'male div/sep': 'personal_status_male div/sep',
            'female div/dep/mar': 'personal_status_female div/dep/mar',
            'male single': 'personal_status_male single',
            'male mar/wid': 'personal_status_male mar/wid'
        },
        'other_parties': {
            'none': 'other_parties_none',
            'guarantor': 'other_parties_guarantor',
            'co-applicant': 'other_parties_co-applicant'
        },
        'property_magnitude': {
            'real estate': 'property_magnitude_real estate',
            'life insurance': 'property_magnitude_life insurance',
            'car': 'property_magnitude_car',
            'no known property': 'property_magnitude_no known property'
        },
        'other_payment_plans': {
            'bank': 'other_payment_plans_bank',
            'stores': 'other_payment_plans_stores',
            'none': 'other_payment_plans_none'
        },
        'housing': {
            'rent': 'housing_rent',
            'own': 'housing_own',
            'for free': 'housing_for free'
        },
        'job': {
            'unemp/unskilled non res': 'job_unemp/unskilled non res',
            'unskilled resident': 'job_unskilled resident',
            'skilled': 'job_skilled',
            'high qualif/self emp/mgmt': 'job_high qualif/self emp/mgmt'
        },
        'own_telephone': {
            'yes': 'own_telephone_yes',
            'none': 'own_telephone_none'
        },
        'foreign_worker': {
            'yes': 'foreign_worker_yes',
            'no': 'foreign_worker_no'
        }
    }
    
    # Set the appropriate categorical columns to 1
    for category, options in categorical_mapping.items():
        if category in input_dict:
            selected_option = input_dict[category]
            if selected_option in options:
                backend_column = options[selected_option]
                if backend_column in input_df.columns:
                    input_df[backend_column] = 1
                else:
                    st.warning(f"Feature {backend_column} not found in dataframe columns")
    
    # Handle age group
    age = input_dict.get('age', 30)
    if age <= 25:
        input_df['age_group_18-25'] = 1
    elif age <= 35:
        input_df['age_group_26-35'] = 1
    elif age <= 50:
        input_df['age_group_36-50'] = 1
    else:
        input_df['age_group_50+'] = 1
    
    # Scale numerical features using the exact features the scaler was trained on
    if scaler_features is not None and len(scaler_features) > 0:
        # Create a dataframe with only the features the scaler expects
        numerical_data = input_df[scaler_features].copy()
        
        # Scale the numerical data
        scaled_numerical = scaler.transform(numerical_data)
        
        # Update the input dataframe with scaled values
        input_df[scaler_features] = scaled_numerical
    
    return input_df

# Main app
def main():
    st.title("🏦 FairLoan AI: Bias-Aware Loan Approval System")
    st.markdown("""
    This demo showcases our fairness-optimized machine learning model that predicts loan approvals 
    while actively mitigating demographic bias using a three-stage fairness pipeline.
    """)
    
    # Load artifacts
    artifacts = load_artifacts()
    if artifacts is None:
        st.stop()
    
    # Debug information
    if st.sidebar.checkbox("Show debug info"):
        st.sidebar.write("Scaler features:", artifacts.get('scaler_features', 'Not available'))
        st.sidebar.write("Feature names:", artifacts.get('feature_names', 'Not available')[:10])
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Loan Decision", 
        "Performance Analysis", 
        "Fairness Insights", 
        "Technical Details"
    ])
    
    with tab1:
        st.header("🏠 Submit Loan Application")
        st.info("Our AI system uses fairness-aware machine learning to ensure equitable lending decisions.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Financial Information")
            # ADD CHECKING STATUS - THE MOST IMPORTANT FEATURE
            checking_status = st.selectbox("Checking Account Status*", [
                "<0 DM", "0 <= ... < 200 DM", ">= 200 DM", "no checking account"
            ], help="Current status of your checking account")
            
            duration = st.slider("Loan Duration (months)", 4, 72, 24)
            credit_amount = st.slider("Credit Amount (DM)", 250, 20000, 5000)
            installment_commitment = st.slider("Installment Rate (% of income)", 1, 4, 2)
            savings_status = st.selectbox("Savings Status", [
                "<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"
            ])
            employment = st.selectbox("Employment Duration (years)", [
                "unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"
            ])
        
        with col2:
            st.subheader("Personal Information")
            age = st.slider("Age", 18, 75, 35)
            personal_status = st.selectbox("Personal Status", [
                "male div/sep", "female div/dep/mar", "male single", "male mar/wid"
            ])
            housing = st.selectbox("Housing Status", ["rent", "own", "for free"])
            job = st.selectbox("Job Type", [
                "unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"
            ])
            foreign_worker = st.radio("Foreign Worker", ["yes", "no"])
        
        # Additional financial info
        with st.expander("Additional Financial Information"):
            col3, col4 = st.columns(2)
            with col3:
                credit_history = st.selectbox("Credit History", [
                    "no credits/all paid", "all paid", "existing paid", 
                    "delayed previously", "critical/other existing credit"
                ])
                purpose = st.selectbox("Loan Purpose", [
                    "new car", "used car", "furniture/equipment", "radio/tv",
                    "domestic appliances", "repairs", "education", "vacation",
                    "retraining", "business", "other"
                ])
            with col4:
                existing_credits = st.slider("Number of Existing Credits", 1, 4, 1)
                num_dependents = st.slider("Number of Dependents", 1, 2, 1)
                residence_since = st.slider("Years at Current Residence", 1, 4, 2)
        
        # Predict button
        if st.button("Get Fairness-Aware Decision", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'checking_status': checking_status,  # ADD THIS
                'duration': duration,
                'credit_amount': credit_amount,
                'installment_commitment': installment_commitment,
                'savings_status': savings_status,
                'employment': employment,
                'age': age,
                'personal_status': personal_status,
                'housing': housing,
                'job': job,
                'foreign_worker': foreign_worker,
                'credit_history': credit_history,
                'purpose': purpose,
                'existing_credits': existing_credits,
                'num_dependents': num_dependents,
                'residence_since': residence_since
            }
            
            # Preprocess input
            processed_input = preprocess_input(
                input_data, 
                artifacts['feature_names'], 
                artifacts['scaler'],
                artifacts.get('scaler_features', [])
            )
            
            # Make predictions
            try:
                # PRIMARY DECISION: Fairness pipeline prediction (THE ACTUAL DECISION)
                fair_pred = artifacts['fairness_pipeline']['model'].predict(processed_input.values)[0]
                fair_proba = artifacts['fairness_pipeline']['model'].predict_proba(processed_input.values)[0][1]
                
                # For research comparison only
                baseline_pred = artifacts['baseline_model'].predict(processed_input.values)[0]
                baseline_proba = artifacts['baseline_model'].predict_proba(processed_input.values)[0][1]
                
                # Display PRIMARY decision
                st.subheader("🎯 Fairness-Aware Loan Decision")
                
                # Main decision card
                if fair_pred == 1:
                    st.success(f"## ✅ APPROVED")
                    st.info(f"**Confidence:** {fair_proba:.2%}")
                else:
                    st.error(f"## ❌ REJECTED")
                    st.info(f"**Confidence:** {fair_proba:.2%}")
                
                st.caption("This decision uses our bias-mitigated AI model with fairness constraints")
                
                # Research comparison section
                with st.expander("🔍 Research Comparison: Baseline vs Fairness Model"):
                    st.info("""
                    **Understanding the comparison:**
                    - **Baseline Model**: Standard XGBoost without fairness interventions
                    - **Fairness-Aware Model**: Our optimized model with bias mitigation
                    """)
                    
                    col_compare1, col_compare2 = st.columns(2)
                    with col_compare1:
                        st.metric(
                            label="Baseline Model", 
                            value="APPROVED" if baseline_pred == 1 else "REJECTED",
                            delta=None,
                            help=f"Confidence: {baseline_proba:.2%}"
                        )
                    
                    with col_compare2:
                        decision_changed = fair_pred != baseline_pred
                        st.metric(
                            label="Fairness Impact",
                            value="Decision Changed" if decision_changed else "Decision Same",
                            delta="Fairer" if decision_changed else None,
                            help="Shows if fairness interventions affected the outcome"
                        )
                
                # SHAP explanation
                # SHAP explanation - USING PRE-COMPUTED IMAGES
                st.subheader("📊 Explanation of AI Decision")
                st.info("This transparency report shows how features influence our model's decisions")

                col1, col2 = st.columns(2)

                with col1:
                    try:
                        shap_summary_img = Image.open('results/figures/shap_summary.png')
                        st.image(shap_summary_img, caption="SHAP Summary Plot", use_column_width=True)
                        st.caption("Shows how each feature contributes to model predictions across all instances")
                    except:
                        st.warning("SHAP summary plot not available")

                with col2:
                    try:
                        shap_importance_img = Image.open('results/figures/shap_feature_importance.png')
                        st.image(shap_importance_img, caption="SHAP Feature Importance", use_column_width=True)
                        st.caption("Global feature importance based on mean absolute SHAP values")
                    except:
                        st.warning("SHAP feature importance plot not available")

                plt.close()

                
                
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                import traceback
                st.error("Detailed error information:")
                st.code(traceback.format_exc())
    
    with tab2:
        st.header("📈 Model Performance Analysis")
        
        # Load comparison results
        try:
            comparison_df = pd.read_csv('results/metrics/comparison_results.csv', index_col=0)
            
            st.subheader("Performance Metrics Comparison")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Baseline Accuracy", f"{comparison_df.loc['XGBoost', 'baseline_accuracy']:.3f}")
                st.metric("Baseline F1 Score", f"{comparison_df.loc['XGBoost', 'baseline_f1']:.3f}")
            
            with col2:
                st.metric("Fairness Model Accuracy", f"{comparison_df.loc['XGBoost', 'fairness_accuracy']:.3f}")
                st.metric("Fairness Model F1 Score", f"{comparison_df.loc['XGBoost', 'fairness_f1']:.3f}")
            
            with col3:
                accuracy_diff = comparison_df.loc['XGBoost', 'fairness_accuracy'] - comparison_df.loc['XGBoost', 'baseline_accuracy']
                f1_diff = comparison_df.loc['XGBoost', 'fairness_f1'] - comparison_df.loc['XGBoost', 'baseline_f1']
                st.metric("Accuracy Δ", f"{accuracy_diff:.3f}", 
                         delta="improved" if accuracy_diff > 0 else "decreased")
                st.metric("F1 Score Δ", f"{f1_diff:.3f}",
                         delta="improved" if f1_diff > 0 else "decreased")
            
            st.subheader("Fairness Metrics Improvement")
            st.info("""
            **Key Fairness Metrics:**
            - **Statistical Parity Difference (SPD)**: Measures demographic parity (closer to 0 = better)
            - **Equal Opportunity Difference (EOD)**: Measures equality of true positive rates (closer to 0 = better)
            """)
            
            spd_improvement = comparison_df.loc['XGBoost', 'personal_status_SPD_improvement']
            eod_improvement = comparison_df.loc['XGBoost', 'personal_status_EOD_improvement']
            
            col4, col5 = st.columns(2)
            with col4:
                st.metric("SPD Improvement", f"{abs(spd_improvement):.4f}", 
                         delta="More fair" if spd_improvement > 0 else "Less fair")
            with col5:
                st.metric("EOD Improvement", f"{abs(eod_improvement):.4f}",
                         delta="More fair" if eod_improvement > 0 else "Less fair")
            
            # Show comparison chart
            metrics_to_compare = ['accuracy', 'f1', 'personal_status_SPD', 'personal_status_EOD']
            comparison_data = []
            
            for metric in metrics_to_compare:
                comparison_data.append({
                    'Metric': metric,
                    'Baseline': comparison_df.loc['XGBoost', f'baseline_{metric}'],
                    'Fairness Model': comparison_df.loc['XGBoost', f'fairness_{metric}']
                })
            
            comparison_chart_df = pd.DataFrame(comparison_data)
            fig = px.bar(comparison_chart_df, x='Metric', y=['Baseline', 'Fairness Model'],
                        title="Model Comparison: Baseline vs Fairness-Aware",
                        barmode='group', labels={'value': 'Score', 'variable': 'Model Type'})
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error loading comparison data: {e}")
    
    with tab3:
        st.header("⚖️ Fairness Insights")
        
        st.subheader("Demographic Distribution Analysis")
        try:
            sensitive_img = Image.open('results/figures/sensitive_attributes_distribution.png')
            st.image(sensitive_img, caption="Distribution of Sensitive Attributes in Training Data")
            st.caption("Understanding the demographic composition of our training data helps ensure fair model development")
        except:
            st.warning("Demographic distribution visualization not available")
        
        st.subheader("Global Feature Importance")
        try:
            shap_img = Image.open('results/figures/shap_feature_importance.png')
            st.image(shap_img, caption="Overall Feature Importance from SHAP Analysis")
            st.caption("These features have the greatest impact on model decisions across all predictions")
        except:
            st.warning("Feature importance visualization not available")
        
        st.subheader("Bias Mitigation Results")
        try:
            comparison_img = Image.open('results/figures/baseline_vs_fairness_comparison.png')
            st.image(comparison_img, caption="Effectiveness of Fairness Interventions")
            st.caption("Comparison showing how our fairness pipeline improves model equity")
        except:
            st.warning("Bias mitigation visualization not available")
    
    with tab4:
        st.header("🔧 Technical Implementation")
        
        st.subheader("Three-Stage Fairness Pipeline")
        st.markdown("""
        Our fairness-aware pipeline implements a comprehensive approach:
        
        1. **🔄 Pre-processing**: Data-level interventions
           - Reweighting techniques to address class imbalance
           - Sampling methods to balance demographic representation
        
        2. **⚙️ In-processing**: Algorithmic fairness during training
           - Adversarial debiasing to learn fair representations
           - Fairness constraints integrated into XGBoost optimization
        
        3. **📊 Post-processing**: Output-level adjustments
           - Group-specific threshold optimization using GHOST algorithm
           - Calibration to ensure equitable outcomes across demographics
        """)
        
        st.subheader("Dataset Information")
        st.markdown("""
        - **Source**: German Credit Dataset (UCI Machine Learning Repository)
        - **Samples**: 1,000 historical loan applications
        - **Features**: 20+ attributes including financial, employment, and demographic information
        - **Target**: Binary classification (Good/Bad credit risk)
        - **Fairness Focus**: Personal status, age groups, and other protected attributes
        """)
        
        st.subheader("Model Architecture")
        st.markdown("""
        - **Base Algorithm**: XGBoost Classifier
        - **Fairness Technique**: Integrated three-stage bias mitigation pipeline
        - **Evaluation Metrics**: 
          - Performance: Accuracy, F1 Score, AUC-ROC
          - Fairness: SPD, EOD, Disparate Impact, Average Odds Difference
        - **Interpretability**: SHAP values for transparent decision explanation
        - **Compliance**: Designed to align with GDPR and fair lending regulations
        """)
        
        st.subheader("Research Contribution")
        st.markdown("""
        This work demonstrates that machine learning models can achieve both:
        - **High predictive accuracy** for credit risk assessment
        - **Strong fairness guarantees** across demographic groups
        
        By implementing a comprehensive fairness pipeline, we show that the perceived 
        trade-off between accuracy and fairness can be effectively managed.
        """)

if __name__ == "__main__":
    main()