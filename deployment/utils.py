import pandas as pd
import joblib
import numpy as np

pd.set_option('display.max_columns', None)


def predict_early_risk(raw_input):
    # Load encoders and model
    model = joblib.load("models/Early screening/Early Risk Screening.pkl")
    ohe = joblib.load("models/Early screening/ES_ohe_Encoder.pkl")
    scaler = joblib.load('models/Early screening/ES_minmaxscaler.pkl')

    # Define column groups
    ES_data_num_cols = ['age']
    ES_data_multi_cat_cols = ['ethnicity', 'existing_conditions']
    ES_data_bin_cat_cols = ['gender','geographical_location', 'dietary_habits', 'family_history', 'smoking_habits', 'alcohol_consumption', 'helicobacter_pylori_infection']
    
    # Convert input to DataFrame
    raw_input = pd.DataFrame([raw_input])
    print(raw_input)

    # One-hot encode multi-categorical features
    ohe_encoded = ohe.transform(raw_input[ES_data_multi_cat_cols])

    raw_input['age'] = scaler.transform(raw_input[['age']])

    # Combine numeric, binary, and ohe-encoded features
    final_input = np.concatenate([
        raw_input[ES_data_num_cols + ES_data_bin_cat_cols].values,
        ohe_encoded
    ], axis=1)

    print(final_input)

    # Make prediction
    prediction = model.predict(final_input)[0]
    return prediction


def predict_genetic(raw_input):

    # Load encoders and model
    model = joblib.load("models/Genetic data/Genetic.pkl")
    ohe = joblib.load("models/Genetic data/Genetic_ohe_encoder.pkl")
    scaler = joblib.load('models/Genetic data/Genetic_scaler.pkl')

    G_data_multi_cat_cols = ['mature_mirna_acc', 'target_symbol']

    raw_input = pd.DataFrame([raw_input])

    # Fit and transform the data
    encoded_array = ohe.transform(raw_input[G_data_multi_cat_cols])

    # Convert to DataFrame with proper column names
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=ohe.get_feature_names_out(G_data_multi_cat_cols),
        index=raw_input.index  # preserve original index
    )

    # Drop original multi-cat columns and concatenate the encoded ones
    raw_input = raw_input.drop(columns=G_data_multi_cat_cols)
    raw_input = pd.concat([raw_input, encoded_df], axis=1)

    tool_cols = ['diana_microt','elmmo','microcosm','miranda','mirdb','pictar','pita','targetscan']

    raw_input['mean_score'] = raw_input[tool_cols].mean(axis=1)
    raw_input['std_score'] = raw_input[tool_cols].std(axis=1)
    raw_input['max_score'] = raw_input[tool_cols].max(axis=1)
    raw_input['min_score'] = raw_input[tool_cols].min(axis=1)
    raw_input['consensus_votes'] = (raw_input[tool_cols] > 0.7).sum(axis = 1)


    print(raw_input)

    input_scaled = scaler.transform(raw_input)


    #Make prediction

    threshold = 0.045

    prediction = model.predict_proba(input_scaled)[:,1]
    
    y_val_custom_pred = (prediction >= threshold).astype(int)

    return y_val_custom_pred


def predict_clinical(raw_input):

    # Load encoders and model
    model = joblib.load("models/Clinical tests/Clinical tests.pkl")
    le = joblib.load("models/Clinical tests/CT_le_encoder.pkl")

    raw_input = pd.DataFrame([raw_input])

    for col in ['endoscopic_images', 'biopsy_results', 'ct_scan']:
        raw_input[col] = le.transform(raw_input[col])

    print(raw_input)

    #Make prediction
    prediction = model.predict(raw_input)[0]
    return prediction