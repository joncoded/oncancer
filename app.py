# =========================================================
# onCancer by @joncoded (aka @jonchius)
# cancer screening with machine learning 
# app.py 
#
# The data file contained a list of *cancer screening patients* with features related to their:
# 
# * health habits (e.g. smoking and alcohol)
# * living conditions (e.g. air pollution and occupation hazards)
# * other medical variables

# Our goals of this exercise:
# 
# * split data into training and testing data for use in multiple models
#   * training data will form the basis of the models
#   * testing data will see if the models
#     * can predict the `Overall_Risk_Score` (`y_pred`)
#     * ...by validating against the real `Overall_Risk_Score` (`y_test`)
#   * get accuracy level of models
#   * select the best model
# * use the best model predict `Overall_Risk_Score`
# * re-create the `Risk_Level` with that predicted score

# =========================================================
# IMPORTS
# =========================================================

# streamlit ui
import streamlit as st

# data processing
import pandas as pd
data_path = './data/cancer.xlsx'

page_title = "🎗️ onCancer"
page_icon = "🎗️"
app_tagline = "calculating cancer risk via sample data and machine learning"

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        *, h1, h2 { font-family: 'Barlow Semi Condensed' !important; }
        .stExpander > details > summary > span > span:first-child { display: none; }
        /* expander button fixes due to streamlit glitch */
        .stExpander > details > summary > span > div::before { 
            content: "▼"; display: inline-block; margin-right: 8px; 
        }          
        .stExpander > details[open] > summary > span > div::before { 
            content: "▲"; display: inline-block; margin-right: 8px; 
        }
    </style>
    """,
    unsafe_allow_html = True
)

# sticky header hack
def header(content):
    st.markdown(f"""
        <div style="position:fixed; top:60px; left:0; width:100%; background-color:#222; color:#fff; padding:5px; z-index:9999">
            <div style="display:flex; justify-content:center; align-items:center;">
                {content}
            </div>
        </div>""", unsafe_allow_html = True)

header(f"<h1 style=\"font-size:24px\">{page_title}</h1><div style=\"font-size:12px\">{app_tagline}</div>")

# padding hack
st.write("<br><br>", unsafe_allow_html = True)

# =========================================================
# DATA LOADING
# =========================================================

# load file onto data
data = pd.read_excel(data_path)

# =========================================================
# DATA COLLECTION
# =========================================================

st.markdown("# Objective")

st.write("Using a dataset of 497 records of anonymized cancer patients, we will try to predict an `Overall_Risk_Score` and hence, a `Risk_Level`, of a new patient using a machine learning model.")

st.write("This app is only to be used for educational purposes and not for real medical advice. Always consult a healthcare professional for medical concerns.")

st.markdown("<button><a href=\"#model-exploration\" style=\"text-decoration:none; color:inherit\">🧮 Calculate a cancer risk</a></button>", unsafe_allow_html=True)

st.markdown("## Data collection")

with st.expander("View raw data", expanded=True):
  st.dataframe(data)

# =========================================================
# DATA UNDERSTANDING
# =========================================================

st.markdown("## Data understanding")

with st.expander("Variables included in the model"):
  st.markdown(f"""

Looking at the data features, we had the following options as independent variables (`X[i]`):
 
 * Patient_ID 
 * Cancer_Type
 * Age
 * Gender
 * Smoking
 * Alcohol_Use
 * Obesity
 * Family_History
 * Diet_Red_Meat
 * Diet_Salted_Processed
 * Fruit_Veg_Intake
 * Physical_Activity
 * Air_Pollution
 * Occupational_Hazards
 * BRCA_Mutation
 * H_Pylori_Infection
 * Calcium_Intake
 * BMI
 * Physical_Activity_Level
""")
  
# Apply the mapping to create a new numerical column
risk_level_mapping = {
    'Low': 0,
    'Medium': 0.5,
    'High': 1
}
data['Risk_Level_Numeric'] = data['Risk_Level'].map(risk_level_mapping)

st.markdown("## Data cleaning ")

# Calculate the correlation between Overall_Risk_Score and the new numerical Risk_Level
correlation = data['Overall_Risk_Score'].corr(data['Risk_Level_Numeric'])

with st.expander("Proof of strong correlation between two features"):          
  st.markdown(f"""    

    The data file also had a quantitative dependent variable, `Overall_Risk_Score`
    
    When sorting any of the two columns, we could see:
      * `0.00 to 0.33` -> `Low` 
      * `0.33 to 0.66` -> `Medium` 
      * `0.66 to 1.00` -> `High` 
    
    To ensure this, a correlation was calculated between the `Overall_Risk_Score` and `Risk_Level` features (see the "Data cleaning" section). Since `Risk_Level` is categorical, we first needed to convert it into a numerical representation, before we did a correlation calculation:
    * `Low` to `0`
    * `Medium` to `0.5`
    * `High` to `1`
              
    ```python     
      import pandas as pd
      data = pd.read_excel(data_path)               
      risk_level_mapping = {{
        'Low': 0,
        'Medium': 0.5,
        'High': 1
      }}
      data['Risk_Level_Numeric'] = data['Risk_Level'].map(risk_level_mapping)
      correlation = data['Overall_Risk_Score'].corr(data['Risk_Level_Numeric'])
    ```
        
    **Correlation between Overall_Risk_Score and Risk_Level** = {correlation:.4f}
              
""")
  
with st.expander("Variables excluded from the model"):
 
  st.markdown(f"""

    We did not include the feature `Overall_Risk_Score` and the duplicate feature, `Risk_Level` as they were our dependent variables (`y`). 

    Due to their labeling and already quantitative data types, most of the variables were quantitative enough to be fed into a **linear regression model**!

    However, three features were deemed inappropriate for the model:

    * `Patient_ID` (string)
      - this was not useful to our prediction model as the value for each record was obviously used to uniquely identify each "patient"
    * `Cancer_Type` (string)
      - we would remove this variable because we are looking at overall risk rather than the risk of each type of cancer   
    
    Also:
    
    * `Gender` (binary)
      - we could include this variable but using a `0` for one gender and `1` for another gender might affect the model too greatly if they are to be multiplied by a coefficient
      - for this exercise we will simplify things and look at cancer risk _regardless of gender_
    
    So, we will at first discard `Patient_ID` and `Cancer_Type` upon recommendation of the data provider, and also, discarding `Gender` due to its categorial nature.
""")
  
# Drop the temporary numerical Risk_Level column as it's not needed for further modeling
data = data.drop('Risk_Level_Numeric', axis=1)

# Drop features deemed inappropriate for the model
data = data.drop('Patient_ID', axis= 1)   # not quantitative
data = data.drop('Cancer_Type', axis= 1)  # doubt because of categorical nature (not truly quantitative)
data = data.drop('Gender', axis= 1)       # doubt because of categorical nature (not truly quantitative)
data = data.drop('Risk_Level', axis= 1)   # somewhat redundant

with st.expander("Removing features"):
  st.markdown(f"""
  
    After removing the features `Patient_ID`, `Cancer_Type`, `Gender` and `Risk_Level`, our data now looks like this:
  
  """)
  st.dataframe(data)

# =========================================================
# DATA MODELING
# =========================================================

st.markdown("## Data modeling")

from sklearn.model_selection import train_test_split

X = data.drop(['Overall_Risk_Score'], axis=1)
y = data['Overall_Risk_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

with st.expander("Splitting into training and testing datasets"):

  st.markdown(f"""
  
    We split the data into training and testing datasets:
    
    * training data: 80% of the data
    * testing data: 20% of the data
    
    This was done using the `train_test_split` function from `sklearn.model_selection`:
    
    ```python
    from sklearn.model_selection import train_test_split

    X = data.drop(['Overall_Risk_Score'], axis=1)
    y = data['Overall_Risk_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    ```            
  
  """)

  st.write("Training dataset size: ", len(X_train))
  st.write("Testing dataset size: ", len(X_test))
  st.write("Maximum y_train value: ", y_train.max())
  st.write("Minimum y_train value: ", y_train.min())
  st.write("Maximum y_test value: ", y_test.max())
  st.write("Minimum y_test value: ", y_test.min())

with st.expander("Model building", expanded=False):

  from sklearn.linear_model import LinearRegression

  # model choice (linear regression)
  model_slr = LinearRegression()
  model_slr = model_slr.fit(X_train,y_train)

  # regression equation building
  feature_names = X_train.columns
  coefficients = model_slr.coef_

  st.markdown(f"""

    With this code: 
              
    ```python
    from sklearn.linear_model import LinearRegression

    # model choice (linear regression)
    model_slr = LinearRegression()
    model_slr = model_slr.fit(X_train,y_train)

    # regression equation building
    feature_names = X_train.columns
    coefficients = model_slr.coef_
    
    # predicting y with 20% of test data, X (ignoring the "real" y)
    y_pred = model_slr.predict(X_test)

    # reset indices
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    ```

  """)

  st.write("The regression model yielded the following equation:")
  st.write(f"Overall_Risk_Score = {model_slr.intercept_:.4f}")
  for i, feature in enumerate(feature_names):
      st.write(f" + {coefficients[i]:.4f} * {feature}")
  st.write()

  # predicting y with 20% of test data, X (ignoring the "real" y)
  y_pred = model_slr.predict(X_test)

  # reset indices
  y_pred = pd.Series(y_pred).reset_index(drop=True)
  
with st.expander("Model evaluation"):

  st.markdown(f"""
              
More objectively, we can evaluate the model mathematically by running through three tests:

* Mean absolute error (MAE)
* Mean squared error (MSE)
* Root mean squared error (RMSE)

The closer to 0 each of these scores, the better, so with this code:
              
  ```python
  from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

  mae1 = mean_absolute_error(y_pred, y_test)
  mse1 = mean_squared_error(y_pred, y_test)
  rmse1 = root_mean_squared_error(y_pred, y_test)

  metrics1 = [ ("Mean absolute error (MAE)", mae1), ("Mean squared error (MSE)", mse1), ("Root mean squared error (RMSE)", rmse1) ]
  ```

""")
  
  from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

  mae_slr = mean_absolute_error(y_pred, y_test)
  mse_slr = mean_squared_error(y_pred, y_test)
  rmse_slr = root_mean_squared_error(y_pred, y_test)

  metrics_slr = [ ("Mean absolute error (MAE)", mae_slr), ("Mean squared error (MSE)", mse_slr), ("Root mean squared error (RMSE)", rmse_slr) ]

  for name, metric_value in metrics_slr:
    st.write(f"{name}: {metric_value:.4f}")

  import matplotlib.pyplot as plt
  
  df1 = pd.DataFrame(y_pred)
  df2 = pd.DataFrame(y_test)
  df_comparison = pd.concat([df1, df2.reset_index(drop=True)], axis=1)
  df_comparison.columns = ['Predicted_Risk_Score (regression)', 'Actual_Risk_Score (from data)']
  df_comparison['Difference'] = df_comparison['Predicted_Risk_Score (regression)'] - df_comparison['Actual_Risk_Score (from data)']
  
  fig, ax = plt.subplots()
  ax.plot(df_comparison['Predicted_Risk_Score (regression)'], color='red', label='prediction data')
  ax.plot(df_comparison['Actual_Risk_Score (from data)'], color='blue', label='test data')
  ax.set_xlabel('Sample Index')
  ax.set_ylabel('Overall Risk Score')
  ax.set_title('Predicted vs. Actual Risk Scores (Aligned by Index)')
  ax.legend()
  st.pyplot(fig)

# =========================================================
# MODEL EXPLORATION
# =========================================================

if st.session_state.get("scroll_to") == "model_exploration":
  st.session_state.scroll_to = None

st.markdown("## Model exploration", unsafe_allow_html=True)

with st.expander("Sample prediction", expanded=True):

  st.write("For the sliders, use 0 for low and 10 for high amounts:")

  col1, col2, col3 = st.columns(3)

  with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=40)

  with col2:
    family_history = st.selectbox("Family history of cancer?", options=[("No", 0), ("Yes", 1)], index=1, format_func=lambda x: x[0])
    family_history = family_history[1]

  with col3:        
    bmi = st.number_input("Enter BMI", min_value=15.0, value=25.0, step=0.1)
    st.markdown("[BMI calculator](https://www.diabetes.ca/resources/tools-resources/body-mass-index-(bmi)-calculator)", text_alignment="right")

  col1, col2 = st.columns(2)

  with col1:
    brca_mutation = st.selectbox("BRCA Mutation?", options=[("No", 0), ("Yes", 1)], index=0, format_func=lambda x: x[0])
    brca_mutation = brca_mutation[1]

  with col2:
    h_pylori_infection = st.selectbox("H Pylori Infection?", options=[("No", 0), ("Yes", 1)], index=0, format_func=lambda x: x[0])
    h_pylori_infection = h_pylori_infection[1]
    
  col1, col2, col3 = st.columns(3)
  
  with col1:
    smoking = st.slider("Smoking", min_value=0, max_value=10, value=5)
    alcohol_use = st.slider("Alcohol Use", min_value=0, max_value=10, value=5)
    obesity = st.slider("Obesity", min_value=0, max_value=10, value=5)    
    diet_red_meat = st.slider("Diet of Red Meat", min_value=0, max_value=10, value=5)
    
  with col2:
    diet_salted_processed = st.slider("Diet Salted Processed", min_value=0, max_value=10, value=5)
    fruit_veg_intake = st.slider("Fruit Veg Intake", min_value=0, max_value=10, value=5)
    physical_activity = st.slider("Physical Activity", min_value=0, max_value=10, value=5)
    air_pollution = st.slider("Air Pollution", min_value=0, max_value=10, value=5)
    
  
  with col3:    
    calcium_intake = st.slider("Calcium Intake", min_value=0, max_value=10, value=5)    
    physical_activity_level = st.slider("Physical Activity Level", min_value=0, max_value=10, value=5)
    occupational_hazards = st.slider("Occupational Hazards", min_value=0, max_value=10, value=5)
    
  # Create sample data from input values
  sample_data = [[
      age, smoking, alcohol_use, obesity, family_history,
      diet_red_meat, diet_salted_processed, fruit_veg_intake, physical_activity,
      air_pollution, occupational_hazards, brca_mutation, h_pylori_infection,
      calcium_intake, bmi, physical_activity_level
  ]]
  sample_df = pd.DataFrame(sample_data, columns=X_train.columns)

  # make a prediction with the model
  single_prediction = model_slr.predict(sample_df)

  st.write(f"Overall risk score: **{single_prediction[0]:.4f}**")

  # assign risk level interpretations
  if single_prediction[0] >= 0.66 + mae_slr:
    st.write("**Risk level: ❗❗❗ High risk**")
  elif single_prediction[0] >= 0.66 - mae_slr and single_prediction[0] < 0.66 + mae_slr:
    st.write("**Risk level: ❗❗ Medium to high risk**")
  elif single_prediction[0] >= 0.33 + mae_slr and single_prediction[0] < 0.66 - mae_slr:
    st.write("**Risk level: ❗ Medium risk**")
  elif single_prediction[0] >= 0.33 - mae_slr and single_prediction[0] < 0.33 + mae_slr:
    st.write("**Risk level: ⚠️ Low to medium risk**")
  elif single_prediction[0] > 0 and single_prediction[0] < 0.33 - mae_slr:
    st.write("**Risk level: 🎗️ Low risk - but take care**")
  else:
    st.write("**Error: prediction is not between 0 and 1**")

st.write("This app is only to be used for educational purposes and not for real medical advice. Always consult a healthcare professional for medical concerns.")