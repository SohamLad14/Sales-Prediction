import streamlit as st
import pandas as pd
import pickle
import bz2

# Set the background image and text color
# background_image = """
# <style>
# [data-testid="stAppViewContainer"] > .main {
#     background-image: url("https://media.istockphoto.com/id/1035988708/photo/3d-business-graph-shows-financial-growth.jpg?s=1024x1024&w=is&k=20&c=H_PVfYDXN1jDGu3Z7hCVTg46xTQE1kbS8dpppu1SiIU=");
#     background-size: 100vw 100vh;
#     background-position: center;  
#     background-repeat: no-repeat;
#     color: #0000FF; /* Set text color to blue */
# }
# </style>
# """

# st.markdown(background_image, unsafe_allow_html=True)

# # Load the trained Random Forest model from the pickle file
# with open('rf_model.pkl', 'rb') as f:
#     rf_model = pickle.load(f)

# Load the trained Random Forest model from the compressed pickle file
def load_compressed_pickle(file_path):
    with bz2.BZ2File(file_path, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

# Load the compressed model
rf_model = load_compressed_pickle('rf_model.pbz2')

# Define the function to map Yes/No to 1/0
def map_yes_no_to_binary(value):
    return 1 if value == 'Yes' else 0

# Define the function to make predictions
def predict_sales(store_type, location_type, region_code, holiday, discount, month, order):
    # Map 'Yes'/'No' to 1/0
    holiday = map_yes_no_to_binary(holiday)
    discount = map_yes_no_to_binary(discount)
    
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Store_Type': [store_type],
        'Location_Type': [location_type],
        'Region_Code': [region_code],
        'Holiday': [holiday],
        'Discount': [discount],
        'Month': [month],
        '#Order': [order]
    })
    
    # Make the prediction
    prediction = rf_model.predict(input_data)
    return prediction[0]

# Create the Streamlit web app
def main():
    st.title('Sales Prediction')
    # new_title = '<p style="font-family:sans-serif; color:Blue; font-size: 42px;">Sales Prediction </p>'
    # st.markdown(new_title, unsafe_allow_html=True)
    
    # Add input widgets for user input
    store_type = st.selectbox('Store Type', [1.0, 2.0, 3.0, 4.0])
    location_type = st.selectbox('Location Type', [1.0, 2.0, 3.0, 4.0, 5.0])
    region_code = st.selectbox('Region Code', [1.0, 2.0, 3.0, 4.0])
    holiday = st.selectbox('Holiday', ['Yes', 'No'])
    discount = st.selectbox('Discount', ['Yes', 'No'])
    month = st.slider('Month', 1, 12, 1)
    order = st.slider('#Order', 1, 400, 1)
    
    # Make predictions when the 'Predict' button is clicked
    if st.button('Predict'):
        prediction = predict_sales(store_type, location_type, region_code, holiday, discount, month, order)
        st.success(f'Predicted Sales: {prediction:.2f}')

if __name__ == '__main__':
    main()


