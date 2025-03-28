import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open(r'C:\Users\LENOVO\Downloads\trained_model.sav','rb'))

# Creating a function for prediction
def subsc_pred(CreditScore,Tenure,Balance,NumOfProducts,IsActiveMember,EstimatedSalary,Germany,Spain):
    input_data = np.array((CreditScore,Tenure,Balance,NumOfProducts,IsActiveMember,EstimatedSalary,Germany,Spain), dtype=float)
    input_data_reshaped = input_data.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'Person will not exit.'
    else:
        return 'Person will exit.'

def main():
    # Title for our webpage
    st.title("Subscribers Experiment")

    # Getting input data from the user
    CreditScore = st.text_input("Credit Score")
    Tenure = st.text_input("Tenure")
    Balance = st.text_input("Account Balance")
    NumOfProducts = st.text_input("Number of products")
    IsActiveMember = st.text_input("Are you an Active member? (Active member-1;Not Active-0)")
    EstimatedSalary = st.text_input("Estimated Salary")
    Germany = st.text_input("Are you from Germany? (yes - 1;no - 0)")
    Spain = st.text_input("Are you from Spain? (yes - 1;no - 0)")

    # Code for prediction
    exited = ''

    # Creating a button for prediction
    if st.button('Result'):
        exited = subsc_pred(CreditScore,Tenure,Balance,NumOfProducts,IsActiveMember,EstimatedSalary,Germany,Spain)

    st.success(exited)

if __name__ == '__main__':
    main()
