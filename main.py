import streamlit as st
import pickle
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
pic = pickle.load(open('lr1.pkl','rb'))
with st.sidebar:
    selected = option_menu('Lungs Cancer Prediction',
                           ['About','Dataset','Prediction'],
                           icons = ['folder-fill','folder-fill'],
                           default_index = 0)
if (selected == 'About'):
        st.title('ABOUT')
        st.subheader('A lungs  cancer that begins in the lungs and most often occurs in people who smoke. This web is created in the  purposes. So, we can   predict the person is suffer from the lungs cancer or not.  We can use this web in medical purposes.')
if (selected == 'Dataset'):
    dataframe = pd.read_csv('lung cancer.csv')
    dataframe
if (selected == 'Prediction'):
        st.title('LUNGS CANCER PREDICTION')
        gender = st.radio('Enter your gender',('Male','Female'))
        age = st.number_input('Age of the person')
        smoke = st.radio('Do you smoke?',('yes','no'))
        alcohol_consuming = st.radio('Do you drink alcohol?',('yes','no'))
        breathproblem = st.radio('Are you suffering from shortness of breath?',('yes','no'))
        chestpain = st.radio('Are you troubled by the problem of chest pain?',('yes','no'))
        lungs_diagonsis = ''
        cat_val=[smoke,alcohol_consuming,breathproblem,chestpain]
        for e,val in enumerate(cat_val):
            if val == 'yes':
                cat_val[e]=2
            else:
                cat_val[e]=1
        print(cat_val)
        if gender=='male':
            gender=1
        else:
            gender=0
        if st.button('Predict'):
            # cat_val = [smoke, alcohol_consuming, breathproblem, chestpain]
            lungs_cancer_prediction = pic.predict_proba([[gender,age,cat_val[0],cat_val[1],cat_val[2],cat_val[3]]])
            lungs_cancer_prediction = np.round(lungs_cancer_prediction,2)
            print(lungs_cancer_prediction)
            print(np.argmax(lungs_cancer_prediction))

            if (np.argmax(lungs_cancer_prediction) ==1):
                lungs_diagonsis=f'The person has lungs cancer with probability {max(lungs_cancer_prediction[0])}'
                st.text(lungs_diagonsis)
            else:
                lungs_diagonsis = f'The person has not lungs cancer with probability {max(lungs_cancer_prediction[0])}'
                st.text(lungs_diagonsis)








