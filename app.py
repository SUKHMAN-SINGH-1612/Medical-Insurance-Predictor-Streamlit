import streamlit as st
import os
import joblib as jb
import numpy as np

heading_style = '''
<div style="color:red;" align='center'>
<h1>Medical Cost Prediction</h1>
</div>
'''

@st.cache_resource()
def base_model():
    bmodel=jb.load(os.path.join('random-forest-model.pkl'))
    return bmodel


def sex_dict(sex):
    mapping={'male':1,'female':0}
    return mapping[sex]


def smoker_dict(smoker):
    mapping={'yes':1,'no':0}
    return mapping[smoker]

def region_dict(region):
    mapping={'southwest':0,'southeast':1,'northwest':2,'northeast':3}
    return mapping[region]

st.markdown(heading_style, unsafe_allow_html=True)
age=st.number_input('Enter your age:',min_value=0)
sex=sex_dict(st.selectbox('Choose your gender:',['male','female']))
bmi=st.number_input('Enter you BMI:')
children=st.number_input('How many children you have?',min_value=0)
smoker=smoker_dict(st.selectbox('Do you have smoking habit?',['yes','no']))
region=region_dict(st.selectbox('Choose your region?',['southwest','southeast','northwest','northeast']))

if st.button('Know your medical costs billed by health insurance !'):
    user=np.array([age,sex,bmi,children,smoker,region])
    model=base_model()
    st.write(f'Final medical cost by health insurance is {model.predict(user.reshape(1,-1))[0]}')
    st.balloons()


