import streamlit as st
import pandas as pd
 
st.write("""
# Past MPV predictions
""")
 
df = pd.read_csv("./outputrank.csv")


st.write(df)