# app.py
import streamlit as st
from dashboard.layout import render_layout

st.set_page_config(
    page_title="Risk Model Pricing Framework",
    layout="wide",
)

if __name__ == "__main__":
    render_layout()
