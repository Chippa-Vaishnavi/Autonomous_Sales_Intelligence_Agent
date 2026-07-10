import pandas as pd
import streamlit as st


def load_dataset(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        st.success('Dataset loaded successfully.')
        return df
    except Exception as exc:
        st.error(f'Failed to load dataset: {exc}')
        raise
