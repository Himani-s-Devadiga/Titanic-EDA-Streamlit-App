
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("AI Data Science Assistant Agent")

uploaded = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded:

    df = pd.read_csv(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.write(df.columns.tolist())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    if st.button("Clean Missing Values"):

        for column in df.columns:

            if df[column].dtype=="object":

                df[column].fillna(
                    df[column].mode()[0],
                    inplace=True
                )

            else:

                df[column].fillna(
                    df[column].mean(),
                    inplace=True
                )

        st.success("Missing values cleaned")

    if st.button("Generate Heatmap"):

        numeric=df.select_dtypes(
            include=["number"]
        )

        fig,ax=plt.subplots(figsize=(10,8))

        sns.heatmap(
            numeric.corr(),
            annot=True,
            ax=ax
        )

        st.pyplot(fig)
