import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")

st.title(" Melbourne House Price Dashboard")

df = pd.read_csv("melb_data.csv").head(3000)

df = df[
    [
        "Rooms",
        "Distance",
        "Bedroom2",
        "Bathroom",
        "Car",
        "Landsize",
        "BuildingArea",
        "YearBuilt",
        "Price"
    ]
]


df = df.fillna(df.mean(numeric_only=True))

X = df.drop("Price", axis=1)

y = df["Price"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

c1, c2, c3 = st.columns(3)

c1.metric("Rows", df.shape[0])

c2.metric("Columns", df.shape[1])

c3.metric("R2 Score", round(score, 2))

st.markdown("---")

st.subheader("Dataset")

st.dataframe(df.head())

col1, col2 = st.columns(2)

with col1:

    st.subheader("Actual vs Predicted")

    fig1, ax1 = plt.subplots()

    ax1.scatter(y_test, y_pred)

    st.pyplot(fig1)

with col2:

    st.subheader("Price Distribution")

    fig2, ax2 = plt.subplots()

    ax2.hist(df["Price"], bins=20)

    st.pyplot(fig2)