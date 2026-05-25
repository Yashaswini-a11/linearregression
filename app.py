
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


st.set_page_config(
    page_title="House Price Dashboard",
    layout="wide"
)

st.title(" Melbourne House Price Dashboard")

st.markdown("---")


df = pd.read_csv("melb_data.csv")



df.fillna(df.mean(numeric_only=True), inplace=True)

df = pd.get_dummies(df, drop_first=True)


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



col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])

col2.metric("Columns", df.shape[1])

col3.metric("R2 Score", round(score, 2))

st.markdown("---")


st.subheader("Dataset Preview")

st.dataframe(df.head())


col4, col5 = st.columns(2)


with col4:

    st.subheader("Actual vs Predicted")

    fig1, ax1 = plt.subplots()

    ax1.scatter(y_test, y_pred)

    ax1.set_xlabel("Actual Price")

    ax1.set_ylabel("Predicted Price")

    ax1.set_title("Prediction Graph")

    st.pyplot(fig1)


with col5:

    st.subheader("Price Distribution")

    fig2, ax2 = plt.subplots()

    ax2.hist(df["Price"], bins=20)

    ax2.set_xlabel("Price")

    ax2.set_ylabel("Count")

    ax2.set_title("House Price Distribution")

    st.pyplot(fig2)


st.subheader("Correlation Matrix")

corr = df.corr(numeric_only=True)

fig3, ax3 = plt.subplots(figsize=(12,6))

cax = ax3.imshow(corr, cmap="coolwarm")

plt.colorbar(cax)

st.pyplot(fig3)

st.subheader("Prediction Results")

result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

st.dataframe(result.head(20))