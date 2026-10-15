
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Chance of Admission Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def train_model():
    url = (
        "https://github.com/ybifoundation/Dataset/"
        "raw/main/Admission%20Chance.csv"
    )

    df = pd.read_csv(url)

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    # Define input features and target
    X = df.drop(
        columns=["Serial No", "Chance of Admit"]
    )

    y = df["Chance of Admit"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.7,
        random_state=2529
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    return model, list(X.columns), mae


st.title("🎓 Chance of Admission Predictor")

st.write(
    "Estimate your graduate admission chance "
    "using your academic profile."
)

st.info(
    "This tool provides an estimate based on historical data, "
    "not a guarantee of admission."
)

try:
    model, feature_names, mae = train_model()

    st.subheader("Enter Your Academic Details")

    with st.form("admission_form"):

        gre = st.number_input(
            "GRE Score",
            min_value=290,
            max_value=340,
            value=315
        )

        toefl = st.number_input(
            "TOEFL Score",
            min_value=92,
            max_value=120,
            value=105
        )

        university_rating = st.slider(
            "University Rating",
            min_value=1,
            max_value=5,
            value=3
        )

        sop = st.slider(
            "Statement of Purpose (SOP)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )

        lor = st.slider(
            "Letter of Recommendation (LOR)",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5
        )

        cgpa = st.number_input(
            "Undergraduate CGPA",
            min_value=6.8,
            max_value=9.92,
            value=8.0,
            step=0.01
        )

        research = st.selectbox(
            "Research Experience",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        submitted = st.form_submit_button(
            "Predict Admission Chance"
        )

    if submitted:

        input_data = pd.DataFrame(
            [[
                gre,
                toefl,
                university_rating,
                sop,
                lor,
                cgpa,
                research
            ]],
            columns=feature_names
        )

        prediction = float(model.predict(input_data)[0])

        # Keep the displayed estimate within the dataset's
        # observed target range.
        prediction = max(0.34, min(0.97, prediction))

        st.subheader("Your Prediction")

        st.metric(
            label="Estimated Chance of Admission",
            value=f"{prediction * 100:.2f}%"
        )

        st.progress(prediction)

        st.caption(
            "The percentage is a model estimate, not a calibrated "
            "probability or a university admission decision."
        )

        st.write(f"Model test-set MAE: {mae:.4f}")

except Exception as error:
    st.error(
        "The application could not load the dataset or train the model."
    )
    st.caption("Check your internet connection and dataset URL.")

