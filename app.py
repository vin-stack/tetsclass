import cohere
import pandas as pd
from cohere import ClassifyExample
import streamlit as st

# Initialize Cohere client
co = cohere.Client('ldlI6DSf2OeIBg7ZsJHUumgiqKEcR15FpH8RU8w3')  # This is your trial API key

# Define inputs
inputs = [
    "where do we get dosa?",
    "how to start with arduino uno?",
    "How to buy BTC for less price?",
    "What is the capital of france?",
    "How to get KFC 50% coupon?",
    "Suggest me best SCIFI series to watch this weekend?",
    "What is the time complexity of bubble sort?",
    "Do you offer self-directed and managed investments?",
    "What is the current interest rate in your savings account?",
    "How to invest in Bitcoin?",
    "What are the best stocks to buy right now?",
    "What is the role of the army in national defense?",
    "How can I protect my personal data online?",
    "What are the top trending video games?",
    "Who won the best director award at the Oscars?",
    "How can I improve my coding skills?",
    "What are the best practices for software development?",
    "What is DevOps?",
    "Explain full stack development.",
    "What is the MERN stack?",
    "How does cloud computing work?",
    "What are the applications of AR/VR?",
    "How do I manage my accounts?",
    "What is a ledger in accounting?",
    "How can I attract more buyers to my online store?",
    "What are the steps in the sales process?",
    "How do I handle purchase orders?",
    "What are the different types of audits?",
    "What are the current tax rates?",
    "What is compound interest?",
    "How to get 50% profit?",
    "How to complete my urgent task?",
    "Is there any scheduled meeting for today?"
]

# Load examples from CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    examples_df = pd.read_csv(uploaded_file)
    examples = [
        ClassifyExample(text=row['text'], label=row['label'])
        for _, row in examples_df.iterrows()
    ]

    # Get classification response
    response = co.classify(model='embed-english-v2.0', inputs=inputs, examples=examples)

    # Prepare data for tabular format
    data = []
    for i, classification in enumerate(response.classifications):
        data.append({
            'Input': inputs[i],
            'Predicted Label': classification.prediction,
            'Confidence': classification.confidence  # assuming confidence is a float directly
        })

    # Create DataFrame
    df = pd.DataFrame(data)

    st.write(df)
else:
    st.write("Please upload a CSV file")

# User input and predict button
user_input = st.text_input("Enter text to predict:")
if st.button("Predict"):
    if user_input:
        # Get classification response for user input
        user_response = co.classify(model='embed-english-v2.0', inputs=[user_input], examples=examples)
        st.write("Predicted Label:", user_response.classifications[0].prediction)
        st.write("Confidence:", user_response.classifications[0].confidence)
    else:
        st.write("Please enter text to predict")
