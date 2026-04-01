# -------------------------------
# IMPORTS
# -------------------------------
import streamlit as st
import base64
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# PAGE CONFIG 
# -------------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    layout="wide"
)

# -------------------------------
# BACKGROUND IMAGE
# -------------------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_path = r"C:\Users\user\Documents\guvi\guvi project 1\fourth_project\bg_for_shopper.jpg"   
encoded = get_base64_image(image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    clustering_model = joblib.load(r"C:\Users\user\Documents\guvi\guvi project 1\fourth_project\kmeans_rfm_model.pkl")
    scaler = joblib.load(r"C:\Users\user\Documents\guvi\guvi project 1\fourth_project\rfm_scaler.pkl")
    cluster_labels = joblib.load(r"C:\Users\user\Documents\guvi\guvi project 1\fourth_project\cluster_labels.pkl")
    similarity = joblib.load(r"C:\Users\user\Documents\guvi\guvi project 1\fourth_project\product_similarity_matrix.pkl")
    products = pd.read_csv(r"C:\Users\user\Documents\guvi\guvi project 1\fourth_project\product_similarity_data.csv")
    return clustering_model, scaler, cluster_labels, similarity, products

clustering_model, scaler, cluster_labels, similarity, products = load_models()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📊 Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Customer Segmentation", "Product Recommendation"]
)

# -------------------------------
# BUTTON STYLE
# -------------------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "Home":
    st.markdown("""
    <h1 style='
    color: white;
    text-align: center;
    text-shadow: 0 0 10px green;
'>
📦 Welcome to Shopper Spectrum
</h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🔍 Features:
    - 📊 Customer Segmentation (RFM + KMeans)
    - 🎯 Product Recommendation (Collaborative Filtering)

    ### 💡 Use Case:
    - Identify customer value
    - Recommend similar products
    """)

# -------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------
elif page == "Customer Segmentation":
    st.title("📊 Customer Segmentation")

    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.number_input("Recency (days)", 0, 365, 50)

    with col2:
        frequency = st.number_input("Frequency", 1, 20, 5)

    with col3:
        monetary = st.number_input("Monetary", 0.0, 10000.0, 1000.0)

    if st.button("Predict Segment"):

        # Prepare input
        input_data = np.array([[recency, frequency, monetary]], dtype=float)

        # SAME preprocessing as training
        input_data = np.log1p(input_data)
        scaled = scaler.transform(input_data)

        # Predict
        cluster = clustering_model.predict(scaled)[0]
        segment = cluster_labels.get(cluster, "Unknown")

        # Output
        st.success(f"Customer belongs to: {segment}")
        st.write(f"Cluster ID: {cluster}")

        # 🎉 Bonus effect
        if segment == "High-Value":
            st.balloons()

# -------------------------------
# PRODUCT RECOMMENDATION
# -------------------------------
elif page == "Product Recommendation":
    st.title("🎯 Product Recommend")

    # Clean product list
    product_list = products['Description'].dropna().unique()
    product_list.sort()

    selected_product = st.selectbox("Select a Product", product_list)

    def recommend(product):
        try:
            idx = products[products['Description'] == product].index[0]
            distances = similarity[idx]

            product_indices = sorted(
                list(enumerate(distances)),
                key=lambda x: x[1],
                reverse=True
            )[1:6]

            recommended_products = [
                products.iloc[i[0]].Description for i in product_indices
            ]

            return recommended_products

        except:
            return []

    if st.button("Get Recommendations"):
        recommendations = recommend(selected_product)

        if not recommendations:
            st.error("No recommendations found!")
        else:
            st.subheader("Recommended Products:")
            for prod in recommendations:
                st.markdown(f"✅ {prod}")