import streamlit as st
import base64
 
st.set_page_config(layout="wide")
 
def get_base64_image(image_path):
     with open(image_path, "rb") as f:
         data = f.read()
     return base64.b64encode(data).decode()
 
image_path = r"C:\Users\user\Documents\guvi\guvi project 1\fourth_project\bg_for_shopper.jpg"
encoded = get_base64_image(image_path)
 
 # Inject CSS to add the background
st.markdown(
     f"""
     <style>
     .stApp {{
         background-image: url("data:image/webp;base64,{encoded}");
         background-size: cover;
         background-repeat: no-repeat;
         background-attachment: fixed;
         background-position: center;
     }}
     </style>
     """,
     unsafe_allow_html=True
)

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Shopper Spectrum Streamlit", layout="wide")

# -------------------------------
# Load Models (SAFE LOAD)
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
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📊 Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Customer Segmentation", "Product Recommendation"]
)

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HOME PAGE
# -------------------------------
if page == "Home":
    st.markdown("""
    <div style="
        background-color: black;
        border-radius: 16px;
        padding: 10px 6px;
        margin-bottom: 18px;
        text-align: center;
        display: inline-block;
        width: 100%;
    ">
        <h1 style="
            color: white;
            -webkit-text-stroke: 2px blue;
            font-weight: bold;
            margin: 0;
        ">
            📦 Welcome to Shopper Spectrum
        </h1>
    </div>
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
        recency = st.number_input("Recency (days)", min_value=0, max_value=400, value=50)

    with col2:
        frequency = st.number_input("Frequency", min_value=1, max_value=100, value=5)

    with col3:
        monetary = st.number_input("Monetary", min_value=0.0, max_value=200000.0, value=1000.0)

    if st.button("Predict Segment"):

        # ✅ Correct input format
        input_data = np.array([[recency, frequency, monetary]], dtype=float)

        # ⚠️ Uncomment ONLY if used during training
        # input_data = np.log1p(input_data)

        # Scale
        scaled_data = scaler.transform(input_data)

        # Predict
        cluster = int(clustering_model.predict(scaled_data)[0])

        # Get label
        segment = cluster_labels.get(cluster, "Unknown")

        st.success(f"Cluster: {cluster}")
        st.info(f"Segment: {segment}")

        

# -------------------------------
# PRODUCT RECOMMENDATION
# -------------------------------
elif page == "Product Recommendation":
    st.title("🎯 Product Recommender")

    # Clean product list
    product_list = products['Description'].dropna().unique()
    product_list.sort()

    # Dropdown selection
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

        if len(recommendations) == 0:
            st.error("No recommendations found!")
        else:
            st.subheader("Recommended Products:")
            for prod in recommendations:
                st.markdown(f"✅ {prod}")