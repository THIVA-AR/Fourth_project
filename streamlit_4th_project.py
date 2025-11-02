import streamlit as st
import base64
 
st.set_page_config(layout="wide")
 
def get_base64_image(image_path):
     with open(image_path, "rb") as f:
         data = f.read()
     return base64.b64encode(data).decode()
 
image_path = r"C:\Users\user\Documents\guvi\guvi project 1\fourth project\bg_for_shopper.jpg"
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
import pandas as pd
import pickle

# ----------- Load Data Functions -----------
@st.cache_data
def load_product_data():
    df = pd.read_csv(r"C:\Users\user\Documents\guvi\guvi project 1\fourth project\online_retail.csv")
    products = df[['Description']].dropna().drop_duplicates().reset_index(drop=True)

    with open(r"C:\Users\user\Documents\guvi\guvi project 1\fourth project\item_similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    # Align product list with similarity matrix
    if len(products) > len(similarity):
        products = products.iloc[:len(similarity)]
    elif len(products) < len(similarity):
        similarity = similarity[:len(products)]

    return products, similarity

@st.cache_resource
def load_cluster_model():
    with open(r"C:\Users\user\Documents\guvi\guvi project 1\fourth project\rfm_cluster_model.pkl", "rb") as f:
        model_dict = pickle.load(f)
    scaler = model_dict["scaler"]
    kmeans = model_dict["kmeans"]
    return scaler, kmeans

# ----------- Main App Navigation -----------
st.sidebar.title("🧠 Retail AI App")
page = st.sidebar.radio("Navigate", ["Home", "Recommendation", "Clustering"], key="nav_radio")

# ----------- Pages -----------
if page == "Home":
    st.markdown("""
    <div style="
        background-color: black;
        border-radius: 16px;
        padding: 18px 8px;
        margin-bottom: 24px;
        text-align: center;
        display: inline-block;
        width: 100%;
    ">
        <h1 style="
            color: white;
            -webkit-text-stroke: 2px greeen;
            font-weight: bold;
            margin: 0;
        ">
            📦 Welcome to the Retail AI Dashboard
        </h1>
    </div>
""", unsafe_allow_html=True)
    st.markdown("Use the sidebar to explore product recommendations and customer segmentation.")

# ------------------ RECOMMENDATION ------------------
elif page == "Recommendation":
    st.title("🔎 Product Recommendation")

    products_df, similarity_matrix = load_product_data()
    product_names = products_df['Description'].tolist()
    selected_product = st.selectbox("Select a product:", product_names)

    if st.button("Recommend"):
        try:
            index = product_names.index(selected_product)
            distances = similarity_matrix[index]
            top_indices = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]

            st.subheader("Recommended Products:")
            for i in top_indices:
                st.write(f"- {product_names[i[0]]}")
        except Exception as e:
            st.error(f"Recommendation failed: {e}")

# ------------------ CLUSTERING ------------------
elif page == "Clustering":
    st.title("🧬 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=1.0, value=1000.0, step=100.0)

    if st.button("Predict Segment"):
        try:
            # Load the saved scaler and model
            scaler, model = load_cluster_model()

            # Match column names used during training
            input_data = pd.DataFrame(
                [[recency, frequency, monetary]],
                columns=['Recency', 'Frequency', 'Monetary']
            )

            # Transform and predict
            scaled_input = scaler.transform(input_data)
            segment = model.predict(scaled_input)[0]

            # ✅ Label mapping based on your actual cluster centers
            segment_labels = {
    0: "High-Value Customer",
    1: "At-Risk Customer",       # ✅ Cluster 1 = highest recency = worst
    2: "Top Spender",
    3: "Regular Customer"
}


            # ✅ Show prediction and label
            st.write("🔢 Predicted cluster number:", segment)
            st.success(f"🧾 This customer belongs to: {segment_labels.get(segment, 'Unknown')}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")





