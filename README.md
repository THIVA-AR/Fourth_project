# 📦 Shopper Spectrum

#### Problem statement for project intro
📣 
The global e-commerce industry generates vast amounts of transaction data daily, offering valuable insights into customer purchasing behaviors. Analyzing this data is essential for identifying meaningful customer segments and recommending relevant products to enhance customer experience and drive business growth. This project aims to examine transaction data from an online retail business to uncover patterns in customer purchase behavior, segment customers based on Recency, Frequency, and Monetary (RFM) analysis, and develop a product recommendation system using collaborative filtering techniques.





### Customer Segmentation & Product Recommendation System

---

## 🚀 Project Overview

**Shopper Spectrum** is an end-to-end Machine Learning project that analyzes customer purchasing behavior and provides:

* 📊 **Customer Segmentation** using RFM Analysis + KMeans Clustering
* 🎯 **Product Recommendation** using Collaborative Filtering

This project helps businesses understand their customers and improve marketing strategies.

---

## 🎯 Features

### 📊 1. Customer Segmentation

* Uses **RFM (Recency, Frequency, Monetary)** analysis
* Applies **KMeans Clustering**
* Segments customers into:

  * 💎 High-Value
  * 🙂 Regular
  * 🛍️ Occasional
  * ⚠️ At-Risk

---

### 🎯 2. Product Recommendation

* Uses **Item-based Collaborative Filtering**
* Recommends **Top 5 similar products**
* Dropdown-based product selection in UI

---

### 💻 3. Interactive Web App

Built using **Streamlit**

* Clean UI with background styling
* Real-time predictions
* Visual feedback for each segment

---

## 🧠 Machine Learning Workflow

1. Data Cleaning & Preprocessing
2. Feature Engineering (RFM Calculation)
3. Log Transformation (to reduce skewness)
4. Feature Scaling (StandardScaler)
5. Clustering (KMeans)
6. Model Evaluation:

   * Elbow Method
   * Silhouette Score
7. Cluster Labeling based on business logic
8. Model Saving using Joblib
9. Deployment using Streamlit

---

## 📁 Project Structure

```
Shopper-Spectrum/
│
├── streamlit_app.py
├── kmeans_rfm_model.pkl
├── rfm_scaler.pkl
├── cluster_labels.pkl
├── product_similarity_matrix.pkl
├── product_similarity_data.csv
├── bg_for_shopper.jpg
├── README.md
```

---

## ⚙️ Installation



## ▶️ Run the App

```bash
streamlit run streamlit_app.py
```

---

## 📊 Sample Input

| Recency | Frequency | Monetary | Output         |
| ------- | --------- | -------- | -------------- |
| 10      | 15        | 8000     | 💎 High-Value  |
| 70      | 5         | 2000     | 🙂 Regular     |
| 20      | 2         | 500      | 🛍️ Occasional |
| 200     | 1         | 300      | ⚠️ At-Risk     |

---

## 🎨 UI Highlights

* 🔵 Glowing title design
* 🎉 Confetti animation for high-value customers
* ⚠️ Alert system for at-risk customers
* 📦 Dropdown product selector

---

## 🛠️ Technologies Used

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Streamlit
* Joblib

---

## 🎤 Key Learnings

* Feature Engineering using RFM
* Handling skewed data with log transformation
* Clustering techniques (KMeans)
* Model deployment using Streamlit
* Debugging real-world ML pipeline issues

---

## 🚀 Future Improvements

* 📈 Add customer behavior dashboard
* 🌐 Deploy on Streamlit Cloud
* 🤖 Improve recommendation system with deep learning
* 📊 Add interactive charts

---

## 🙌 Acknowledgements

This project was built as part of a Data Science learning journey to understand real-world business applications of Machine Learning.

---

## 📬 Contact

If you like this project, feel free to connect!

* 💼 LinkedIn: (Add your link)
* 📧 Email: (Add your email)

---

⭐ Don’t forget to **star this repo** if you found it useful!
