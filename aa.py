import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# Background image function
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-attachment: fixed;
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://static.vecteezy.com/system/resources/thumbnails/038/236/603/small_2x/4k-luxury-gold-and-black-stripes-background-video.jpg")

# Path where CSVs are stored
DATA_PATH = r"C:\Users\ajayj\Desktop\archive"

# Map dataset display names to filenames
datasets = {
    "Customers": "olist_customers_dataset.csv",
    "Geolocation": "olist_geolocation_dataset.csv",
    "Order Items": "olist_order_items_dataset.csv",
    "Order Payments": "olist_order_payments_dataset.csv",
    "Order Reviews": "olist_order_reviews_dataset.csv",
    "Orders": "olist_orders_dataset.csv",
    "Products": "olist_products_dataset.csv",
    "Sellers": "olist_sellers_dataset.csv",
    "Product Category Translation": "product_category_name_translation.csv"
}

@st.cache_data
def load_dataset(file_name):
    path = os.path.join(DATA_PATH, file_name)
    return pd.read_csv(path)

# Sidebar Styling & Content
st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(to bottom, #6a11cb, #2575fc);
            color: white;
            padding: 15px;
            border-radius: 10px;
        }
        .sidebar .sidebar-content h1 {
            font-size: 2rem;
            margin-bottom: 10px;
        }
        .sidebar .sidebar-content p {
            font-size: 1rem;
            margin-top: -10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar.container():
    st.markdown(
        """
        <div style="text-align:center;">
            <img src="https://cdn-icons-png.flaticon.com/512/126/126509.png" width="80" alt="Logo"/>
            <h1>🛒 SmartRetail</h1>
            <p><i>Customer Insights Dashboard</i></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Select dataset to load
    selected_dataset = st.selectbox("Select Dataset", list(datasets.keys()))

    # Load data button
    load_data_button = st.button("Load Dataset")

# Placeholder for dataset
data = None

if load_data_button:
    data = load_dataset(datasets[selected_dataset])
    st.success(f"{selected_dataset} dataset loaded successfully!")

if data is not None:
    st.title(f"SmartRetail - {selected_dataset} Dataset")

    # Show data preview or description
    explore_option = st.radio("Choose View", ["Preview Data", "Dataset Description"])

    if explore_option == "Preview Data":
        st.header("📄 Dataset Preview")
        st.dataframe(data.head())

    elif explore_option == "Dataset Description":
        st.header("📊 Dataset Description")
        st.write(data.describe(include='all'))

    # Show dataset-specific insights & visualizations
    st.header("📈 Insights & Visualizations")

    if selected_dataset == "Customers":
        if 'customer_state' in data.columns:
            state_counts = data['customer_state'].value_counts().reset_index()
            state_counts.columns = ['State', 'Number of Customers']
            fig = px.bar(state_counts, x='State', y='Number of Customers',
                         title="Customer Distribution by State")
            st.plotly_chart(fig)

        if 'customer_id' in data.columns:
            data['id_length'] = data['customer_id'].apply(len)
            fig2, ax2 = plt.subplots()
            sns.histplot(data['id_length'], kde=True, ax=ax2)
            ax2.set_title("Customer ID Length Distribution")
            st.pyplot(fig2)

    elif selected_dataset == "Order Items":
        if 'order_item_id' in data.columns:
            fig = px.histogram(data, x='order_item_id', nbins=10, title="Distribution of Order Item IDs")
            st.plotly_chart(fig)

        if 'price' in data.columns:
            fig2 = px.box(data, y='price', title="Price Distribution of Order Items")
            st.plotly_chart(fig2)

    elif selected_dataset == "Products":
        if 'product_category_name' in data.columns:
            top_categories = data['product_category_name'].value_counts().head(10)
            fig = px.bar(top_categories, x=top_categories.index, y=top_categories.values,
                         labels={'x': 'Product Category', 'y': 'Count'}, title="Top 10 Product Categories")
            st.plotly_chart(fig)

    elif selected_dataset == "Orders":
        if 'order_status' in data.columns:
            status_counts = data['order_status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index, title="Order Status Distribution")
            st.plotly_chart(fig)

    # Add your own dataset-specific visualizations here...

    # Simple churn model demo for customers dataset
    if selected_dataset == "Customers" and {'customer_state', 'id_length'}.issubset(data.columns):
        if st.checkbox("Show Churn Prediction Model"):
            dummy_data = data[['customer_state', 'id_length']].copy()
            dummy_data = pd.get_dummies(dummy_data, drop_first=True)
            dummy_data['churn'] = np.random.randint(0, 2, dummy_data.shape[0])  # Random churn simulation

            X = dummy_data.drop('churn', axis=1)
            y = dummy_data['churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.subheader("Classification Report:")
            st.text(classification_report(y_test, predictions))

            st.subheader("Confusion Matrix:")
            st.text(confusion_matrix(y_test, predictions))

    # Export option
    if st.button("Export Current Dataset to CSV"):
        export_path = f"cleaned_{datasets[selected_dataset]}"
        data.to_csv(export_path, index=False)
        st.success(f"Dataset exported as {export_path}")

else:
    st.title("Welcome to SmartRetail Dashboard")
    st.write("Select a dataset from the sidebar and click 'Load Dataset' to start exploring.")

