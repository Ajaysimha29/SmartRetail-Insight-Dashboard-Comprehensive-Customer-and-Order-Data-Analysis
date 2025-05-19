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

DATA_PATH = r"C:\Users\ajayj\Desktop\archive"

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

    selected_dataset = st.selectbox("Select Dataset", list(datasets.keys()))
    load_data_button = st.button("Load Dataset")

data = None

if load_data_button:
    data = load_dataset(datasets[selected_dataset])
    st.success(f"{selected_dataset} dataset loaded successfully!")

if data is not None:
    st.title(f"SmartRetail - {selected_dataset} Dataset")

    explore_option = st.radio("Choose View", ["Preview Data", "Dataset Description"])

    if explore_option == "Preview Data":
        st.header("📄 Dataset Preview")
        st.dataframe(data.head())

    elif explore_option == "Dataset Description":
        st.header("📊 Dataset Description")
        st.write(data.describe(include='all'))

    st.header("📈 Insights & Visualizations")

    # --- Extended part starts here ---

    if selected_dataset == "Customers":
        # Customer distribution
        if 'customer_state' in data.columns:
            state_counts = data['customer_state'].value_counts().reset_index()
            state_counts.columns = ['State', 'Number of Customers']
            fig = px.bar(state_counts, x='State', y='Number of Customers',
                         title="Customer Distribution by State")
            st.plotly_chart(fig)

        # Customer ID length distribution
        if 'customer_id' in data.columns:
            data['id_length'] = data['customer_id'].apply(len)
            fig2, ax2 = plt.subplots()
            sns.histplot(data['id_length'], kde=True, ax=ax2)
            ax2.set_title("Customer ID Length Distribution")
            st.pyplot(fig2)

        # Load related reviews dataset for happiness prediction
        st.subheader("Customer Happiness Prediction")

        if st.checkbox("Load Reviews and Predict Happiness"):
            # Load reviews dataset
            reviews = load_dataset(datasets["Order Reviews"])

            # Merge customers with reviews on customer_id via orders
            orders = load_dataset(datasets["Orders"])
            merged = orders[['order_id', 'customer_id']].merge(reviews[['order_id', 'review_score']], on='order_id', how='inner')

            # Aggregate average review score per customer
            customer_reviews = merged.groupby('customer_id')['review_score'].mean().reset_index()
            customer_reviews.rename(columns={'review_score': 'avg_review_score'}, inplace=True)

            # Merge with customers data
            customer_data = data.merge(customer_reviews, on='customer_id', how='left').fillna(0)

            # Define happiness label (e.g., avg_review_score >=4 -> Happy (1), else Not Happy (0))
            customer_data['happy'] = (customer_data['avg_review_score'] >= 4).astype(int)

            st.write("Customer data with happiness label:")
            st.dataframe(customer_data[['customer_id', 'avg_review_score', 'happy']].head())

            # Prepare features for selection
            customer_data['id_length'] = customer_data['customer_id'].apply(len)

            # Define possible features for prediction
            possible_features = ['id_length', 'avg_review_score']  # add more features if available

            # Feature selector in sidebar
            selected_features = st.sidebar.multiselect(
                "Select Features for Prediction",
                options=possible_features,
                default=possible_features
            )

            if len(selected_features) == 0:
                st.warning("Please select at least one feature to train the model.")
            else:
                X = customer_data[selected_features]
                y = customer_data['happy']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                st.subheader("Classification Report:")
                st.text(classification_report(y_test, preds))

                st.subheader("Confusion Matrix:")
                st.text(confusion_matrix(y_test, preds))

    if selected_dataset == "Order Items":
        if 'order_item_id' in data.columns:
            fig = px.histogram(data, x='order_item_id', nbins=10, title="Distribution of Order Item IDs")
            st.plotly_chart(fig)

        if 'price' in data.columns:
            fig2 = px.box(data, y='price', title="Price Distribution of Order Items")
            st.plotly_chart(fig2)

        # Product sales summary
        st.subheader("Product Sales Summary")
        product_sales = data.groupby('product_id')['order_item_id'].count().reset_index()
        product_sales.rename(columns={'order_item_id': 'total_sold'}, inplace=True)
        top_products = product_sales.sort_values(by='total_sold', ascending=False).head(10)

        fig3 = px.bar(top_products, x='product_id', y='total_sold',
                      labels={'product_id': 'Product ID', 'total_sold': 'Total Units Sold'},
                      title="Top 10 Best Selling Products")
        st.plotly_chart(fig3)

    # Add more dataset-specific insights here...

    # Export option
    if st.button("Export Current Dataset to CSV"):
        export_path = f"cleaned_{datasets[selected_dataset]}"
        data.to_csv(export_path, index=False)
        st.success(f"Dataset exported as {export_path}")

else:
    st.title("Welcome to SmartRetail Dashboard")
    st.write("Select a dataset from the sidebar and click 'Load Dataset' to start exploring.")
