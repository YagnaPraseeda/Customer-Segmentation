import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Function to preprocess the data
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    df['Total_Purchase'] = df['Quantity'] * df['UnitPrice']

    Q1_total = df['Total_Purchase'].quantile(0.25)
    Q3_total = df['Total_Purchase'].quantile(0.75)
    IQR_total = Q3_total - Q1_total
    lower_bound_total = Q1_total - 1.5 * IQR_total
    upper_bound_total = Q3_total + 1.5 * IQR_total
    df = df[(df['Total_Purchase'] >= lower_bound_total) & (df['Total_Purchase'] <= upper_bound_total)]

    Q1_quantity = df['Quantity'].quantile(0.25)
    Q3_quantity = df['Quantity'].quantile(0.75)
    IQR_quantity = Q3_quantity - Q1_quantity
    lower_bound_quantity = Q1_quantity - 1.5 * IQR_quantity
    upper_bound_quantity = Q3_quantity + 1.5 * IQR_quantity
    df = df[(df['Quantity'] >= lower_bound_quantity) & (df['Quantity'] <= upper_bound_quantity)]

    Q1_unit_price = df['UnitPrice'].quantile(0.25)
    Q3_unit_price = df['UnitPrice'].quantile(0.75)
    IQR_unit_price = Q3_unit_price - Q1_unit_price
    lower_bound_unit_price = Q1_unit_price - 1.5 * IQR_unit_price
    upper_bound_unit_price = Q3_unit_price + 1.5 * IQR_unit_price
    df = df[(df['UnitPrice'] >= lower_bound_unit_price) & (df['UnitPrice'] <= upper_bound_unit_price)]

    df = df[(df['CustomerID'] >= 12000) & (df['CustomerID'] <= 18500)]

    return df

# Function for customer-based clustering
def perform_customer_clustering(df):
    customer_data = df[['CustomerID', 'Total_Purchase']].dropna()
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data[['Total_Purchase']])

    distortions = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(customer_data_scaled)
        distortions.append(kmeans.inertia_)

    # Capture elbow plot
    elbow_fig, elbow_ax = plt.subplots(figsize=(10, 6))
    elbow_ax.plot(k_range, distortions, marker='o')
    elbow_ax.set_title('Elbow Method for Optimal k (Customer Clustering)')
    elbow_ax.set_xlabel('Number of Clusters (k)')
    elbow_ax.set_ylabel('Distortion (Within-cluster Sum of Squares)')

    # Show elbow plot in Streamlit
    st.pyplot(elbow_fig)

    optimal_k = st.sidebar.slider("Select optimal k", min_value=1, max_value=10, value=3)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

    scaled_cluster_centers = kmeans.cluster_centers_
    centers_original_scale = scaler.inverse_transform(scaled_cluster_centers)

    plt.figure(figsize=(15, 8))
    plt.scatter(customer_data['CustomerID'], customer_data['Total_Purchase'], c=customer_data['Cluster'], cmap='viridis')

    for i, center_scaled in enumerate(scaled_cluster_centers):
        center_val = centers_original_scale[i, 0]
        mean_customer_id = customer_data[customer_data['Cluster'] == i]['CustomerID'].mean()

        plt.scatter(mean_customer_id, center_val, c='black', marker='o', s=200)
        plt.annotate(f'Center: {center_val:.2f}', (mean_customer_id, center_val), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title('Customer-based Clustering')
    plt.xlabel('CustomerID')
    plt.ylabel('Total_Purchase')
    st.pyplot(plt)

# Function for geographical clustering
def perform_geographical_clustering(df):
    geo_data = df[['Country', 'Total_Purchase']].dropna()
    geo_data['Country'] = geo_data['Country'].astype('category').cat.codes
    scaler = StandardScaler()
    geo_data_scaled = scaler.fit_transform(geo_data[['Total_Purchase']])

    distortions = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(geo_data_scaled)
        distortions.append(kmeans.inertia_)

    # Capture elbow plot
    elbow_fig, elbow_ax = plt.subplots(figsize=(10, 6))
    elbow_ax.plot(k_range, distortions, marker='o')
    elbow_ax.set_title('Elbow Method for Optimal k (Geographical Clustering)')
    elbow_ax.set_xlabel('Number of Clusters (k)')
    elbow_ax.set_ylabel('Distortion (Within-cluster Sum of Squares)')

    # Show elbow plot in Streamlit
    st.pyplot(elbow_fig)

    optimal_k = st.sidebar.slider("Select optimal k", min_value=1, max_value=10, value=3)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    geo_data['Cluster'] = kmeans.fit_predict(geo_data_scaled)

    scaled_cluster_centers = kmeans.cluster_centers_
    centers_original_scale = scaler.inverse_transform(scaled_cluster_centers)

    plt.figure(figsize=(15, 8))
    plt.scatter(geo_data['Country'], geo_data['Total_Purchase'], c=geo_data['Cluster'], cmap='viridis')

    country_names = df['Country'].unique()
    plt.xticks(np.arange(len(country_names)), country_names, rotation=45, ha='right')

    plt.title('Geographical Clustering')
    plt.xlabel('Country')
    plt.ylabel('Total_Purchase')
    st.pyplot(plt)

# Function for product-based clustering
def perform_product_clustering(df):
    product_data = df[['Quantity', 'UnitPrice']].dropna()
    scaler = StandardScaler()
    product_data_scaled = scaler.fit_transform(product_data)

    distortions = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(product_data_scaled)
        distortions.append(kmeans.inertia_)

    # Capture elbow plot
    elbow_fig, elbow_ax = plt.subplots(figsize=(10, 6))
    elbow_ax.plot(k_range, distortions, marker='o')
    elbow_ax.set_title('Elbow Method for Optimal k (Product Clustering)')
    elbow_ax.set_xlabel('Number of Clusters (k)')
    elbow_ax.set_ylabel('Distortion (Within-cluster Sum of Squares)')

    # Show elbow plot in Streamlit
    st.pyplot(elbow_fig)

    optimal_k = st.sidebar.slider("Select optimal k", min_value=1, max_value=10, value=3)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    product_data['Cluster'] = kmeans.fit_predict(product_data_scaled)

    scaled_cluster_centers = kmeans.cluster_centers_
    centers_original_scale = scaler.inverse_transform(scaled_cluster_centers)

    plt.figure(figsize=(15, 8))
    plt.scatter(product_data['Quantity'], product_data['UnitPrice'], c=product_data['Cluster'], cmap='viridis')

    for i, center_scaled in enumerate(scaled_cluster_centers):
        center_quantity, center_price = centers_original_scale[i, :]

        plt.scatter(center_quantity, center_price, c='black', marker='o', s=200)
        plt.annotate(f'Center: Quantity={center_quantity:.2f}, Price={center_price:.2f}', 
                     (center_quantity, center_price), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title('Product-based Clustering')
    plt.xlabel('Quantity')
    plt.ylabel('UnitPrice')
    st.pyplot(plt)

# Function for RFM clustering
def calculate_rfm(data):
    rfm_data = data.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (pd.to_datetime('now') - pd.to_datetime(x.max())).days,
        'InvoiceNo': 'nunique',
        'Total_Purchase': 'sum'
    }).reset_index()

    rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    return rfm_data

def perform_rfm_clustering(data):
    rfm_data = data[['Recency', 'Frequency', 'Monetary']].dropna()
    scaler = StandardScaler()
    rfm_data_scaled = scaler.fit_transform(rfm_data)

    distortions = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_data_scaled)
        distortions.append(kmeans.inertia_)

    # Capture elbow plot
    elbow_fig, elbow_ax = plt.subplots(figsize=(10, 6))
    elbow_ax.plot(k_range, distortions, marker='o')
    elbow_ax.set_title('Elbow Method for Optimal k (RFM Clustering)')
    elbow_ax.set_xlabel('Number of Clusters (k)')
    elbow_ax.set_ylabel('Distortion (Within-cluster Sum of Squares)')

    # Show elbow plot in Streamlit
    st.pyplot(elbow_fig)

    optimal_k = st.sidebar.slider("Select optimal k", min_value=1, max_value=10, value=3)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    rfm_data['Cluster'] = kmeans.fit_predict(rfm_data_scaled)

    plt.figure(figsize=(15, 8))
    plt.scatter(rfm_data['Recency'], rfm_data['Monetary'], c=rfm_data['Cluster'], cmap='viridis')
    plt.title('RFM Clustering')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title("Customer Segmentation App")

    # Provide the hardcoded file path
    #file_path = "C:/Users/user/Informatics/Project/praseeda project/dataset.csv"
    file_path = 'dataset.csv'
    
    df = preprocess_data(file_path)

    clustering_type = st.sidebar.selectbox("Select Clustering Type", ["Customer-Based Clustering", "Geographical Clustering", "Product-Based Clustering", "RFM Clustering"])

    if clustering_type == "Customer-Based Clustering":
        perform_customer_clustering(df)
    elif clustering_type == "Geographical Clustering":
        perform_geographical_clustering(df)
    elif clustering_type == "Product-Based Clustering":
        perform_product_clustering(df)
    elif clustering_type == "RFM Clustering":
        rfm_df = calculate_rfm(df)
        perform_rfm_clustering(rfm_df)

if __name__ == "__main__":
    main()
