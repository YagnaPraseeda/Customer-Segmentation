# Customer-Segmentation
Streamlit App link: https://customer-segmentation-8tp8y79nncsmbjyhhlyis5.streamlit.app/

## Abstract
The project focused on implementing data clustering techniques for e-commerce customer segmentation. The primary aim was to uncover meaningful patterns within the customer data, enabling targeted strategies for marketing, inventory management, and customer engagement. By employing customer-based clustering, geographical clustering, product-based clustering, and RFM clustering, the project successfully identified distinct customer segments, regional preferences, product groupings, and behavioral patterns. The Elbow Method aided in determining optimal cluster numbers. Stakeholders, including marketing teams, supply chain managers, expansion strategists, and customer support, can now leverage these insights for personalized campaigns, inventory optimization, targeted expansions, and enhanced customer retention. The project showcases the value of data-driven decision-making in tailoring business strategies to meet the diverse needs of e-commerce customers, ultimately fostering improved operational efficiency and customer satisfaction.

## Data Description
### Data Extraction and cleaning:
Load Data: The dataset is loaded using the Pandas library in Python, allowing for efficient data manipulation and analysis.
Duplicate Removal: Duplicate entries are identified and removed to maintain data integrity and prevent redundancy.
Total Purchase Calculation: A new column, 'Total_Purchase,' is created by multiplying the 'Quantity' and 'UnitPrice' columns, providing a measure of the total purchase value for each transaction.
Outlier Removal: Outliers in 'Total_Purchase,' 'Quantity,' and 'UnitPrice' are identified and removed using the Interquartile Range (IQR) method to enhance the accuracy of subsequent analyses.
CustomerID Filtering: Data is filtered to include only customer IDs within a specified range (e.g., between 12000 and 18500), allowing for focused analysis on a specific subset of customers.
### Data Description After Cleaning:
The cleaned dataset includes columns such as 'CustomerID,' 'Quantity,' 'UnitPrice,' 'Total_Purchase,' 'InvoiceDate,' 'Country,' and others. Duplicate entries and outliers have been addressed, and the dataset is refined to meet the requirements of the clustering analyses. The 'Total_Purchase' column serves as a crucial feature for customer-based and RFM clustering, while 'Country' is employed in geographical clustering. The dataset is now prepared for further exploration and application of clustering algorithms to derive meaningful insights for stakeholders.

## Algorithm Description
The web app employs several clustering algorithms to analyze and segment the e-commerce customer data effectively. Each clustering technique is tailored to address specific aspects of customer behavior, providing valuable insights for stakeholders. The main clustering algorithms used in the web app are as follows:

1. **Customer-Based Clustering: K-Means Algorithm**
   **Description:** The K-Means algorithm partitions customers into k clusters based on their total purchase behavior. It minimizes the within-cluster sum of squares, creating distinct groups with similar purchasing patterns.
   **Application:** Useful for marketing teams to tailor strategies for different customer segments.

2. **Geographical Clustering: K-Means Algorithm with Label Encoding**
   **Description:** K-Means clustering is applied after label encoding the 'Country' variable, converting it to numerical values. This technique groups countries with similar purchasing patterns.
   **Application:** Supports geographical expansion strategies by identifying regions with comparable customer behavior.

3. **Product-Based Clustering: K-Means Algorithm**
   **Description:** Utilizes K-Means clustering to group products based on quantity and unit price. This technique helps in understanding product preferences and optimizing inventory management.
   **Application:** Valuable for supply chain and inventory management stakeholders.

4. **RFM Clustering: K-Means Algorithm**
   **Description:** Applies K-Means clustering to Recency, Frequency, and Monetary (RFM) values. Segments customers based on how recently they made a purchase, how frequently they buy, and the monetary value of their transactions.
   **Application:** Provides insights for customer support and retention teams to tailor strategies and enhance customer satisfaction.

5. **Elbow Method for Optimal k:**
   **Description:** Determines the optimal number of clusters (k) for each clustering task. It involves running K-Means with different values of k and identifying the "elbow" point in the distortion plot.
   **Application:** Supports the effectiveness and interpretability of the clustering results.

The combination of these clustering algorithms enables the web app to uncover patterns within the e-commerce customer data, empowering stakeholders to make informed decisions across marketing, inventory management, geographical expansion, and customer retention. The algorithms are implemented using Python and relevant libraries such as scikit-learn for machine learning and data analysis.

## Tools Used
Here is a list of tools used and their purposes:

Pandas: Pandas is used for data manipulation and analysis. It facilitates tasks such as loading, cleaning, and transforming the e-commerce dataset.
NumPy: NumPy is employed for numerical operations and array manipulations, enhancing efficiency in mathematical computations.
Matplotlib and Seaborn: Matplotlib and Seaborn are utilized for data visualization. They help in creating informative plots and graphs to visually represent clustering results, elbow plots, and geographical data.
scikit-learn: scikit-learn is a machine learning library that provides tools for clustering, including the implementation of the K-Means algorithm and the Elbow Method for optimal k determination.
StandardScaler from scikit-learn: StandardScaler is used for standardizing numerical features, ensuring that variables are on the same scale, which is crucial for clustering algorithms like K-Means.
Label Encoding from scikit-learn: Label Encoding is applied to convert categorical variables, such as 'Country,' into numerical format for geographical clustering.
Jupyter Notebooks: Jupyter Notebooks are used for creating an interactive and iterative development environment, allowing for code execution, visualization, and documentation in a single platform.
Elbow Method: The Elbow Method, implemented using code, aids in determining the optimal number of clusters (k) for each clustering task.

These tools collectively provide a comprehensive and efficient framework for data preprocessing, analysis, and presentation, making it possible to derive meaningful insights from the e-commerce customer data and communicate these insights effectively to stakeholders.







