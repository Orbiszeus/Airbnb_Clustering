from data_preperation import CleaningData
from clustering import GeographicalClustering, PriceTimeClustering, ClusteringWithElbow
from pyspark.ml.feature import VectorAssembler

# Step 1: Data Cleaning
cleaner = CleaningData()
cleaned_data = cleaner.preprocess_data()

# Step 2: Geographical Clustering
geo_clustering = ClusteringWithElbow(feature_col="geo_features", max_k=10)

# Extract latitude and longitude for clustering
geo_assembler = VectorAssembler(inputCols=["latitude", "longitude"], outputCol="geo_features")
geo_data = geo_assembler.transform(cleaned_data)

# Find the optimal k for geographical clustering
geo_inertia = geo_clustering.find_optimal_k(geo_data)

# Choose the optimal k (from visual inspection of the elbow curve)
optimal_k_geo = 5  # Replace this with the chosen k from the elbow method

# Perform geographical clustering
geo_clustered_data = geo_clustering.perform_clustering(geo_data, k=optimal_k_geo)

# Visualize geographical clusters
geo_visualizer = GeographicalClustering(k=optimal_k_geo)
geo_visualizer.visualize(geo_clustered_data)

# Step 3: Price-Time Clustering
price_time_clustering = ClusteringWithElbow(feature_col="price_time_features", max_k=10)

# Extract price, years_hosting, and license for clustering
price_time_assembler = VectorAssembler(inputCols=["price", "years_hosting", "license"], outputCol="price_time_features")
price_time_data = price_time_assembler.transform(cleaned_data)

# Find the optimal k for price-time clustering
price_time_inertia = price_time_clustering.find_optimal_k(price_time_data)

# Choose the optimal k (from visual inspection of the elbow curve)
optimal_k_price_time = 4  # Replace this with the chosen k from the elbow method

# Perform price-time clustering
price_time_clustered_data = price_time_clustering.perform_clustering(price_time_data, k=optimal_k_price_time)

# Visualize price-time clusters
price_time_visualizer = PriceTimeClustering(k=optimal_k_price_time)
price_time_visualizer.visualize(price_time_clustered_data)
