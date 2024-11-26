from services.clustering import AirbnbClustering
from services.data_preperation import CleaningData
from config import FULL_FEATURE_COL
from pyspark.ml.clustering import KMeans
from services.evaluation import Evaluation
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np

selected_features = ["price", "latitude", "longitude", "license"] #those are the selected features by me
 
# Initialize and preprocess data
cleaner = CleaningData()
data = cleaner.preprocess_data()
evaluator = Evaluation()
# Perform clustering
clustering = AirbnbClustering(k=4)
clustered_data = clustering.perform_clustering(data)

# Visualize clusters
clustering.visualize_geo_clusters(clustered_data)
clustering.visualize_price_time_clusters(clustered_data)
clustering.visualize_by_license(clustered_data)
clustering.visualize_by_room_type(clustered_data)

geo_kmeans = KMeans(featuresCol="features", predictionCol="geo_cluster", k=5, seed=1)
geo_model = geo_kmeans.fit(clustered_data)

# evaluator.analyze_centroids(geo_model, selected_features)
evaluator.visualize_centroids_radar(geo_model, FULL_FEATURE_COL, selected_features)

# Extract feature matrix and cluster labels for silhouette calculation
selected_data = clustered_data.select("latitude", "longitude", "price", "room_type", "cluster")
selected_data.show()

# Extract features and labels for Silhouette Evaluation
features = (
    clustered_data.select("features")
    .rdd.map(lambda row: row.features.toArray())  # Convert Spark DenseVector to numpy array
    .collect()
)
labels = (
    clustered_data.select("cluster")
    .rdd.flatMap(lambda row: row)
    .collect()
)

# Convert to numpy array for scikit-learn compatibility
features = np.array(features)
labels = np.array(labels)

# Evaluate Silhouette Score
silhouette_avg = silhouette_score(features, labels)
sample_silhouette_values = silhouette_samples(features, labels)

# Optionally, join features and labels with the selected columns for further inspection
selected_data_with_silhouette = selected_data.toPandas()  # Convert PySpark DataFrame to pandas DataFrame
selected_data_with_silhouette["silhouette"] = sample_silhouette_values

# Plot Silhouette
evaluator.plot_silhouette(features, labels, silhouette_avg, sample_silhouette_values)
print(f"Average Silhouette Score: {silhouette_avg}")

# Show updated DataFrame with silhouette values
print(selected_data_with_silhouette.head())  # Display the first few rows for review