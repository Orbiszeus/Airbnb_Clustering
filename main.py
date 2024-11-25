from services.clustering import AirbnbClustering
from services.data_preperation import CleaningData
from config import FULL_FEATURE_COL
from pyspark.ml.clustering import KMeans

selected_features = ["price", "latitude", "longitude", "license"] #those are the selected features by me
 
# Initialize and preprocess data
cleaner = CleaningData()
data = cleaner.preprocess_data()

# Perform clustering
clustering = AirbnbClustering(k=5)
clustered_data = clustering.perform_clustering(data)

# Visualize clusters
clustering.visualize_geo_clusters(clustered_data)
clustering.visualize_price_time_clusters(clustered_data)
clustering.visualize_by_license(clustered_data)

geo_kmeans = KMeans(featuresCol="features", predictionCol="geo_cluster", k=5, seed=1)
geo_model = geo_kmeans.fit(clustered_data)

# clustering.analyze_centroids(geo_model, feature_col)
clustering.visualize_centroids_radar(geo_model, FULL_FEATURE_COL, selected_features)