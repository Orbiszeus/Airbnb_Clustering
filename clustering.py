from pyspark.ml.clustering import KMeans
import folium
import matplotlib.pyplot as plt
import numpy as np
from math import pi


class AirbnbClustering:
    def __init__(self, k=5):
        self.k = k

    def perform_clustering(self, data):
        # Perform K-Means clustering
        kmeans = KMeans(featuresCol="scaled_features", predictionCol="cluster", k=self.k, seed=1)
        model = kmeans.fit(data)
        clustered_data = model.transform(data)
        return clustered_data

    def visualize_geo_clusters(self, clustered_data):
        try:
            print("Starting to cluster geographically..")
            geo_data = clustered_data.select("latitude", "longitude", "cluster").collect()
            malaga_map = folium.Map(location=[36.7213, -4.4216], zoom_start=12)
            colors = ["red", "blue", "green", "purple", "orange"]

            for row in geo_data:
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=5,
                    color=colors[row["cluster"] % len(colors)],
                    fill=True,
                    fill_opacity=0.7
                ).add_to(malaga_map)

            malaga_map.save("geographical_clusters.html")
            print("Geographical clusters saved as 'geographical_clusters.html'")
        except Exception as e:
            print(f"Error in visualize_geo_clusters: {e}")
    def visualize_by_license(self, data):
        try:
            print("Starting to visualize by license..")
        
            # Collect data for latitude, longitude, and license
            geo_data = data.select("latitude", "longitude", "license").collect()
            malaga_map = folium.Map(location=[36.7213, -4.4216], zoom_start=12)
        
            # Define two colors for licensed (blue) and not licensed (red)
            license_colors = {0: "red", 1: "blue"}

            for row in geo_data:
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=5,
                    color=license_colors[row["license"]],
                    fill=True,
                    fill_color=license_colors[row["license"]],
                    fill_opacity=0.7
                ).add_to(malaga_map)

            malaga_map.save("visualize_by_license.html")
            print("Map visualizing by license saved as 'visualize_by_license.html'")
        except Exception as e:
            print(f"Error in visualize_by_license: {e}")

    def visualize_price_time_clusters(self, clustered_data):
        try:
            print("Starting to cluster for price/time..")
            price_time_data = clustered_data.select("price", "host_since_days", "cluster").toPandas()

            plt.figure(figsize=(10, 6))
            plt.scatter(
                price_time_data["host_since_days"],
                price_time_data["price"],
                c=price_time_data["cluster"],
                cmap="viridis",
                alpha=0.7
            )
            plt.colorbar(label="Cluster")
            plt.xlabel("Host Since Days")
            plt.ylabel("Price")
            plt.title("Price-Time Clustering")
            plt.show()
        except Exception as e:
            print(f"Error in visualize_price_time_clusters: {e}")

    def analyze_centroids(self, model, feature_col):
        """
        Analyze and visualize the centroids of clusters.
        :param model: Trained KMeans model.
        :param feature_col: List of feature column names.
        """
        centroids = model.clusterCenters()
        num_features = len(feature_col)
        num_clusters = len(centroids)

        print("Cluster Centroids:")
        for idx, centroid in enumerate(centroids):
            print(f"Cluster {idx}: {centroid}")

        # Visualize using bar plots
        for idx, centroid in enumerate(centroids):
            plt.figure(figsize=(10, 6))
            plt.bar(np.arange(num_features), centroid, tick_label=feature_col, alpha=0.7)
            plt.title(f"Cluster {idx} Feature Importance")
            plt.xlabel("Features")
            plt.ylabel("Value")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    def visualize_centroids_radar(self, model, full_feature_col, selected_features):
        """
        Visualize centroids using a radar chart with a subset of features.
        :param model: Trained KMeans model.
        :param full_feature_col: List of all feature column names used in clustering.
        :param selected_features: List of feature column names to include in the radar chart.
        """
        # Get indices of the selected features
        selected_indices = [full_feature_col.index(f) for f in selected_features]

        # Extract centroids and filter for selected features
        centroids = model.clusterCenters()
        filtered_centroids = [[centroid[i] for i in selected_indices] for centroid in centroids]

        num_features = len(selected_features)
        num_clusters = len(filtered_centroids)

        # Set up the radar chart
        categories = selected_features
        angles = np.linspace(0, 2 * pi, num_features, endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        plt.figure(figsize=(8, 8))

        for idx, centroid in enumerate(filtered_centroids):
            values = centroid
            values += values[:1]  # Close the circle for the radar plot

            plt.polar(angles, values, label=f"Cluster {idx}")
            plt.fill(angles, values, alpha=0.25)

        plt.xticks(angles[:-1], categories, color="grey", size=10)
        plt.yticks(color="grey", size=8)
        plt.title("Cluster Centroid Analysis - Radar Chart", size=14, y=1.1)
        plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        plt.show()



