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
        """
        Visualize clusters geographically using latitude and longitude.
        """
        try:
            print("Clustering based on latitude and longitude...")
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

            malaga_map.save("frontend/geographical_clusters.html")
            print("Updated geographical clusters saved as 'geo_clusters_updated.html'")
        except Exception as e:
            print(f"Error in visualize_geo_clusters: {e}")

    def visualize_by_room_type(self, data):
        """
        Visualize clusters based on room_type.
        """
        try:
            print("Clustering by room type...")
            room_type_colors = {
                "Entire home/apt": "blue",
                "Private room": "green",
                "Shared room": "red",
                "Hotel room": "orange"
            }

            room_type_data = data.select("latitude", "longitude", "room_type").collect()
            malaga_map = folium.Map(location=[36.7213, -4.4216], zoom_start=12)

            for row in room_type_data:
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=5,
                    color=room_type_colors.get(row["room_type"], "gray"),
                    fill=True,
                    fill_opacity=0.7
                ).add_to(malaga_map)

            malaga_map.save("frontend/room_type_clusters.html")
            print("Map visualizing room types saved as 'room_type_clusters.html'")
        except Exception as e:
            print(f"Error in visualize_by_room_type: {e}")
        
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

            malaga_map.save("frontend/visualize_by_license.html")
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

    

