from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import folium
import matplotlib.pyplot as plt

class GeographicalClustering:
    def __init__(self, k=5):
        self.k = k

    def perform_clustering(self, data):
        # Extract latitude and longitude for clustering
        geo_assembler = VectorAssembler(inputCols=["latitude", "longitude"], outputCol="geo_features")
        geo_data = geo_assembler.transform(data)

        # Perform K-Means clustering
        geo_kmeans = KMeans(featuresCol="geo_features", predictionCol="geo_cluster", k=self.k, seed=1)
        geo_model = geo_kmeans.fit(geo_data)
        geo_clustered = geo_model.transform(geo_data)

        return geo_clustered

    def visualize(self, geo_clustered):
        geo_data = geo_clustered.select("latitude", "longitude", "geo_cluster").collect()

        # Initialize map
        malaga_map = folium.Map(location=[36.7213, -4.4216], zoom_start=12)
        colors = ["red", "blue", "green", "purple", "orange"]

        # Add points
        for row in geo_data:
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=colors[row["geo_cluster"] % len(colors)],
                fill=True,
                fill_opacity=0.7
            ).add_to(malaga_map)

        malaga_map.save("geographical_clusters.html")
        print("Geographical clusters saved as 'geographical_clusters.html'")

class PriceTimeClustering:
    def __init__(self, k=5):
        self.k = k

    def perform_clustering(self, data):
        # Extract price, years_hosting, and license for clustering
        assembler = VectorAssembler(inputCols=["price", "years_hosting", "license"], outputCol="price_time_features")
        price_time_data = assembler.transform(data)

        # Perform K-Means clustering
        price_time_kmeans = KMeans(featuresCol="price_time_features", predictionCol="price_time_cluster", k=self.k, seed=1)
        price_time_model = price_time_kmeans.fit(price_time_data)
        price_time_clustered = price_time_model.transform(price_time_data)

        return price_time_clustered

    def visualize(self, price_time_clustered):
        price_time_data = price_time_clustered.select("price", "years_hosting", "price_time_cluster").toPandas()

        # Scatter plot
        
        plt.figure(figsize=(10, 6))
        plt.scatter(
            price_time_data["years_hosting"],
            price_time_data["price"],
            c=price_time_data["price_time_cluster"],
            cmap="viridis",
            alpha=0.7
        )
        plt.colorbar(label="Cluster")
        plt.xlabel("Years Hosting")
        plt.ylabel("Price")
        plt.title("Price-Time Clustering")
        plt.show()


class ClusteringWithElbow:
    def __init__(self, feature_col, max_k=10):
        self.feature_col = feature_col
        self.max_k = max_k

    def find_optimal_k(self, data):
        # List to store WSSSE (inertia) values for each k
        inertia = []

        # Evaluate inertia for k=2 to max_k
        for k in range(2, self.max_k + 1):
            kmeans = KMeans(featuresCol=self.feature_col, predictionCol="cluster", k=k, seed=1)
            model = kmeans.fit(data)
            wssse = model.summary.trainingCost
            inertia.append((k, wssse))

        # Plot the Elbow Curve
        ks, wssses = zip(*inertia)
        plt.figure(figsize=(8, 5))
        plt.plot(ks, wssses, marker="o")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia (WSSSE)")
        plt.title("Elbow Method for Optimal k")
        plt.grid(True)
        plt.show()
        print("Inertia: ", inertia)
        return inertia

    def perform_clustering(self, data, k):
        # Perform clustering with the optimal k
        kmeans = KMeans(featuresCol=self.feature_col, predictionCol="cluster", k=k, seed=1)
        model = kmeans.fit(data)
        clustered_data = model.transform(data)
        return clustered_data