import matplotlib.pyplot as plt
import numpy as np
from math import pi
from sklearn.metrics import silhouette_samples, silhouette_score

class Evaluation:
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

    def plot_silhouette(self, features, labels, silhouette_avg, sample_silhouette_values):
        n_clusters = len(np.unique(labels))
        y_lower = 10
        plt.figure(figsize=(10, 6))
        
        for i in range(n_clusters):
            # Aggregate silhouette scores for each cluster
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            plt.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # Add spacing between clusters

        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.title("Silhouette Plot for Clusters")
        plt.xlabel("Silhouette Coefficient")
        plt.ylabel("Cluster")
        plt.yticks([])
        plt.tight_layout()
        plt.show()

