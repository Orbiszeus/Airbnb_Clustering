from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, to_date, datediff, current_date, avg, sum
from pyspark.ml.feature import VectorAssembler, StandardScaler

class CleaningData:

    def __init__(self):
        self.spark = SparkSession.builder.appName("AirbnbClustering").getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")  # Suppress verbose output

    def preprocess_data(self):
        # Load dataset
        data = self.spark.read.csv(
            "raw_data.csv",
            header=True,
            inferSchema=True,
            multiLine=True,
            escape='"',
            quote='"',
            sep=','
        )

        # Process the 'price' column
        data = data.withColumn("price", regexp_replace(col("price"), "[$,]", "").cast("double"))

        # Handle the 'license' column as binary
        if "license" in data.columns:
            data = data.withColumn(
                "license",
                when(col("license") == "Exempt", 0).otherwise(1).cast("integer")
            )

        # Convert 'host_since' to a numeric column (days since host joined)
        if "host_since" in data.columns:
            data = data.withColumn("host_since_days", datediff(current_date(), to_date(col("host_since"))).cast("integer"))
        else:
            data = data.withColumn("host_since_days", col("host_since").cast("integer"))

        # Fill null values for numeric columns
        float_columns = [
            "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
            "review_scores_checkin", "review_scores_communication", "review_scores_location",
            "review_scores_value", "latitude", "longitude", "reviews_per_month"
        ]
        integer_columns = [
            "calculated_host_listings_count", "calculated_host_listings_count_entire_homes",
            "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms",
            "availability_30", "availability_60", "availability_90", "availability_365",
            "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d", "host_since_days"
        ]
        numeric_columns = float_columns + integer_columns + ["price", "license"]
        data = data.fillna({col_name: 0 for col_name in numeric_columns})

        # Vectorize numerical columns for clustering
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
        data = assembler.transform(data)

        # Scale the features
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        data = scaler.fit(data).transform(data)

        # # Calculate average price
        # avg_price = data.select(avg(col("price"))).first()[0]
        # print(f"Average Price: {int(avg_price)}")
        # # Calculate total listings
        # total_listings = data.count()
        # print(f"Total Listings: {int(total_listings)}")

        # avg_reviews = data.select(avg(col("reviews_per_month"))).first()[0]
        # print(f"Average Reviews: {int(avg_reviews)}")
        return data