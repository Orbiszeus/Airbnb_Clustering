from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, to_date, year
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark import SparkConf, SparkContext


class CleaningData:

    def __init__(self):
        self.spark = SparkSession.builder.appName("AirbnbClustering").getOrCreate()
        # Set log level to suppress verbose output
        self.spark.sparkContext.setLogLevel("ERROR")

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

        # Columns to process
        float_columns = [
            "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
            "review_scores_checkin", "review_scores_communication", "review_scores_location",
            "review_scores_value", "latitude", "longitude", "reviews_per_month"
        ]
        
        integer_columns = [
            "calculated_host_listings_count", "calculated_host_listings_count_entire_homes",
            "calculated_host_listings_count_private_rooms", "calculated_host_listings_count_shared_rooms",
            "availability_30", "availability_60", "availability_90", "availability_365",
            "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d"
        ]

        # Clean and cast the 'price' column
        data = data.withColumn("price", regexp_replace(col("price"), "[$,]", "").cast("double"))

        # Handle 'license' column as binary
        if "license" in data.columns:
            data = data.withColumn(
                "license",
                when(col("license") == "Exempt", 0).otherwise(1).cast("integer")
            )

        # Ensure no nulls in numeric columns
        numeric_columns = float_columns + integer_columns + ["price", "license"]
        data = data.fillna({col_name: 0 for col_name in numeric_columns})

        # Vectorize numerical columns for clustering
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
        data = assembler.transform(data)

        # Scale the features
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
        data = scaler.fit(data).transform(data)

        # Show processed data
        data.select("scaled_features").show(5)

        return data
