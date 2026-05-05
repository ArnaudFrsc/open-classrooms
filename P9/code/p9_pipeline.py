"""
P9 — Featurization MobileNetV2 distribuée + PCA sur Fruits-360.
Version script pour spark-submit (équivalent du notebook_aws.ipynb).

Usage local de test :
    python3 p9_pipeline.py s3://mon-bucket/data s3://mon-bucket/features s3://mon-bucket/pca

Usage EMR :
    spark-submit \
        --deploy-mode cluster \
        s3://mon-bucket/code/p9_pipeline.py \
        s3://mon-bucket/data \
        s3://mon-bucket/features \
        s3://mon-bucket/pca
"""

import os
import sys
import io

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split, udf
from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT


def main(path_data: str, path_features: str, path_pca: str):
    # ---- Session Spark ----
    spark = (
        SparkSession.builder
        .appName('P9-Fruits-Distributed')
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', '1024')
        .config('spark.sql.parquet.writeLegacyFormat', 'true')
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel('WARN')

    # ---- 1. Lecture des images ----
    images = (
        spark.read.format('binaryFile')
        .option('pathGlobFilter', '*.jpg')
        .option('recursiveFileLookup', 'true')
        .load(path_data)
    )
    images = images.withColumn('label', element_at(split(images['path'], '/'), -2))
    print(f"[INFO] Nombre d'images : {images.count()}")

    # ---- 2. Modèle + broadcast ----
    base = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    new_model = Model(inputs=base.input, outputs=base.layers[-2].output)
    bc_weights = sc.broadcast(new_model.get_weights())

    def model_fn():
        m = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        for layer in m.layers:
            layer.trainable = False
        nm = Model(inputs=m.input, outputs=m.layers[-2].output)
        nm.set_weights(bc_weights.value)
        return nm

    def preprocess(content):
        img = Image.open(io.BytesIO(content)).resize([224, 224])
        return preprocess_input(img_to_array(img))

    def featurize_series(model, content_series):
        arr = np.stack(content_series.map(preprocess))
        preds = model.predict(arr)
        return pd.Series([p.flatten() for p in preds])

    @pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
    def featurize_udf(content_series_iter):
        model = model_fn()
        for content_series in content_series_iter:
            yield featurize_series(model, content_series)

    # ---- 3. Featurization distribuée ----
    features_df = (
        images.repartition(64)
        .select(col('path'), col('label'), featurize_udf('content').alias('features'))
    )
    features_df.write.mode('overwrite').parquet(path_features)
    print(f'[INFO] Features écrites : {path_features}')

    # ---- 4. PCA ----
    df_feat = spark.read.parquet(path_features)
    to_vector = udf(lambda arr: Vectors.dense(arr), VectorUDT())
    df_vec = df_feat.withColumn('features_vec', to_vector('features'))

    # Recherche du k optimal
    pca_explore = PCA(k=200, inputCol='features_vec', outputCol='pcaFeatures')
    cum = pca_explore.fit(df_vec).explainedVariance.cumsum()
    mask = cum >= 0.90
    k_opt = int(np.argmax(mask) + 1) if mask.any() else 200
    print(f'[INFO] k optimal (90% variance) = {k_opt}')

    # PCA finale + écriture
    pca_final = PCA(k=k_opt, inputCol='features_vec', outputCol='pcaFeatures')
    result = pca_final.fit(df_vec).transform(df_vec).drop('features_vec')
    result.write.mode('overwrite').parquet(path_pca)
    print(f'[INFO] PCA écrite : {path_pca}')

    spark.stop()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: p9_pipeline.py <path_data> <path_features> <path_pca>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
