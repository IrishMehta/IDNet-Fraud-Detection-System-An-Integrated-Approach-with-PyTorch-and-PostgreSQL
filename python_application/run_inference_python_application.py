
"""
Connects to PostgreSQL, loads CNN model, runs inference on images
stored in the out_of_sample_data table, records inference results,
and executes analysis queries. Threshold-based prediction logic.
"""
import psycopg2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import time
import logging
import base64
import datetime


# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


# User Configuration
DB_CONFIG = {
    'host': '<host>',
    'port': 6432,
    'dbname': '<db_name>',
    'user': '<db_user>',
    'password': '<db_password>'
}

MODEL_PATH = 'path to CNN model>' # in this case best_model.pth


# Database Helper

def get_db_connection():
    logging.info("Connecting to PostgreSQL database...")
    conn = psycopg2.connect(**DB_CONFIG)
    logging.info("Database connection established.")
    return conn


# PyTorch Model Definition
class FraudClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model():
    logging.info(f"Loading {MODEL_PATH}…")
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    m = FraudClassifier()
    sd = {('model.'+k if not k.startswith('model.') else k): v for k, v in ckpt.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    logging.info("Model ready.")
    return m


# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(model, image_bytes, filename):
    # logging.info(f"Decoding image for {filename}")
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        logging.debug("Raw bytes failed, trying Base64…")
        raw = base64.b64decode(image_bytes)
        img = Image.open(io.BytesIO(raw)).convert("RGB")

    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1).squeeze().tolist()
        prob_class1 = probs[1]

        if prob_class1 >= 0.5:
            label = 1
            confidence = prob_class1
        else:
            label = 0
            confidence = 1 - prob_class1

    return label, confidence


# Inference Table Management
def ensure_inference_table(cursor):
    logging.info("Ensuring inference_results_out_of_sample table exists...")
    cursor.execute("""
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    CREATE TABLE IF NOT EXISTS inference_results_out_of_sample (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        filename TEXT UNIQUE,
        predicted_label INT NOT NULL,
        confidence FLOAT NOT NULL,
        inference_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """)

def update_inference(cursor, filename, label, confidence):
    cursor.execute("""
    INSERT INTO inference_results_out_of_sample (filename, predicted_label, confidence)
    VALUES (%s, %s, %s)
    ON CONFLICT (filename) DO UPDATE
      SET predicted_label = EXCLUDED.predicted_label,
          confidence = EXCLUDED.confidence,
          inference_timestamp = CURRENT_TIMESTAMP;
    """, (filename, label, confidence))


# Inference Batch Run
def run_inference_batch(conn, cursor, model, batch_size):
    logging.info(f"Running inference for batch size: {batch_size}...")
    cursor.execute("SELECT filename, image_data FROM out_of_sample_data ORDER BY filename DESC LIMIT %s;", (batch_size,))
    rows = cursor.fetchall()
    start = time.time()
    for filename, img in rows:
        label, conf = predict_image(model, img.tobytes(), filename)
        update_inference(cursor, filename, label, conf)
    conn.commit()
    elapsed = time.time() - start
    logging.info(f"Batch {batch_size} completed in {elapsed:.4f} seconds.")


# Analysis Queries
def run_analysis_queries(cursor, start_time):
    logging.info("Executing analysis queries...")

    logging.info("Query: Fraud count by issue year")
    cursor.execute("""
        SELECT EXTRACT(YEAR FROM issue_date)::INT AS issue_year,
               COUNT(*) AS fraud_count
        FROM out_of_sample_data d
        JOIN inference_results_out_of_sample r ON d.filename = r.filename
        WHERE r.predicted_label = 1
        GROUP BY issue_year
        ORDER BY issue_year DESC;
    """)
    for row in cursor.fetchall():
        logging.info(row)

    logging.info("Query: Top 5 fraudulent zip codes")
    cursor.execute("""
        SELECT RIGHT(address, 5) AS zipcode,
               COUNT(*) AS fraud_count
        FROM out_of_sample_data d
        JOIN inference_results_out_of_sample r ON d.filename = r.filename
        WHERE r.predicted_label = 1
        GROUP BY zipcode
        ORDER BY fraud_count DESC
        LIMIT 5;
    """)
    for row in cursor.fetchall():
        logging.info(row)

    # 2. Fraud % by Age Group
    logging.info("Query: Fraud % by age group")
    cursor.execute("""
        SELECT
            CASE
                WHEN age < 20 THEN '<20'
                WHEN age BETWEEN 20 AND 29 THEN '20-29'
                WHEN age BETWEEN 30 AND 39 THEN '30-39'
                WHEN age BETWEEN 40 AND 49 THEN '40-49'
                ELSE '50+'
            END AS age_group,
            COUNT(*) FILTER (WHERE predicted_label = 1) * 100.0 / COUNT(*) AS fraud_percentage,
            COUNT(*) AS total
        FROM (
            SELECT d.filename, r.predicted_label,
                EXTRACT(YEAR FROM AGE(CURRENT_DATE, d.birthday)) AS age
            FROM out_of_sample_data d
            JOIN inference_results_out_of_sample r ON d.filename = r.filename
        ) sub
        GROUP BY age_group
        ORDER BY age_group;
    """)

    for row in cursor.fetchall():
        age_group = row[0]
        fraud_pct = float(row[1])  # Converts Decimal → float
        total = row[2]
        logging.info(f"→ Age Group: {age_group}, Fraud %: {fraud_pct:.2f}%, Total: {total}")


    # 3. Implausible Height/Weight Outliers
    logging.info("Query: Height/Weight outlier frauds")
    cursor.execute("""
        SELECT COUNT(*) AS outlier_count,
               COUNT(*) FILTER (WHERE predicted_label = 1) AS fraud_outliers,
               COUNT(*) FILTER (WHERE predicted_label = 1) * 100.0 / NULLIF(COUNT(*), 0) AS fraud_outlier_pct
        FROM (
            SELECT d.filename, r.predicted_label,
                   CAST(SPLIT_PART(height, '''', 1) AS INT) * 12 +
                   CAST(SPLIT_PART(SPLIT_PART(height, '''', 2), '"', 1) AS INT) AS inches,
                   CAST(REPLACE(weight, ' lb', '') AS FLOAT) AS pounds,
                   CAST(REPLACE(weight, ' lb', '') AS FLOAT) /
                       NULLIF(
                           CAST(SPLIT_PART(height, '''', 1) AS INT) * 12 +
                           CAST(SPLIT_PART(SPLIT_PART(height, '''', 2), '"', 1) AS INT),
                           0
                       ) AS ratio
            FROM out_of_sample_data d
            JOIN inference_results_out_of_sample r ON d.filename = r.filename
            WHERE height IS NOT NULL AND weight IS NOT NULL
        ) sub
        WHERE ratio < 1.2 OR ratio > 3.5;
    """)
    for row in cursor.fetchall():
        logging.info(f"→ Height/Weight Outlier Stats: outliers={row[0]}, fraud_outliers={row[1]}, fraud_pct={float(row[2]):.2f}%")

    # 4. Total Execution Time
    total_time = datetime.datetime.now() - start_time
    logging.info(f"Total Execution Time: {total_time}")

def main():
    import datetime
    start_time = datetime.datetime.now()

    conn = get_db_connection()
    cursor = conn.cursor()
    ensure_inference_table(cursor)
    model = load_model()

    for size in [100]:
        run_inference_batch(conn, cursor, model, size)

    run_analysis_queries(cursor, start_time)

    logging.info("All tasks completed, closing connection.")
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
