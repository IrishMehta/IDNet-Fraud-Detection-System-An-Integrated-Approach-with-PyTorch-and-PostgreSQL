CREATE EXTENSION IF NOT EXISTS plpython3u;

select * from pg_extension;

CREATE OR REPLACE FUNCTION py_add(a integer, b integer)
 RETURNS integer
 AS $$
     return a + b
$$ LANGUAGE plpython3u;


CREATE OR REPLACE FUNCTION reverse_string(s text)
RETURNS text
AS $$
    return s[::-1]
$$ LANGUAGE plpython3u;

SELECT reverse_string('chatgpt');

SELECT py_add(10, 20);

create or replace function batch_predict(imgs_base64 text[])
returns text as $$
	import torch
	import torch.nn as nn
	import torchvision
	from torchvision import transforms, models
	from PIL import Image
	import os
	import time
	import base64
	from io import BytesIO

	if 'loaded_model' not in globals():
		MODEL_PATH = '/tmp/best_model.pth'
		CLASS_NAMES = ['genuine', 'fraud']
		DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
		transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

		def load_model():
			model = models.resnet50(weights=None)
			model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
			model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
			model.to(DEVICE)
			model.eval()
			return model

		loaded_model = load_model()

		def predict_base64_image(base64_string, model):
			image_data = base64.b64decode(base64_string)
			image = Image.open(BytesIO(image_data)).convert('RGB')
			input_tensor = transform(image).unsqueeze(0).to(DEVICE)
			with torch.no_grad():
				output = model(input_tensor)
				pred = torch.argmax(output, dim=1).item()
			return CLASS_NAMES[pred]
		
		start_time = time.time()

		counts = {'genuine': 0, 'fraud': 0}
		for base64_img in imgs_base64:
			pred = predict_base64_image(base64_img, loaded_model)
			counts[pred] += 1
		
		end_time = time.time()
		
		elapsed_time = end_time - start_time
		return f"{counts}, Total time: {elapsed_time:.6f} seconds"
$$ language plpython3u;

SELECT batch_predict(ARRAY(SELECT image_data_base64 FROM out_of_sample_data LIMIT 1000));

drop function predict_row;

create or replace function predict_row(imgs_base64 text)
returns table (pred varchar(100), elapsed_time float) AS $$
	import torch
	import torch.nn as nn
	import torchvision
	from torchvision import transforms, models
	from PIL import Image
	import os
	import time
	import base64
	from io import BytesIO
	import numpy as np

	global loaded_model
	global transform
	global DEVICE
	global CLASS_NAMES

	if 'loaded_model' not in globals():
		MODEL_PATH = '/tmp/best_model.pth'
		CLASS_NAMES = ['genuine', 'fraud']
		DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
		transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

		def load_model():
			model = models.resnet50(weights=None)
			model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
			model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
			model.to(DEVICE)
			model.eval()
			return model

		loaded_model = load_model()

	def predict_base64_image(base64_string, model):
		image_data = base64.b64decode(base64_string)
		image = Image.open(BytesIO(image_data)).convert('RGB')
		input_tensor = transform(image).unsqueeze(0).to(DEVICE)
		with torch.no_grad():
			output = model(input_tensor)
			pred = torch.argmax(output, dim=1).item()
		return CLASS_NAMES[pred]
		
	start_time = time.time()
	pred = predict_base64_image(imgs_base64, loaded_model)
	end_time = time.time()
	elapsed_time = end_time - start_time	
	
	return np.array([pred, float(elapsed_time)]).reshape(1,-1)
$$ language plpython3u;


SELECT * FROM predict_row((SELECT image_base64 FROM images limit 1));



SELECT i.document_id, r.pred, r.elapsed_time
FROM out_of_sample_data i
CROSS JOIN LATERAL predict_row(i.image_data_base64) AS r(pred, elapsed_time)
LIMIT 1000;


WITH inference AS (
SELECT i.*, r.pred, r.elapsed_time,
        CASE WHEN r.pred = 'fraud' THEN 1 ELSE 0 END AS predicted_label
FROM out_of_sample_data i
CROSS JOIN LATERAL predict_row(i.image_data_base64) AS r(pred, elapsed_time)
LIMIT 1000
)
SELECT EXTRACT(YEAR FROM issue_date)::INT AS issue_year,
    COUNT(*) AS fraud_count
FROM inference
WHERE predicted_label = 1
GROUP BY issue_year
ORDER BY issue_year DESC;


WITH inference AS (
  SELECT i.*, r.pred, r.elapsed_time,
         CASE WHEN r.pred = 'fraud' THEN 1 ELSE 0 END AS predicted_label
  FROM out_of_sample_data i
  CROSS JOIN LATERAL predict_row(i.image_data_base64) AS r(pred, elapsed_time)
  LIMIT 1000
)
SELECT RIGHT(address, 5) AS zipcode,
       COUNT(*) AS fraud_count
FROM inference
WHERE predicted_label = 1
GROUP BY zipcode
ORDER BY fraud_count DESC
LIMIT 5;


WITH inference AS (
  SELECT i.*, r.pred, r.elapsed_time,
         CASE WHEN r.pred = 'fraud' THEN 1 ELSE 0 END AS predicted_label
  FROM out_of_sample_data i
  CROSS JOIN LATERAL predict_row(i.image_data_base64) AS r(pred, elapsed_time)
  LIMIT 1000
),
with_ages AS (
  SELECT *,
         EXTRACT(YEAR FROM AGE(CURRENT_DATE, birthday)) AS age
  FROM inference
)
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
FROM with_ages
GROUP BY age_group
ORDER BY age_group;


WITH inference AS (
  SELECT i.*, r.pred, r.elapsed_time,
         CASE WHEN r.pred = 'fraud' THEN 1 ELSE 0 END AS predicted_label
  FROM out_of_sample_data i
  CROSS JOIN LATERAL predict_row(i.image_data_base64) AS r(pred, elapsed_time)
  LIMIT 1000
),
height_weight AS (
  SELECT *,
         CAST(SPLIT_PART(height, '''', 1) AS INT) * 12 +
         CAST(SPLIT_PART(SPLIT_PART(height, '''', 2), '"', 1) AS INT) AS inches,
         CAST(REPLACE(weight, ' lb', '') AS FLOAT) AS pounds
  FROM inference
  WHERE height IS NOT NULL AND weight IS NOT NULL
),
ratios AS (
  SELECT *,
         pounds / NULLIF(inches, 0) AS ratio
  FROM height_weight
)
SELECT COUNT(*) AS outlier_count,
       COUNT(*) FILTER (WHERE predicted_label = 1) AS fraud_outliers,
       COUNT(*) FILTER (WHERE predicted_label = 1) * 100.0 / NULLIF(COUNT(*), 0) AS fraud_outlier_pct
FROM ratios
WHERE ratio < 1.2 OR ratio > 3.5;
