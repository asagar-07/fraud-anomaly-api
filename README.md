Fraud Detection & Anomaly Scoring API (Project Spec)

1. Project Overview

This project implements a production-style fraud detection service for financial transactions using deep learning (PyTorch).
It combines:
	•	A supervised binary classifier to estimate fraud probability
	•	An unsupervised anomaly detection model to surface unusual transaction behavior

Both models are served via a containerized FastAPI service, designed with enterprise ML practices: reproducibility, monitoring, drift checks, and clear decision policies
with strict validation, batch inference, Dockerized deployment and CI-backed testing.

⸻

2. Problem Statement

Financial fraud is rare, highly imbalanced, and costly. Relying on accuracy alone is misleading.
This system focuses on:
	•	Maximizing fraud capture under strict false-positive constraints
	•	Detecting novel or unknown fraud patterns
	•	Providing actionable decisions (ALLOW / REVIEW / BLOCK) rather than raw scores

⸻

3. Dataset

Credit Card Fraud Detection (European cardholders)
Source: Kaggle
	•	~285K transactions
	•	~0.17% fraud rate
	•	Fully numeric, PCA-transformed features (V1–V28)
	•	Additional features: Time, Amount
	•	Binary label: fraud / non-fraud

Note: Features are anonymized; the project emphasizes ML engineering and system design over business interpretability.

⸻

4. Modeling Approach

4.1 Fraud Classifier (Primary Model)
	•	Type: Deep neural network (MLP) in PyTorch
	•	Objective: Binary classification (fraud probability)
	•	Imbalance handling: Class-weighted loss
	•	Output: fraud_probability ∈ [0,1]

4.2 Anomaly Detection (Secondary Model)
	•	Type: Autoencoder (unsupervised)
	•	Training data: Predominantly non-fraud transactions
	•	Signal: Reconstruction error → anomaly_score
	•	Purpose: Detect unusual patterns not confidently flagged by the classifier

⸻

5. Decision Policy

Model outputs are combined into an operational decision:

Condition	Action
High fraud probability	BLOCK
Medium fraud probability OR high anomaly score	REVIEW
Low risk	ALLOW

Thresholds are configurable artifacts, not hard-coded logic.

⸻

6. API Design

Core Endpoints
	•	GET /health – service readiness & model load status
	•	GET /version – model version, training timestamp, artifact hash
	•	POST /predict – single transaction inference
	•	POST /batch:predict – batch inference

Observability
	•	GET /metrics – latency, request counts, decision rates
	•	POST /feedback (optional) – delayed ground truth ingestion

Example /predict Response

{
  "fraud_probability": 0.87,
  "anomaly_score": 1.42,
  "is_fraud": true,
  "is_anomalous": true,
  "decision": "BLOCK",
  ---more---
}


⸻

7. Evaluation Metrics

Classifier
	•	PR-AUC (primary metric)
	•	Recall @ fixed precision
	•	Precision @ fixed recall
	•	Confusion matrix at decision thresholds
	•	Optional cost-based evaluation (FP vs FN cost)

Anomaly Model
	•	AUROC and PR-AUC (score vs label)
	•	Recall at low false-positive rate
	•	Top-K fraud capture (e.g., top 0.5% anomaly scores)

Service Metrics
	•	p50 / p95 / p99 latency
	•	Throughput (req/sec)
	•	Error rate
	•	% transactions flagged over time

⸻

8. Enterprise ML Practices
	•	Time-based train/validation/test split
	•	Reproducible training (config + fixed seeds)
	•	Versioned model artifacts (weights, scalers, thresholds, metrics)
	•	Schema validation & input sanity checks
	•	Structured logging with request IDs
	•	Simple data drift detection (PSI / KS tests)
	•	Containerized deployment (Docker)

⸻

9. Deployment
	•	Framework: FastAPI
	•	Packaging: Docker + docker-compose
	•	Artifacts: Loaded at startup (fail-fast if missing)
	•	Config: Environment-driven (dev / stage / prod)

⸻

10. Intended Use & Limitations

Intended use
	•	Educational and portfolio demonstration of ML engineering for fraud systems
	•	Reference architecture for classifier + anomaly hybrid pipelines

Limitations
	•	Anonymized PCA features limit interpretability
	•	No real-time label feedback loop
	•	Dataset reflects historical card-present transactions only

⸻

11. Success Criteria
	•	Demonstrates correct handling of extreme class imbalance
	•	Produces stable, reproducible model artifacts
	•	Serves predictions reliably via API
	•	Applies realistic fraud decision logic
	•	Includes monitoring and drift awareness


-------------
Dataset: Credit Card Fraud Detection (European cardholders)
Source: Kaggle
Target variable: Class (0 = Legitimate, 1 = Fraud)

The dataset contains fully numeric, anonymized transaction features generated via PCA, along with transaction timing and amount.

-------------
Feature		Type			Description																						Comments
Time		float			Relative seconds elapsed between this transaction and the first transaction in the dataset		Continuous numeric, preserve ordering for time-based splits
Amount		float			Transaction amount (raw monetary value)															Continuous numeric, highly right-skewed, tranformed using log1p during preprocess
V1 – V28	float			Anonymized PCA-transformed features representing transaction characteristics					continuous numeric, treated as independent numerical inputs
Class		int (binary)	Target label: 0 = legitimate, 1 = fraud															Binary, extremely impalanced (~0.17% fraud), supervised classification and evaluation of anomaly scores

------------
Data Characteristics
	•	Total transactions: ~284,807
	•	Fraud rate: ~0.17%
	•	Missing values: None
	•	Categorical features: None
-----------
Modeling Considerations
	•	Accuracy is not a meaningful metric due to class imbalance
	•	Precision, recall, and PR-AUC are emphasized
	•	Time-based splits are used to prevent data leakage
	•	Feature anonymity limits interpretability but is suitable for ML engineering demonstrations	

-----------
Canonical Feature Order (Model Input)

This model expect features in the following orer after preprocessing:
[Time, Amount, V1, V2, V3, ....., V28]

--------------------------
API CONTRACT

Endpoint
POST /predict

Predicts fraud risk and anomaly score for a single transaction.

REQUEST SCHEMA (INPUT)

Content Type: application/json
Requried Fields: All input features must be numeric and pre-scaled in the same order as training.

Input/Request Sample:
{
  "transaction_id": "string (optional)",
  "features": [
    0.0,
    149.62,
    -1.359807,
    -0.072781,
    2.536346,
    1.378155,
    -0.338321,
    0.462388,
    0.239599,
    0.098698,
    -0.222995,
    -0.166975,
    0.622387,
    -0.066084,
    -0.580558,
    0.035386,
    -0.373003,
    0.345515,
    -0.291090,
    -0.257360,
    0.753074,
    -0.022761,
    0.124458,
    -0.201823,
    0.240279,
    -0.304215,
    0.368090,
    -0.176843,
    0.110507,
    0.041291
  ]
}

Field				Type			Required	Description
transaction_id		string			No			Optional identifier returned in response
features			array[float]	Yes			Ordered numerical feature vector

-------------------

Feature Ordering:
The features array must follow this exact order:

	[Time, Amount, V1, V2, V3, ....., V28]

Requests with incorrect length or non-numeric values will be rejected.


RESPONSE SCEHMA (Output):

Successful Response (200 OK)
{
  "transaction_id": "xyz-123",
  "fraud_probability": 0.88,
  "anomaly_score": 1.45,
  "is_fraud": true,
  "is_anomalous": true,
  "decision": "BLOCK"
}

Field				Type		Description
transaction_id		string		Echoed from request if provided
fraud_probability	float		Probability from supervised classifier
anomaly_score		float		Reconstruction error from autoencoder
is_fraud			boolean		Classifier decision based on threshold
is_anomalous		boolean		Anomaly decision based on AE threshold
decision			string		One of ALLOW, REVIEW, BLOCK

----------------------------------------
DECISION LOGIC:

Final decision is derived as:
	•	BLOCK if (fraud_probability >= fraud_block_threshold) is True
	•	REVIEW if (fraud_probability >= fraud_review_threshold OR anomaly_score >= anomaly_review_threshold) is True
	•	ALLOW  Otherwise

Thresholds are configurable and stored as model artifacts.

-------------------------------

Error Responses:

Example 1:
Invalid Request (422 Unprocessable Entity)
{
  "detail": "Invalid feature vector length or non-numeric values"
}

Example 2:
Service Not Ready (503 Service Unavailable)
{
  "detail": "Model artifacts not loaded"
}

Input Validation Rules:
	•	Feature array length must be exactly 30
	•	All values must be numeric (float-compatible)
	•	NaNs and infinities are rejected
	•	Payload size limits enforced

------------------------------
Time-Based Split Strategy

To simulate real-world fraud detection, data is split chronologically so that model is trained on past transactions only and evaluated on future transactions.

Split Method:
	1. Data is pre-sorted by Time feature (ascending)
	2. Split sequentially into;
		•	Training set: first 70% of transactions
		•	Validation set: next 15% of transactions
		•	Test set: final 15% of transactions
		
Class Distribution Policy
	•	Class imbalance (~0.17% fraud) is preserved in validation and test sets
	•	Any imbalance handling (class-weighted loss) is applied only to the training set
	•	No oversampling, undersampling, or synthetic data generation is performed on validation or test data		
	•	Fraud samples are up-weighted relative to legitimate transactions
	•	Preserves real-world distributions while improving fraud RECALL

Leakage Prevention Rules
	•	No data from validation or test periods is used during training
	•	Feature scalers are fit only on training data
	•	All preprocessing steps are applied consistently across splits using saved artifacts

--------------------------------

12. Development Workflow (Step-by-Step)

This project was built incrementally using a production-style ML engineering workflow rather than a single notebook.

Day 1 – Data Ingestion & Validation
	•	Loaded raw dataset from data/raw/creditcard.csv
	•	Verified:
	•	expected columns and types
	•	no missing values
	•	duplicate rows (deduplicated)
	•	class imbalance (~0.17% fraud)
	•	Defined canonical schema assumptions:
	•	target column
	•	feature list
	•	feature order

Artifacts produced:
	•	dataset statistics
	•	schema assumptions embedded in code

⸻

Day 2 – Preprocessing & Time-Based Splits
	•	Applied preprocessing:
	•	log1p transform for skewed Amount
	•	feature scaling (StandardScaler)
	•	Enforced guardrails:
	•	scaler fit only on training data
	•	validation/test never rebalanced
	•	Implemented time-based split:
	•	70% train / 15% val / 15% test
	•	chronological ordering by Time

Artifacts produced:
	•	processed train/val/test datasets
	•	scaler artifact
	•	split statistics JSON

⸻

Day 3 – Supervised Fraud Classifier
	•	Implemented PyTorch MLP classifier
	•	Used class-weighted loss to address imbalance
	•	Tracked:
	•	PR-AUC
	•	ROC-AUC
	•	recall, precision, F1 at fixed threshold
	•	Saved versioned artifacts:
	•	model.pt
	•	model_config.json
	•	metrics.json
	•	threshold.json

⸻

Day 4 – Unsupervised Anomaly Detection
	•	Trained autoencoder on predominantly non-fraud data
	•	Used reconstruction error as anomaly score
	•	Derived percentile-based thresholds (p95, p99, p995)
	•	Stored reconstruction statistics as artifacts

⸻

Day 5 – Unified Fraud Scoring Service
	•	Combined classifier + autoencoder into a single decision engine
	•	Implemented:
	•	consistent output contract
	•	timing instrumentation
	•	decision flags + reasons
	•	Added smoke tests for inference correctness

⸻

Day 6 – FastAPI Service
	•	Built a production-style API:
	•	/health
	•	/version
	•	/predict
	•	/predict:batch
	•	Features:
	•	strict schema validation
	•	centralized error handling
	•	per-item batch error isolation
	•	startup fail-fast if artifacts missing
	•	Verified via curl and Swagger UI

⸻

Day 7 – Docker & CI
	•	Containerized the service using Docker
	•	Added .dockerignore for clean builds
	•	Implemented GitHub Actions CI:
	•	unit tests run by default
	•	integration tests (model-dependent) marked explicitly
	•	Split tests into:
	•	unit tests (schema, error handling)
	•	integration tests (model inference)

⸻-------------------------------------------

13. Testing Strategy

Unit Tests (CI Default)
	•	Schema validation
	•	Error handlers
	•	Request parsing
	•	No model weights required

Run: 
	PYTHONPATH=src pytest -q

Integration Tests (Local Only)
	•	Classifier inference
	•	Autoencoder inference
	•	Unified fraud scoring

Requires trained artifacts.

Run:
	PYTHONPATH=src pytest -q -m integration

⸻-------------------------------------------

14.Running the Service

Local (No Docker)
	PYTHONPATH=src uvicorn fraud.api.app:app --reload --port 8000
Visit:
	http://127.0.0.1:8000/docs
	http://127.0.0.1:8000/health	

Docker ( requires local model artifacts)
	docker build -t fraud-anomaly-api:1.0 -f docker/Dockerfile .
	docker run --rm -p 8000:8000 fraud-anomaly-api:1.0

-----------------------------------------------

15. Repo Structure

.
├── src/
│   └── fraud/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── inference/
│       └── api/
├── artifacts/
│   ├── classifier/
│   ├── autoencoder/
│   └── shared/
├── tests/
│   ├── unit/
│   └── integration/
├── docker/
│   └── Dockerfile
├── .github/workflows/
├── pytest.ini
└── README.md

-------------------------------------

15. Future Enhancements
	•	threshold tuning strategy
	•	drift monitoring (PSI/KS)
	•	action levels (ALLOW/REVIEW/BLOCK)
	•	Prometheus metrics endpoint
	•	docker-compose for monitoring stack