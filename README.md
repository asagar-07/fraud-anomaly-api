Fraud Detection & Anomaly Scoring API (Project Spec)

1. Project Overview

This project implements a production-style fraud detection service for financial transactions using deep learning (PyTorch).
It combines:
	•	A supervised binary classifier to estimate fraud probability
	•	An unsupervised anomaly detection model to surface unusual transaction behavior

Both models are served via a containerized FastAPI service, designed with enterprise ML practices: reproducibility, monitoring, drift checks, and clear decision policies.

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
	•	POST /batch_predict – batch inference

Observability
	•	GET /metrics – latency, request counts, decision rates
	•	POST /feedback (optional) – delayed ground truth ingestion

Example /predict Response

{
  "fraud_probability": 0.87,
  "anomaly_score": 1.42,
  "is_fraud": true,
  "is_anomalous": true,
  "decision": "BLOCK"
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