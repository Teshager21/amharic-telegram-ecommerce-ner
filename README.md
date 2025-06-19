# Amharic Telegram E-commerce NER

🚀 **Amharic Telegram E-commerce NER** is a complete end-to-end NLP pipeline for extracting structured product data from Amharic-language Telegram-based e-commerce channels. It powers the EthioMart platform by fine-tuning transformer-based models to recognize key business entities like **Product Names**, **Prices**, and **Locations** in unstructured messages.

---

## 📌 Project Goals

- 📥 Ingest real-time text and image data from multiple Amharic Telegram e-commerce channels.
- 🧼 Preprocess and structure raw Amharic text data for downstream NLP tasks.
- 🧠 Fine-tune multilingual transformer models (e.g., XLM-R, mBERT) for **Named Entity Recognition (NER)** in Amharic.
- 🧪 Compare model performance using F1-score, precision, recall.
- 🔍 Interpret predictions with SHAP and LIME to ensure transparency.
- 📊 Score vendors using engagement metrics + extracted business info for micro-lending insights.

---

## 🛠️ Tech Stack

| Layer        | Tools / Libraries                             |
|-------------|------------------------------------------------|
| Data Ingestion | `telethon`, `tdlib` for Telegram scraping     |
| NLP & NER    | `transformers`, `datasets`, `seqeval`, `XLM-R` |
| Tracking     | `MLflow`, `DVC`                               |
| Annotation   | `CoNLL`, manual labeling, `doccano` (optional) |
| Explainability | `SHAP`, `LIME`                                |
| Deployment   | Google Colab, local, or cloud                  |

---

## 📦 Features

- 🐦 **Real-time Telegram data extraction**
- 🔠 **Amharic text normalization & tokenization**
- 🏷️ **Custom NER dataset in CoNLL format**
- 🤖 **Fine-tuned transformer models for NER**
- 📈 **Evaluation & model comparison dashboards**
- 💡 **Interpretable outputs using SHAP/LIME**
- 💰 **Vendor scorecard engine for micro-lending insights**

---

## 🗂 Project Structure

amharic-telegram-ecommerce-ner/
├── data/
│ ├── raw/
│ ├── processed/
├── src/
│ ├── ingestion/ # Telegram scraping logic
│ ├── training/ # Model training + MLflow
│ ├── evaluation/ # Model comparison, SHAP/LIME
│ └── scorecard/ # Vendor scoring engine
├── notebooks/ # Exploratory work
├── models/ # Saved models (DVC tracked)
├── dvc.yaml # DVC pipeline
├── mlruns/ # MLflow experiment logs
└── README.md


---

## 🔬 Example Entities Extracted

| Entity Type | Example (Amharic)              | Translated        |
|-------------|-------------------------------|-------------------|
| Product     | የህፃናት ጫማ                   | Baby shoes        |
| Price       | ዋጋ - 1500 ብር                | Price - 1500 ETB  |
| Location    | ቦሌ አደባባይ                    | Bole Square       |

---

## 📈 Vendor Scorecard Metrics

For each vendor:
- 🧮 Avg. Posts per Week
- 👀 Avg. Views per Post
- 💸 Avg. Product Price
- 🌟 Lending Score = (0.5 × Views) + (0.5 × Frequency)

---

## 📅 Timeline

| Task                            | Status |
|---------------------------------|--------|
| ✅ Task 1: Telegram Ingestion   | In Progress |
| ✅ Task 2: NER Labeling (CoNLL) | Pending |
| ✅ Task 3: Model Fine-Tuning    | Pending |
| ✅ Task 4: Model Comparison     | Pending |
| ✅ Task 5: Interpretability     | Pending |
| ✅ Task 6: Vendor Scorecard     | Pending |

---

## 🤝 Team & Mentors

- 👨‍💻 Project Lead: Teshager Admasu
- 🧑‍🏫 Mentors: Mahlet, Rediet, Kerod, Rehmet

---

## 📄 License

[MIT License](LICENSE)

---

## ⭐️ Give us a star if this repo helps you!
