# Duplicate Question Pairs Detector

An NLP-based web app that detects whether two questions are duplicates, built with scikit-learn and deployed on Streamlit.

🔗 **Live App:** https://h-duplicate-question-pairs.streamlit.app

## About
This project detects whether two questions have the same intent (e.g., "What is the capital of India?" vs "Where is India's capital located?"). It uses TF-IDF vectorization, fuzzy/token/length-based features, and cosine similarity between question vectors, fed into a Random Forest classifier.

## Tech Stack
- Python
- scikit-learn (TF-IDF, Random Forest)
- Streamlit (deployment)
- NLTK / fuzzywuzzy (text preprocessing & fuzzy matching)

## Key Challenge
The initial model was biased toward predicting "Not Duplicate" due to class imbalance in the training data. I fixed this using `class_weight='balanced'` in the Random Forest and tuned the training sample size, improving duplicate-detection recall from ~60% to ~90%.

## Run Locally
1. Clone the repo
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
