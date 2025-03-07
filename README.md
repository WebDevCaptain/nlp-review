# Natural Language Processing

This is a repo for reviewing basics of NLP.

## Contents

1. [Basics](./1-nlp_basics.ipynb)

   - Tokenization
   - Part of Speech (POS) tagging and Parse tree
   - Lemmatization
   - Stemming

2. [Text Preprocessing](./2-text_preprocessing.ipynb)

   - Stopword removal
   - Regexp tokenizer
   - Data cleaning

3. [Named Entity Recognition](./3-named_entity_recognition.ipynb)

   - Information Extraction and NER (Named Entity Recognition)
   - Using Spacy pipeline
   - Web Scraping
   - Visualizing Named Entities using `displacy`

4. [Sentiment Analysis](./4-sentiment_analysis.ipynb)

   - Exploratory Data Analysis
   - Using Spacy for training a sentiment analysis model (using custom data)
   - Model evaluation and persistence

5. [Text Summarization](./5-extraction_based_summarization.ipynb)

   - Scraping Wikipedia API for articles
   - TF-IDF based summarization using Scikit-learn (Non-gramatical summaries 😭)
   - TextRank based summarization using Sumy library.

6. [Topic Modelling (NMF)](./6-topic_modelling.ipynb)

   - Using NMF(Non-negative Matrix Factorization) for Topic Modelling
   - Unsupervised text classification
   - WordCloud for visualizing topics
   - Using NMF model with Tf-Idf vectorizer
   - [TODO]: Try LDA (Latent Dirichlet Allocation) for Topic Modelling

---

## Extra Content

1. Speech Recognition
   - Perform _speech to text_ using **Google's speech recognition engine** and `OpenAI's Whisper` models.
   - `Librosa` is used for audio processing
   - `SpeechRecognition` package is used for speech recognition.
