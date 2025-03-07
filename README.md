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

   - Using NMF(Non-negative Matrix Factorization) suggest topics for BBC news articles
   - Unsupervised text classification
   - WordCloud for visualizing topics
   - Using NMF model with Tf-Idf vectorizer
   - [TODO]: Try LDA (Latent Dirichlet Allocation) for Topic Modelling

7. [Recommendation Systems](./7-recommendation_systems.ipynb)

   - Word2Vec model from Gensim
   - Transfer Learning using Word2Vec and Google News weights
   - Netflix recommendation system based on what you watched (using cosine similarity from Scikit learn)

8. [Fake News Detection](./8-fake_news_detection.ipynb)

   - LSTM based neural network for fake news detection (using Tensorflow and Keras)
   - Using custom dataset for training a Deep Neural Network for NLP
   - Data preprocessing and cleaning
   - Binary classification (Fake or Real news)
   - One-hot encoding of features using Keras preprocessing utility
   - Word embeddings

---

## Extra Content

1. Speech Recognition
   - Perform _speech to text_ using **Google's speech recognition engine** and `OpenAI's Whisper` models.
   - `Librosa` is used for audio processing
   - `SpeechRecognition` package is used for speech recognition.

---

## Libraries Used

1. NLTK
2. Pandas
3. Matplotlib
4. Seaborn
5. Scikit-learn
6. Tensorflow
7. Spacy
8. Gensim
9. Sumy
10. OpenAI Whisper
11. SpeechRecognition
12. WordCloud
13. Librosa
