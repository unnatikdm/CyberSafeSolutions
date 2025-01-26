# CyberSafeSolutions
## Spam Detection Model

## Overview

This project implements a spam detection system using the **Multinomial Naive Bayes** algorithm. It preprocesses text data, balances the dataset, and uses **TF-IDF (Term Frequency-Inverse Document Frequency)** for feature extraction. The model is fine-tuned using **GridSearchCV** and evaluated using metrics such as accuracy, precision, recall, and F1-score. A real-time prediction function is also provided to classify messages as spam or legitimate.

---

## Features

- **Text Preprocessing**: Converts text to lowercase, removes punctuation and numbers, and applies stemming and stopword removal.
- **Dataset Balancing**: Upsamples the minority class (spam) to balance the dataset.
- **Feature Extraction**: Uses TF-IDF to convert text data into numerical features.
- **Model Tuning**: Utilizes GridSearchCV to find the best hyperparameters for the Multinomial Naive Bayes model.
- **Real-Time Prediction**: Classifies new messages as spam or legitimate.

---

## Requirements

- Python libraries: `pandas`, `scikit-learn`, `nltk`
- NLTK stopwords dataset (download using `nltk.download('stopwords')`).

---

## Dataset

The dataset (`spam.csv`) contains two columns:
- `label`: Indicates whether a message is spam (`spam`) or legitimate (`ham`).
- `message`: The text content of the message.

---

## Performance Metrics

The best model achieved the following results:
- **Accuracy**: 97.62%
- **Precision (Spam)**: 95.82%
- **Recall (Spam)**: 99.47%
- **F1-Score (Spam)**: 97.61%

### Confusion Matrix

|                | Predicted Ham | Predicted Spam |
|----------------|---------------|----------------|
| **Actual Ham** | 944           | 41             |
| **Actual Spam**| 5             | 940            |

---

## Real-Time Prediction

A function `predict_spam` is provided to classify new messages. For example:

```python
message = "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."
result = predict_spam(message)  # Output: Spam
```

---

