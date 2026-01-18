import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Dataset ---
data = {
    'message': [
        "Hey, how are you doing today? Let's meet up soon.",
        "WINNER! You have won a FREE lottery prize! Click here now.",
        "Free entry ticket for the concert tonight. Respond 'YES' to claim.",
        "Reminder: The meeting is scheduled for 10 AM tomorrow.",
        "Get rich quick with our guaranteed investment scheme!",
        "Could you please review the attached document by end of day?",
        "Your account needs verification. Click the link immediately or your account will be suspended.",
        "Lunch plans for Friday? Let me know what time works best.",
        "Congratulations! Claim your $1000 cash reward!",
        "I am looking forward to seeing you at the event next week.",
    ],
    'label': ['ham', 'spam', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

# --- 2. Feature Extraction ---
X = df['message']
y = df['label_encoded']
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# --- 3. Training ---
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)

# --- 4. Evaluation ---
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")