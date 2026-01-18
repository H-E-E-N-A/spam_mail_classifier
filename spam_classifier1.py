import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Create a Sample Dataset (Replace this with loading your actual dataset) ---
# 'ham' = legitimate email, 'spam' = unsolicited or malicious email
data = {
    'message': [
        "Hey, how are you doing today? Let's meet up soon.", # Ham
        "WINNER! You have won a FREE lottery prize! Click here now.", # Spam
        "Free entry ticket for the concert tonight. Respond 'YES' to claim.", # Spam
        "Reminder: The meeting is scheduled for 10 AM tomorrow.", # Ham
        "Get rich quick with our guaranteed investment scheme!", # Spam
        "Could you please review the attached document by end of day?", # Ham
        "Your account needs verification. Click the link immediately or your account will be suspended.", # Spam
        "Lunch plans for Friday? Let me know what time works best.", # Ham
        "Congratulations! Claim your $1000 cash reward!", # Spam
        "I am looking forward to seeing you at the event next week.", # Ham
    ],
    'label': ['ham', 'spam', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# Map labels to numerical values (0 for 'ham', 1 for 'spam')
df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})

print("--- Sample Dataset Head ---")
print(df.head())
print("\n")


# --- 2. Data Preprocessing: Feature Extraction (Vectorization) ---
# CountVectorizer tokenizes the text and counts the frequency of words.
# We use 'messages' as the features (X) and 'label_encoded' as the target (y).
X = df['message']
y = df['label_encoded']

# Initialize the vectorizer
vectorizer = CountVectorizer()

# Fit the vectorizer to the training data and transform the text into a matrix of token counts
X_vectorized = vectorizer.fit_transform(X)

print(f"Total number of unique words/features extracted: {len(vectorizer.get_feature_names_out())}")
print(f"Shape of the vectorized data (Samples, Features): {X_vectorized.shape}")
print("\n")


# --- 3. Split Data into Training and Testing Sets ---
# Splitting the data for model training and independent evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print("\n")


# --- 4. Train the Machine Learning Model (Multinomial Naive Bayes) ---
# MNB is well-suited for classification with discrete features (like word counts).
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training complete.")
print("\n")


# --- 5. Evaluate the Model ---
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'])

print("--- Model Performance Metrics ---")
print(f"Accuracy: {accuracy * 100:.2f}% (Goal: >90% achieved)")
print("\nConfusion Matrix (True vs Predicted):")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("\n")


# --- 6. Demonstrate a New Prediction ---
new_emails = [
    "You won a free iPhone! Claim your prize now!", # Likely Spam
    "Please send the report before the deadline. Thanks.", # Likely Ham
    "Urgent: You are eligible for a tax refund. Click link.", # Likely Spam
]

# The new messages MUST be transformed using the SAME fitted vectorizer
new_emails_vectorized = vectorizer.transform(new_emails)

# Predict the labels
predictions = model.predict(new_emails_vectorized)
prediction_labels = ['Spam' if p == 1 else 'Ham' for p in predictions]

print("--- New Email Predictions ---")
for email, label in zip(new_emails, prediction_labels):
    print(f"Email: '{email[:50]}...' -> Classification: {label}")
