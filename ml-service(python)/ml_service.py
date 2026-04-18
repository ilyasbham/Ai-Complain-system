import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix




# Load dataset
df = pd.read_csv("data.csv")

X = df["text"]
y_category = df["category"]
y_urgency = df["urgency"]

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# FIXED SPLIT (IMPORTANT)
X_train, X_test, y_cat_train, y_cat_test, y_urg_train, y_urg_test = train_test_split(
    X_vec, y_category, y_urgency, test_size=0.2, random_state=42
)

# ================= CATEGORY MODELS =================
lr_cat = LogisticRegression(max_iter=300)
rf_cat = RandomForestClassifier(random_state=42)

lr_cat.fit(X_train, y_cat_train)
rf_cat.fit(X_train, y_cat_train)

acc_lr_cat = accuracy_score(y_cat_test, lr_cat.predict(X_test))
acc_rf_cat = accuracy_score(y_cat_test, rf_cat.predict(X_test))

best_category_model = lr_cat if acc_lr_cat > acc_rf_cat else rf_cat
best_category_name = "Logistic Regression" if acc_lr_cat > acc_rf_cat else "Random Forest"

# ================= URGENCY MODELS =================
lr_urg = LogisticRegression(max_iter=300)
rf_urg = RandomForestClassifier(random_state=42)

lr_urg.fit(X_train, y_urg_train)
rf_urg.fit(X_train, y_urg_train)

acc_lr_urg = accuracy_score(y_urg_test, lr_urg.predict(X_test))
acc_rf_urg = accuracy_score(y_urg_test, rf_urg.predict(X_test))

best_urgency_model = lr_urg if acc_lr_urg > acc_rf_urg else rf_urg
best_urgency_name = "Logistic Regression" if acc_lr_urg > acc_rf_urg else "Random Forest"

# ================= CONFUSION MATRIX =================
y_pred = best_category_model.predict(X_test)

cm = confusion_matrix(y_cat_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Category Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")

# ================= SAVE MODELS =================
joblib.dump(best_category_model, "category_model.pkl")
joblib.dump(best_urgency_model, "urgency_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# SAVE METADATA (IMPORTANT FOR UI)
joblib.dump({
    "category_model": best_category_name,
    "urgency_model": best_urgency_name,
    "acc_cat": float(max(acc_lr_cat, acc_rf_cat)),
    "acc_urg": float(max(acc_lr_urg, acc_rf_urg))
}, "model_info.pkl")




from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model_category = joblib.load("category_model.pkl")
model_urgency = joblib.load("urgency_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
model_info = joblib.load("model_info.pkl")

class Complaint(BaseModel):
    text: str

@app.post("/predict")
def predict(data: Complaint):

    text_vec = vectorizer.transform([data.text])

    if text_vec.sum() == 0:
        return {
            "category": "Unknown",
            "urgency": "Unknown",
            "model_used": "None"
        }

    category = model_category.predict(text_vec)[0]
    urgency = model_urgency.predict(text_vec)[0]

    return {
        "category": category,
        "urgency": urgency,
        "models": {
            "category_model": model_info["category_model"],
            "urgency_model": model_info["urgency_model"]
        },
        "accuracy": {
            "category": model_info["acc_cat"],
            "urgency": model_info["acc_urg"]
        }
    }