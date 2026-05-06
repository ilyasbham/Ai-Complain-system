🚀 AI Complaint Classification System


📌 Overview

This project is an AI-powered complaint management system that automatically classifies user complaints into categories (IT, Food, Hostel, Academic) and predicts urgency level (Low, Medium, High).

It uses Machine Learning (TF-IDF + Logistic Regression / Random Forest) and provides a full-stack solution with:

🧠 Machine Learning (Python + FastAPI)
☕ Backend (Spring Boot integration optional)
⚛️ Frontend (React UI)
📊 Model evaluation (Accuracy + Confusion Matrix)
🎯 Problem Statement

In universities and organizations, complaints are:

Manually sorted
Delayed in response
Not prioritized correctly

👉 This system solves that by automatically categorizing and prioritizing complaints using AI.

⚙️ Tech Stack
🧠 Machine Learning
Python
Pandas
Scikit-learn
TF-IDF Vectorizer
Logistic Regression
Random Forest
🚀 Backend
FastAPI
⚛️ Frontend
React.js
Axios
📊 Visualization
Seaborn
Matplotlib
🧠 Machine Learning Models

We tested multiple models:

Logistic Regression
Random Forest

👉 Final model is selected automatically based on accuracy.

📊 Features

✔ Complaint classification (IT / Food / Hostel / Academic)
✔ Urgency prediction (Low / Medium / High)
✔ Model comparison (Auto selection)
✔ Accuracy display in UI
✔ Confusion Matrix visualization
✔ Full-stack API integration
✔ Real-time prediction system

📁 Project Structure
complaint-ai/
│
├── ml-service/
│   ├── data.csv
│   ├── train.py
│   ├── app.py (FastAPI)
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── confusion_matrix.png
│
├── backend/ (optional Spring Boot integration)
│
├── frontend/
│   ├── src/
│   ├── App.js
│   └── package.json
🚀 How to Run
1️⃣ ML Backend (FastAPI)
cd ml-service
pip install -r requirements.txt
uvicorn app:app --reload

👉 Runs on:

http://127.0.0.1:8000
2️⃣ Frontend (React)
cd frontend
npm install
npm start

👉 Runs on:

http://localhost:3000
🔌 API Endpoint
📍 Predict Complaint
POST /predict
📥 Request
{
  "text": "wifi not working in dorm"
}
📤 Response
{
  "category": "IT",
  "urgency": "High",
  "model_used": "Logistic Regression",
  "accuracy": {
    "category": 0.92,
    "urgency": 0.88
  }
}
📊 Confusion Matrix

step
frontend- npm start
backend - D:\My projects\Complain\backend\demo>mvnw.cmd spring-boot:run
python -venv\Scripts\activate
        uvicorn app:app --reload
train python only one time 


A confusion matrix is generated to evaluate model performance visually.

Saved as:

confusion_matrix.png
🧠 Key Learning
Text is converted using TF-IDF
Models learn patterns from complaint data
Best model is selected using accuracy comparison
System predicts category + urgency automatically
🔥 Future Improvements
Add BERT / Transformer NLP model
Deploy on cloud (Render / Vercel)
Add admin dashboard
Real-time complaint tracking system
Email notification system
👨‍💻 Author

Developed as a Hackathon Project
Focus: AI + Real-world Problem Solving
