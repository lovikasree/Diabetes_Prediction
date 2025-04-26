# Diabetes Prediction using Gaussian Naive Bayes with Flask

This project is a **Diabetes Prediction** web application built using a **Gaussian Naive Bayes** machine learning model.  
The model is trained inside a **Jupyter Notebook** (`app.ipynb`) and can be integrated with a **Flask** frontend for user interaction.

---

## 🚀 Features

- Machine Learning model using **Gaussian Naive Bayes** algorithm
- Trained on a **diabetes dataset** (`diabetes.csv`)
- Flask-based web application (UI files inside `templates/`, static files in `Static/`)
- User-friendly interface for predictions

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Scikit-learn (for model training)
- Flask (for backend)
- HTML/CSS (for frontend)

---

## 📂 Project Structure

```
DIABETES PREDICTION/
├── .ipynb_checkpoints/      # Jupyter auto-saves
├── Static/                  # CSS, images (if any)
├── templates/               # HTML templates (index.html)
├── app.ipynb                # Jupyter Notebook (Training + Flask code)
├── diabetes.csv             # Dataset used for training
```

---

## ⚙️ How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd DIABETES PREDICTION
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   and open `app.ipynb`.

4. **Run the Flask app from a cell inside the notebook.**

5. **Open your browser at:**
   ```
   http://127.0.0.1:5000/
   ```

---

## 🧑‍🧬 Model Training (Overview)

- Dataset: `diabetes.csv`
- Preprocessing steps: (handling missing values, feature scaling if needed)
- Model: **Gaussian Naive Bayes** from `scikit-learn`
- No separate `.pkl` model saved yet; everything runs inside the notebook.

Sample snippet:

```python
from sklearn.naive_bayes import GaussianNB

# Load dataset
import pandas as pd
data = pd.read_csv('diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Model Training
model = GaussianNB()
model.fit(X, y)
```

---

## 📜 Requirements

Example `requirements.txt`:

```
Flask
scikit-learn
pandas
numpy
```

---

## ✨ Future Improvements

- Separate training and Flask serving into different scripts
- Save trained model (`model.pkl`) and load it during inference
- Add better UI design in `templates/`
- Deploy on cloud (like Render, Heroku, or AWS)

---

## 📬 Contact

For any issues or suggestions, feel free to reach out!

