<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg" alt="YouTube Logo" height="90"/>

# 🎥 YouTube Video Performance Analysis
*Data-driven insights into what makes a YouTube video thrive*

</div>

---

## 📌 **Project Summary**
This project dives deep into **real YouTube analytics** to uncover the secrets behind video success.  
Using **Python, data science, and machine learning**, we analyze engagement patterns, engineer impactful features, and predict **estimated revenue** for videos.

---

## 📂 **Dataset Overview**
📥 **[Download from Google Drive](https://drive.google.com/file/d/10IdRG52VvMnRB6C5-a3_YqMtzOyxQnNR/view?usp=sharing)**  

**Size:** 364 rows × 70+ columns  
**Key Variables:**
- 🎞 **Video Duration**
- 👀 **Views, Likes, Shares, Comments**
- 👥 **Subscribers**
- 💰 **Revenue & Ad Impressions**
- 📈 **Engagement & Audience Retention**
- 🎯 **YouTube Premium Revenue & CTR**

---

## 🎯 **Project Objectives**
- 📊 Explore **engagement & performance trends**
- 🛠 Create **derived metrics** like *Revenue per View* & *Engagement Rate*
- 📉 Visualize **correlations & revenue drivers**
- 🤖 Train an ML model to predict **Estimated Revenue (USD)**

---

## 🛠 **Technology Stack**

| Tool / Library       | Use Case |
|----------------------|----------|
| **Python**           | Core programming |
| **Pandas, NumPy**    | Data wrangling |
| **Matplotlib, Seaborn** | Data visualization |
| **Scikit-learn**     | Model training & evaluation |
| **Jupyter Notebook** | Development environment |
| **Git/GitHub**       | Version control |

---

## 📈 **Insights at a Glance**
- Videos with **high engagement** (likes + comments + shares) **earn more revenue**
- Publishing **time & duration** slightly influence performance
- Loyal audiences with **high CTRs** convert into better monetization

---

## 🧠 **Machine Learning Model**
**Model Used:** Random Forest Regressor  
**Target:** `Estimated Revenue (USD)`

**Performance Metrics:**
| Metric               | Score  |
|----------------------|--------|
| **MSE**              | ~0.45  |
| **R² Score**         | ~0.89  |

**Training Example:**
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


## 🚀 **Running the Project**

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd youtube-video-performance-analysis

# Install dependencies
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook
```

**Run:** `youtube_video_performance_analysis.ipynb`

---

## 📊 **Advanced Model Enhancements**

### 🔹 Feature Engineering

* Likes/View, Shares/View, Comments/View
* Extract **publish hour** & **day of week**
* Categorize **Engagement Level** with quartiles

### 🔹 Hyperparameter Tuning

* Used **GridSearchCV** with 5-fold CV
* Tuned `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`

### 🔹 Model Comparison

| Model                 | R² Score | MSE  |
| --------------------- | -------- | ---- |
| Random Forest (Tuned) | \~0.91   | 0.41 |
| Gradient Boosting     | \~0.89   | 0.44 |
| XGBoost               | \~0.90   | 0.42 |

### 🔹 Cross-Validation

```python
from sklearn.model_selection import cross_val_score
cross_val_score(model, X, y, cv=5, scoring='r2')
```

### 🔹 Model Export

```python
import joblib
joblib.dump(model, "best_youtube_revenue_model.pkl")
```

---

<div align="center">

✨ **Outcome:**
Enhanced accuracy, reduced overfitting, and built a **production-ready ML pipeline** for YouTube revenue prediction.

</div>




