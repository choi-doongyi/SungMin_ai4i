import pandas as pd
import numpy as np

df = pd.read_csv("ai4i2020.csv")


# =========================
# 1. Power 파생변수 생성
# =========================

df["Power [kW]"] = (
    df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60
) / 1000


# =========================
# 2. X, y 설정
# =========================

X = df[
    [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Power [kW]",
    ]
]

y = df["Machine failure"]


# =========================
# 3. Type 인코딩
# =========================

X = pd.get_dummies(X, columns=["Type"], drop_first=True)


# =========================
# 4. 확인
# =========================

print(X.head())

print("\nX shape :", X.shape)
print("y shape :", y.shape)

print("\ny 분포")
print(y.value_counts())


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. train / test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. 모델 생성
model = DecisionTreeClassifier(random_state=42)

# 3. 학습
model.fit(X_train, y_train)

# 4. 예측
pred = model.predict(X_test)

# 5. 평가
print("Confusion Matrix")
print(confusion_matrix(y_test, pred))

print("\nClassification Report")
print(classification_report(y_test, pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1. Random Forest 모델 생성
# =========================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# =========================
# 2. 학습
# =========================

rf_model.fit(X_train, y_train)

# =========================
# 3. 예측
# =========================

rf_pred = rf_model.predict(X_test)

# =========================
# 4. 평가
# =========================

print("Random Forest Confusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_pred))

rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train, y_train)

rf_balanced_pred = rf_model.predict(X_test)
print("Random Forest Confusion Matrix")
print(confusion_matrix(y_test, rf_balanced_pred))

print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_balanced_pred))


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 1. 표준화
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 2. 모델 생성
# =========================

lr_model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)

# =========================
# 3. 학습
# =========================

lr_model.fit(X_train_scaled, y_train)

# =========================
# 4. 예측
# =========================

lr_pred = lr_model.predict(X_test_scaled)

# =========================
# 5. 평가
# =========================

print("Logistic Regression Confusion Matrix")
print(confusion_matrix(y_test, lr_pred))

print("\nLogistic Regression Classification Report")
print(classification_report(y_test, lr_pred))


from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

result = pd.DataFrame(
    {
        "Model": [
            "Decision Tree",
            "Random Forest",
            "Balanced Random Forest",
            "Logistic Regression",
        ],
        "Precision": [
            precision_score(y_test, pred),
            precision_score(y_test, rf_pred),
            precision_score(y_test, rf_balanced_pred),
            precision_score(y_test, lr_pred),
        ],
        "Recall": [
            recall_score(y_test, pred),
            recall_score(y_test, rf_pred),
            recall_score(y_test, rf_balanced_pred),
            recall_score(y_test, lr_pred),
        ],
        "F1-score": [
            f1_score(y_test, pred),
            f1_score(y_test, rf_pred),
            f1_score(y_test, rf_balanced_pred),
            f1_score(y_test, lr_pred),
        ],
    }
)

print(result)
# 제조 고장 예측에서는 단순 정확도보다 고장 탐지 성능이 중요하므로 Precision,
# Recall, F1-score를 함께 비교하였다. Decision Tree는 고장 클래스 기준 Recall 0.72,
# F1-score 0.72로 세 모델 중 가장 균형 잡힌 성능을 보여 최종 모델로 선정하였다.


# Evaluation

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, pred)

plt.title("Decision Tree Confusion Matrix")
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall   :", recall_score(y_test, pred))
print("F1-score :", f1_score(y_test, pred))


import pandas as pd

importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(
    ascending=False
)

print(importance)

importance.plot(kind="bar", figsize=(10, 5))

plt.ylabel("Importance")
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.show()


# 본 프로젝트에서는 AI4I 데이터를 활용하여 설비 고장 예측을 수행하였다.
# EDA 과정에서 Rotational speed와 Torque가 고장 여부와 관련된 패턴을 보였으며,
# 두 변수의 물리적 관계를 반영해 Power 파생변수를 생성하였다. 여러 분류 모델을 비교한 결과
# Decision Tree가 고장 클래스의 Precision과 Recall 측면에서 가장 균형 잡힌 성능을 보여 최종 모델로 선정하였다.
# 향후에는 교차검증, 하이퍼파라미터 튜닝, 불균형 데이터 처리 등을 통해 성능을 개선할 수 있다.
