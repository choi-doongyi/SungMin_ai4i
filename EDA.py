import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

df = pd.read_csv("ai4i2020.csv")
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# df[features].hist(figsize=(12, 8), bins=30)

# plt.tight_layout()
# plt.show()
# Air/Process temperature는 비교적 제한된 범위에 분포하며, Torque는 정규분포에 가까운 형태를 보였다.
# Rotational speed는 오른쪽 꼬리가 긴 양의 왜도를 보였으며 일부 고회전값이 존재한다.
# Tool wear는 비교적 넓은 구간에 고르게 분포한다.
# 변수별 단위와 스케일 차이가 존재하므로, 거리 기반 또는 선형 모델 적용 시 표준화를 고려한다.

# for col in features:
#     plt.figure(figsize=(6, 4))
#     plt.boxplot(df[col])
#     plt.title(col)
#     plt.ylabel("Value")
#     plt.show()
# 아직 표준화가 진행되지않아 변수마다 값의 단위가 다르기 때문에 따로 봐준다.

# Rotational speed [rpm]
# → 상단 쪽에 이상치 후보가 매우 많음
# → 히스토그램에서 봤던 오른쪽 꼬리가 긴 양의 왜도와 연결돼
# → 대략 1900rpm 이상부터 이상치들 확인됨

# Torque [Nm]
# 위쪽뿐 아니라 아래쪽에도 이상치 후보가 있음
# 중앙값은 약 40Nm 근처고, 양쪽 극단값들이 존재

# Rotational speed는 상위 구간에서 다수의 이상치 후보가 관찰되었으며, 양의 왜도를 보인다.
# Torque는 상·하위 양쪽에서 이상치 후보가 확인되었다.
# 다만 제조 공정 데이터 특성상 실제 이상 운전 상태일 가능성이 있으므로 단순 제거하지 않고 Machine failure와의 관계를 추가 확인한다.


# df.boxplot(column="Rotational speed [rpm]", by="Machine failure", figsize=(6, 5))

# plt.title("Rotational speed by Machine failure")
# plt.suptitle("")
# plt.xlabel("Machine failure")
# plt.ylabel("Rotational speed [rpm]")
# plt.show()

# 고장 데이터는 전반적으로 낮은 회전속도 구간에 더 집중되어 있지만, 일부 고회전 상태에서도 고장이 발생한다.

# df.boxplot(column="Torque [Nm]", by="Machine failure", figsize=(6, 5))

# plt.title("Torque by Machine failure")
# plt.suptitle("")
# plt.xlabel("Machine failure")
# plt.ylabel("Torque [Nm]")
# plt.show()
# Machine failure가 발생한 그룹에서 Torque의 중앙값과 전체 분포가 정상 그룹보다 높게 나타났다. 따라서 높은 Torque가 고장 발생과 연관될 가능성이 있다.


normal = df[df["Machine failure"] == 0]
failure = df[df["Machine failure"] == 1]

# plt.figure(figsize=(8, 6))

# plt.scatter(
#     normal["Rotational speed [rpm]"], normal["Torque [Nm]"], alpha=0.3, label="Normal"
# )

# plt.scatter(
#     failure["Rotational speed [rpm]"],
#     failure["Torque [Nm]"],
#     alpha=0.7,
#     label="Failure",
# )

# plt.xlabel("Rotational speed [rpm]")
# plt.ylabel("Torque [Nm]")
# plt.title("Rotational speed vs Torque by Machine failure")

# plt.legend()
# plt.show()

# Rotational speed와 Torque는 강한 음의 관계를 보이며,
# 고장 데이터는 주로 낮은 회전속도·높은 토크 영역과 높은 회전속도·낮은 토크 영역에 상대적으로 많이 분포한다.
# 따라서 두 변수의 개별 값뿐 아니라 조합된 운전 조건이 고장과 관련될 가능성이 있다.


df["Power [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60

print(df["Power [W]"].describe())
df["Power [kW]"] = df["Power [W]"] / 1000


# df.boxplot(column="Power [kW]", by="Machine failure", figsize=(7, 5))

# plt.title("Power by Machine failure")
# plt.suptitle("")
# plt.xlabel("Machine failure")
# plt.ylabel("Power [kW]")

# plt.show()

normal = df[df["Machine failure"] == 0]
failure = df[df["Machine failure"] == 1]

# plt.figure(figsize=(8, 6))

# plt.scatter(
#     normal["Rotational speed [rpm]"], normal["Power [kW]"], alpha=0.3, label="Normal"
# )

# plt.scatter(
#     failure["Rotational speed [rpm]"], failure["Power [kW]"], alpha=0.7, label="Failure"
# )

# plt.xlabel("Rotational speed [rpm]")
# plt.ylabel("Power [kW]")
# plt.title("Rotational speed vs Power by Machine failure")
# plt.legend()

# plt.show()


# Power 파생변수 분석 결과, 고장군의 중앙값이 정상군보다 높게 나타났으며 일부 매우 낮은 Power 구간에서도 고장이 관찰되었다.
# 따라서 Power 수준과 고장 발생 사이에 비선형적인 관계가 존재할 가능성이 있다.

df.boxplot(column="Power [kW]", by="PWF", figsize=(7, 5))

plt.title("Power by PWF")
plt.suptitle("")
plt.xlabel("PWF")
plt.ylabel("Power [kW]")

plt.show()


print(df.groupby("PWF")["Power [kW]"].describe())


pwf_data = df[df["PWF"] == 1]

print(
    pwf_data[
        ["Rotational speed [rpm]", "Torque [Nm]", "Power [kW]", "Machine failure"]
    ].head(20)
)
