import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv("ai4i2020.csv")

# ==================================================
# 1. 가설 설정
# H0 : 정상군과 고장군의 Torque 평균에는 차이가 없다.
# H1 : 정상군과 고장군의 Torque 평균에는 차이가 있다.
# ==================================================


# 정상군 / 고장군 분리
normal_torque = df[df["Machine failure"] == 0]["Torque [Nm]"]
failure_torque = df[df["Machine failure"] == 1]["Torque [Nm]"]


# ==================================================
# 2. 기초 통계 확인
# ==================================================

print("정상군")
print("개수 :", len(normal_torque))
print("평균 :", normal_torque.mean())
print("표준편차 :", normal_torque.std())

print()

print("고장군")
print("개수 :", len(failure_torque))
print("평균 :", failure_torque.mean())
print("표준편차 :", failure_torque.std())


# ==================================================
# 3. 정규성 검정
# H0 : 정규분포를 따른다.
# H1 : 정규분포를 따르지 않는다.
# ==================================================

normal_shapiro = stats.shapiro(normal_torque)
failure_shapiro = stats.shapiro(failure_torque)

print("\n정규성 검정")
print("정상군 p-value :", normal_shapiro.pvalue)
print("고장군 p-value :", failure_shapiro.pvalue)


# ==================================================
# 4. 등분산성 검정 (Levene)
# H0 : 두 집단의 분산이 같다.
# H1 : 두 집단의 분산이 다르다.
# ==================================================

levene = stats.levene(normal_torque, failure_torque)

print("\n등분산성 검정")
print("p-value :", levene.pvalue)


# ==================================================
# 5. Welch t-test
# 두 집단의 평균 차이 검정
# equal_var=False → 두 집단의 분산이 같다고 가정하지 않음
# ==================================================

t_stat, p_value = stats.ttest_ind(normal_torque, failure_torque, equal_var=False)

print("\nWelch t-test")
print("t-statistic :", t_stat)
print("p-value :", p_value)


# ==================================================
# 6. 결과 판단
#
# ==================================================

if p_value < 0.05:  # p 벨류가 유의수준을 넘지 못해 귀무가설이 기각되었다.
    print("귀무가설 기각")
    print("→ 정상군과 고장군의 Torque 평균에 통계적으로 유의한 차이가 있다.")
else:
    print("귀무가설 기각 실패")
    print(
        "→ 정상군과 고장군의 Torque 평균에 통계적으로 유의한 차이가 있다고 보기 어렵다."
    )
