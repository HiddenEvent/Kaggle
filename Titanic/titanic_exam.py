import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_df = pd.read_csv('./file/test.csv')
train_df = pd.read_csv('./file/train.csv')

# 1. 인덱스 지정
test_df.set_index('PassengerId', inplace=True)
train_df.set_index('PassengerId', inplace=True)

# print(train_df.head())
# print(test_df.head())

# 2. 인덱스 데이터만 따로 추출
test_index = test_df.index
train_index = train_df.index

# 3. 결과에 해당하는 Y 데이터 추출
y_train_df = train_df.pop("Survived")

# 4. 데이터 전처리 (소수 2째 자리까지만 표현하게)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 5. 결측치를 찾아내서 데이터 drop
test_df.isnull().sum() / len(test_df)  # Cabin      0.78

# Cabin의 nan값이 78%이기 떄문에 삭제 시켜준다
del test_df["Cabin"]
del train_df["Cabin"]

# 6. test데이터와 train데이터를 합쳐준다.
all_df = train_df.append(test_df)

# 7. all data에서 nan값 확인 (matplot 그래프를 사용해서 확인해보자!)
(all_df.isnull().sum() / len(all_df)).plot(kind='bar')
# plt.show() #그래프 그리기

# 8. 쓸모없는 Column 삭제
del all_df["Name"]
del all_df["Ticket"]

# 9. Sex(성별), Embarked(어느 항구에서 출발했는지) one hot encoding 으로 바꿔주는 작업
all_df["Sex"] = all_df["Sex"].replace({"male": 0, "female": 1})
# print(all_df.head())
# print(all_df["Embarked"].unique()) #카테고리 작업
all_df["Embarked"] = all_df["Embarked"].replace({"S": 0, "C": 1, "Q": 2 , np.nan: 99})


# 10. one hot encoding 용 dumy matrix 데이터 만들어 all_df에 merge시키기
matrix_df = pd.merge(
    all_df, pd.get_dummies(all_df["Embarked"], prefix="embarked"),
    left_index=True, right_index=True)
# print(matrix_df.head())

# 11. corr함수를 사용해서 상관관계가 큰 값은 삭제( 여기서는 안함..)
matrix_df.corr()

# 12. groupby 를 통해 nan값들을 평균값으로 치환
# print(all_df.groupby("Pclass")["Age"].mean()) # class별 age평균값

all_df.loc[
    (all_df["Pclass"]==1)  & (all_df["Age"].isnull()), "Age"] = 39.16
all_df.loc[
    (all_df["Pclass"]==2)  & (all_df["Age"].isnull()), "Age"] = 29.51
all_df.loc[
    (all_df["Pclass"]==3)  & (all_df["Age"].isnull()), "Age"] = 24.82
# > transformation으로 한번에 할 수도 있음.

# 13. 중간에 다시 nan값이 있나 확인
all_df.isnull().sum() # 확인해 보니 Fare 값이 또 있음..

# 14. Fare값 평균값으로 넣어 주기
# print(all_df.groupby("Pclass")["Fare"].mean()) # Fare에 평균값 찾기
# print(all_df[all_df["Fare"].isnull()])  # nan Fare Data를 갖는 Row 찾기
all_df.loc[all_df["Fare"].isnull(), "Fare"] = 13.30 # 값 넣기


# 14. one hot encoding 부분 원본 데이터 삭제
del all_df["Embarked"]

# 15. Pclass도 one Hot encoding으로 바꿔주자!
all_df["Pclass"] = all_df["Pclass"].replace({1: "A", 2: "B", 3: "C"})
all_df = pd.get_dummies(all_df)


# 16. matrix이랑 합치기
all_df = pd.merge(
    all_df, matrix_df[["embarked_0","embarked_1","embarked_2","embarked_99"]],
    left_index=True, right_index=True)
print(all_df)