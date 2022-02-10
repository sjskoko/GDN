# 1(attack)으로 분류된 data 위치 산출


temp = main.test_result[2]
temp = [i for i, lab in enumerate(temp) if 1 in lab]
print(temp)

# 실제 attack의 lable이 1인 data 위치 산출
test_attack = pd.read_csv(f'data/HAI/test.csv')['attack'] # 216001
test_attack['attack']
(test_attack[1])
y_label = []
for i in range(len(test_attack)):
    if test_attack[i] == 1:
        y_label.append(i)
print(y_label)

# 예측 attack과 실제 attack 비교
len(test_attack)
len(y_label)
for i in range(len(test_attack)):
    if i

# 
np_test_result = np.array(main.test_result)

test_labels = np_test_result[2, :, 0].tolist() # label
for i, l in enumerate(test_labels):
    if l == 1:
        print(i)