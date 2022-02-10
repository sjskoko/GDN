# 1(attack)으로 분류된 data 위치 산출
iter = 1
iter_list = []
for i in main.test_result[2]:
    if 1 in i:
        # print(iter)
        # print(i)
        iter_list.append(iter)
        iter += 1