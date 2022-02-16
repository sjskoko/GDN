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



###########
###########
import torch
from torch.nn import Sequential as Seq, Linear, ReLU, Parameter
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv as gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops

import torch.nn.functional as F
logits = .8 * torch.rand(10 ** 2, 2)
logits[:, 1] = 0
logits

z = torch.nn.functional.gumbel_softmax(logits, hard=True)


result = z.sum(0)
result
z
result[0]/(result[0] + result[1])

weight = Parameter(torch.Tensor(60, 60))
x = Parameter(torch.Tensor(5, 60))
edge_index = Parameter(torch.Tensor(2, 100))
x_j = edge_index
x = torch.matmul(x, weight)

edge_index, edge_weight = gcn_norm(  # yapf: disable
                            edge_index, None, x.size(0))

x_j * z[:, 0].contiguous().view([-1] + [1] * (x_j.dim() - 1))

z.shape
z[:, 0].shape
z[:, 0].contiguous().view([-1] + [1]*(x_j.dim() - 1)).shape

tt = torch.tensor([[[1,2,3], [4,5,6]],[[7,8,9],[10, 11, 12]],[[13, 14, 15], [16, 17, 18]]])
tt.view(-1, 3, 2)

(x*x).shape