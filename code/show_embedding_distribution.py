import math

import world
import utils
from world import cprint
import torch
import numpy as np
# from tensorboardX import SummaryWriter
# import time
# import Procedure
# from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
# import register
# from register import dataset
import numpy as np
#
# Recmodel = register.MODELS[world.model_name](world.config, dataset)
# Recmodel = Recmodel.to(world.device)
# bpr = utils.BPRLoss(Recmodel, world.config)
#
# weight_file = utils.getFileName()
# print(f"load and save to {weight_file}")
# # 加载预训练模型
# # pretrain_file_path = './checkpoints/mf-gowalla-64.pth.tar'
# pretrain_file_path = './checkpoints/mf-Office-64.pth.tar'
# # pretrain_file_path = './checkpoints/mf-Clothing-64.pth.tar'
# # pretrain_file_path = '/disk/lf/light-gcn/code/checkpoints/similarity0.99_Clothing_max_min-mf-Clothing-64-0.4.pth.tar'
# pretrain_dict = torch.load(pretrain_file_path, map_location=torch.device('cpu'))
# original_model_dict = Recmodel.state_dict()
# # 截取两个模型名称相同的层
# pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in original_model_dict}
# original_model_dict.update(pretrained_dict)
# Recmodel.load_state_dict(original_model_dict)
# world.cprint(f"loaded model weights from {pretrain_file_path}")
#
# user_emb = Recmodel.embedding_user.weight
# item_emb = Recmodel.embedding_item.weight
#
# user_number = user_emb.shape[0]
# user_index = [i for i in range(user_number)]
#
# item_number = item_emb.shape[0]
#
# sample_user_id = np.random.choice(user_index, 10, replace=False)
#
# user_pos_list = dataset.getUserPosItems(sample_user_id)
#
# need_replace_items = []
# # 给每个用户抽样一个item用于替换
# for posList in user_pos_list:
#     pos_item = np.random.choice(posList, 1, replace=False)[0]
#     need_replace_items.append(pos_item)
#
#
#
# need_replace_item_emb = item_emb[need_replace_items]
#
# item_relevance = torch.mm(need_replace_item_emb, item_emb.T)
# sorted_item_relevance = torch.sort(item_relevance, dim=-1, descending=False)
# sorted_item_index = sorted_item_relevance.indices
# #
# # print(item_emb[sorted_item_index].shape, item_emb[sorted_item_index].T.shape)
# # exit()
# sample_user_emb = user_emb[sample_user_id]
# sample_user_emb = sample_user_emb.view(-1, 1, 64)
# #
# # print(sample_user_emb.shape)
# # 计算user和每个item的评分结果
# rating = torch.mul(sample_user_emb, item_emb[sorted_item_index]).sum(-1).view(10, -1)
# import math
# similarities = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
# # 计算每10个档的相关性
# scores = torch.Tensor([]).cuda()
# for (i, step) in enumerate(range(len(similarities) -1 )):
#     start = math.floor(similarities[i] * item_number)
#     end = math.floor(similarities[i+1] * item_number)
#     user_score = rating[:, start:end]
#     scores = torch.cat([scores, user_score], dim=1)
#
# with open('../output/user_item_scores.npy', 'wb') as f:
#     np.save(f, scores.detach().cpu().numpy())
#
# exit()

import matplotlib.pyplot as plt


with open('../output/user_item_scores.npy', 'rb') as f:
    user_item_scores = np.load(f)


# 设置自变量的范围和个数
x = range(user_item_scores.shape[1])

for index in range(user_item_scores.shape[0]):
    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=300)
    plt.plot(x, user_item_scores[index, :])
    plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.05, bottom=0.05, right=0.95, top=0.95)
    plt.show()
    plt.clf()
