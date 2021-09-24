from torch import nn
import torch
import numpy as np
from SelfLoss import MarginLoss
import torch.nn.functional as F


class Similar(nn.Module):
    def __init__(self):
        super(Similar, self).__init__()

    # 计算要替换的item和可替换item之间的相似度和loss
    def calculate_similar_loss(self, replaceable_feature, original_feature):
        raise NotImplementedError

    # 根据用户和item对形成新的特征X 并使用这个特征 给每个要替换的item 选择可以替换的item
    def choose_replaceable_item(self, user_item_id, item_feature, all_items):
        raise NotImplementedError


# 针对计算结果进行正规划
class RegularSimilar(Similar):

    def __init__(self, similarity_ratio, latent_dim, sample_items):
        super(RegularSimilar, self).__init__()
        self.similarity_ratio = similarity_ratio
        self.latent_dim = latent_dim
        # 随机采样item
        self.sample_items = sample_items
        # 计算相相似度
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # 设置损失计算
        self.similarity_loss = MarginLoss()
        # torch.nn.L1Loss()
        # 设置线性变换
        self.user_item_feature = nn.Linear(2 * self.latent_dim, self.latent_dim)

    def calculate_similar_loss(self, replaceable_feature, original_feature):
        labels = torch.empty((replaceable_feature.shape[0])).cuda()
        labels[:] = self.similarity_ratio
        # 获取原始item和要替换的item的特征
        # 计算原始向量 和  可替换向量之间的 相似度
        similarity = self.cos(original_feature, replaceable_feature)
        # 归一化相似度
        similarity = self.regularize_similarity(similarity)
        # 计算每个向量的相似度和相似度阈值的loss
        # todo:是不是均方差更有效果
        similarity_loss = self.similarity_loss(similarity, labels)

        return similarity_loss, similarity.mean()

    def choose_replaceable_item(self, need_replace, union_feature, all_items):
        user_number = need_replace.shape[0]
        user_ids = need_replace[:, 0]
        item_ids = need_replace[:, 1]
        # 原始的item特征
        items_emb = all_items[item_ids].view(-1, 1, self.latent_dim)
        # 根据用户id获取每个用户采样的item列表
        sample_items = self.sample_items[user_ids]
        # 获取采样的item特征
        sample_item_feature = all_items[sample_items]
        # 计算原始item和采样item的原始的得分
        item_rank_score = torch.mul(items_emb, sample_item_feature)
        item_rank_score = item_rank_score.sum(dim=-1)
        # 针对得分结果进行排序
        item_ranking = torch.sort(item_rank_score, dim=1, descending=False)[1]
        # 获得一个逆排序
        item_reverse_ranking = torch.sort(item_ranking, dim=1, descending=False)[1]

        # 设置一个排序之后的item列表
        sample_item_sort_list = []
        for iter_id, rank_element in enumerate(item_ranking):
            sample_item_sort_list.append(sample_items[iter_id][rank_element])
        sample_item_sort_list = torch.stack(sample_item_sort_list)

        # 基于用户和item的联合特征 生成一个新的特征Z
        user_item_feature = self.user_item_feature(union_feature)
        user_item_feature = user_item_feature.view(-1, 1, self.latent_dim)

        # 计算新特征和所有采样item的得分
        replace_score = torch.mul(user_item_feature, sample_item_feature)
        replace_score = replace_score.sum(dim=-1)
        # 采用得分最高的那个元素用于替换
        replace_probability = F.gumbel_softmax(replace_score, tau=1e-4, hard=True)
        # 获取每个最高分的数据在排序之后位置信息
        # 下标是0开始的
        position_index = (item_reverse_ranking * replace_probability).sum(dim=-1) + 1
        # 基于位置信息计算一个相似度
        similarity = position_index / self.sample_items.shape[1]
        # print(position_index, similarity.mean())
        # 设置位置信息的阈值
        labels = torch.empty(user_number).cuda()
        labels[:] = self.similarity_ratio
        # 计算相似度loss
        similarity_loss = self.similarity_loss(similarity, labels)
        # 获取每个item需要替换的item项
        replaceable_items = (sample_item_sort_list * replace_probability).sum(dim=-1).long()
        # 取选择的item特征值
        replace_probability = replace_probability.unsqueeze(2)
        replaceable_items_feature = (sample_item_feature * replace_probability).sum(dim=1)

        return replaceable_items, replaceable_items_feature, similarity_loss, similarity.mean()

    def regularize_similarity(self, replace_scores):
        return (replace_scores + 1) / 2
