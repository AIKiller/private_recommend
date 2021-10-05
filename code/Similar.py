from torch import nn
import torch
import numpy as np
from SelfLoss import MarginLoss
import torch.nn.functional as F
import utils

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
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
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
        user_ids = need_replace[:, 0]
        item_ids = need_replace[:, 1]
        # 原始的item特征
        items_emb = all_items[item_ids]
        # 根据用户id获取每个用户采样的item列表
        sample_items = self.sample_items[user_ids]
        # 获取采样的item特征
        sample_item_feature = all_items[sample_items]

        # 基于用户和item的联合特征 生成一个新的特征Z
        user_item_feature = self.user_item_feature(union_feature)
        user_item_feature = user_item_feature.view(-1, 1, self.latent_dim)

        # 计算新特征和所有采样item的得分
        replace_score = torch.mul(user_item_feature, sample_item_feature)
        replace_score = replace_score.sum(dim=-1)
        # 统计一下采样数据里面 各个item相似度的分布
        similar_distribution = self.analyze_item_similar(items_emb, sample_item_feature)
        # 采用得分最高的那个元素用于替换
        replace_probability = F.gumbel_softmax(replace_score, tau=1e-4, hard=True)

        replaceable_items = (sample_items * replace_probability).sum(dim=-1).long()

        replace_probability = replace_probability.view(-1, sample_items.shape[1], 1)
        replaceable_items_feature = (sample_item_feature * replace_probability).sum(dim=1)
        # 原始的item 和 选择出来的item 做相似度loss计算
        similarity_loss, similarity = self.calculate_similar_loss(items_emb, replaceable_items_feature)

        return replaceable_items, replaceable_items_feature, similarity_loss, similarity, similar_distribution

    def regularize_similarity(self, replace_scores):
        return (replace_scores + 1) / 2

    def analyze_item_similar(self, original_feature, sample_item_feature):
        original_feature = original_feature.unsqueeze(1)
        similarity = self.cos(original_feature, sample_item_feature)
        # 归一化相似度
        similarity = self.regularize_similarity(similarity)
        similarity = similarity.cpu().numpy()
        similar_distribution = utils.similar_dis_statistic(similarity)
        similar_distribution = np.array(similar_distribution)
        similar_distribution = np.sum(similar_distribution, axis=0)
        similar_distribution = similar_distribution / sample_item_feature.shape[0] / sample_item_feature.shape[1]
        return similar_distribution
