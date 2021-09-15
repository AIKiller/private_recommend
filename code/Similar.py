from torch import nn
import torch
import numpy as np
from SelfLoss import SimilarityMarginLoss
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

    def __init__(self, similarity_ratio, latent_dim):
        super(RegularSimilar, self).__init__()
        self.similarity_ratio = similarity_ratio
        self.latent_dim = latent_dim
        # 计算相相似度
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # 设置损失计算
        self.similarity_loss = SimilarityMarginLoss()
        # torch.nn.L1Loss()
        # 设置线性变换
        self.user_item_feature = nn.Sequential(
            nn.Linear(2 * self.latent_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU()
        )

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

    def choose_replaceable_item(self, user_item_id, item_feature, all_items):
        user_item_id = np.array(user_item_id)
        # 用户的数量
        user_number = user_item_id.shape[0]
        user_item_index = np.array([i for i in range(all_items.shape[0])]).reshape(1, -1)
        user_item_index = torch.from_numpy(user_item_index).cuda()
        user_item_index = user_item_index.expand(user_number, -1)
        # 根据<user, item>特征对 生成新的特征Z  needReplaceNodes * latent_dim
        user_item_feature = self.user_item_feature(item_feature)
        # 计算每个需要替换的item 和 所有item的得分
        replace_score = torch.mm(user_item_feature, all_items.T)
        replace_probs = F.gumbel_softmax(replace_score, tau=1e-4, hard=True)
        # 获取每个节点应该替换的特征，用于后续的loss计算
        user_replaceable_feature = torch.mm(replace_probs, all_items)
        # 获取一个要替换的item的列表
        replaceable_items = user_item_index * replace_probs
        replaceable_items = replaceable_items.sum(dim=1).view(-1).long()
        # 获取原始要替换的item列表
        original_items = user_item_id[:, 1]
        original_user_feature = all_items[original_items]
        # 计算可替换item和原始item之间的关系
        similarity_loss, similarity = \
            self.calculate_similar_loss(user_replaceable_feature, original_user_feature)

        return replaceable_items, similarity_loss, similarity

    def regularize_similarity(self, replace_scores):
        return (replace_scores + 1) / 2
