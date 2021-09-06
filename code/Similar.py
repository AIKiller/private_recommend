from torch import nn
import torch
import numpy as np
from SelfLoss import MarginLoss


class Similar(nn.Module):
    def __init__(self):
        super(Similar, self).__init__()

    # 计算要替换的item和可替换item之间的相似度和loss
    def calculate_similar_loss(self, all_items, original_items, sorted_items):
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
        self.similarity_loss = MarginLoss()
        # torch.nn.L1Loss()
        # 设置线性变换
        self.user_item_feature = nn.Sequential(
            nn.Linear(2 * self.latent_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU()
        )

    def calculate_similar_loss(self, all_items, original_items, sorted_items):
        labels = torch.empty((len(original_items))).cuda()
        labels[:] = self.similarity_ratio
        # 获取原始item和要替换的item的特征
        sorted_items = sorted_items.view(-1)
        original_item_features = all_items[original_items]
        sorted_item_feature = all_items[sorted_items]
        # 计算原始向量 和  可替换向量之间的 相似度
        similarity = self.cos(original_item_features, sorted_item_feature)
        # 归一化相似度
        similarity = self.regularize_similarity(similarity)
        # 计算每个向量的相似度和相似度阈值的loss
        # todo:是不是均方差更有效果
        similarity_loss = self.similarity_loss(similarity, labels)
        return similarity_loss, similarity.mean()

    def choose_replaceable_item(self, user_item_id, item_feature, all_items):
        user_item_id = np.array(user_item_id)
        # 根据<user, item>特征对 生成新的特征Z  needReplaceNodes * latent_dim
        user_item_feature = self.user_item_feature(item_feature)
        # 计算每个需要替换的item 和 所有item的得分
        replace_score = torch.mm(user_item_feature, all_items.T)
        # # 归一化点乘结果
        # replace_score = self.regularize_similarity(replace_score)
        # 取每个节点下面得分最高的节点
        top_score_items = torch.topk(replace_score, 1, dim=1)
        original_items = user_item_id[:, 1]
        # 计算可替换item和原始item之间的关系
        similarity_loss, similarity = self.calculate_similar_loss(all_items, original_items, top_score_items[1])
        replaceable_items = top_score_items[1].view(-1)

        return replaceable_items, similarity_loss, similarity

    def regularize_similarity(self, replace_scores):
        return (replace_scores + 1) / 2
