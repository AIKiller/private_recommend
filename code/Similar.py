from torch import nn
import torch
import numpy as np
from SelfLoss import SimilarityMarginLoss
import torch.nn.functional as F
import world

class Similar(nn.Module):
    def __init__(self):
        super(Similar, self).__init__()

    # 计算要替换的item和可替换item之间的相似度和loss
    def calculate_similar_loss(self, replaceable_feature, original_feature, privacy_settings):
        raise NotImplementedError

    # 根据用户和item对形成新的特征X 并使用这个特征 给每个要替换的item 选择可以替换的item
    def choose_replaceable_item(self, user_item_id, item_feature, all_items, privacy_settings):
        raise NotImplementedError


# 针对计算结果进行正规划
class RegularSimilar(Similar):

    def __init__(self, latent_dim, user_sample_items):
        super(RegularSimilar, self).__init__()
        self.latent_dim = latent_dim
        # 计算相相似度
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        # 用户的采样item列表
        self.user_sample_items = user_sample_items
        # 设置损失计算
        self.similarity_loss = SimilarityMarginLoss()
        # torch.nn.L1Loss()
        # 设置线性变换
        self.user_item_feature = nn.Linear((2 * self.latent_dim) + 1, self.latent_dim)

    def calculate_similar_loss(self, replaceable_feature, original_feature, privacy_settings):
        # 获取原始item和要替换的item的特征
        # 计算原始向量 和  可替换向量之间的 相似度
        similarity = self.cos(original_feature, replaceable_feature)
        # 归一化相似度
        similarity = self.regularize_similarity(similarity)
        # 计算每个向量的相似度和相似度阈值的loss
        # todo:是不是均方差更有效果
        similarity_loss = self.similarity_loss(similarity, privacy_settings)

        return similarity_loss, similarity.mean()

    def choose_replaceable_item(self, need_replace, union_feature, all_items, privacy_settings):
        user_ids = need_replace[:, 0]
        item_ids = need_replace[:, 1]
        # 原始item的特征
        items_emb = all_items[item_ids]
        # 采样的item列表
        sample_items = self.user_sample_items[user_ids]
        sample_items_emb = all_items[sample_items]
        # 基于用户和item的联合特征 生成一个新的特征Z
        union_feature = torch.cat([union_feature, privacy_settings.view(-1, 1)], dim=-1)
        user_item_feature = self.user_item_feature(union_feature)
        expand_items_emb = user_item_feature.view(-1, 1, self.latent_dim)
        # 计算新特征和所有采样item的得分
        replace_score = torch.mul(expand_items_emb, sample_items_emb)
        replace_score = replace_score.sum(-1)
        # 采用得分最高的那个元素用于替换
        if world.is_train:
            replace_probability = F.gumbel_softmax(replace_score, tau=1e-4, hard=True)

            replaceable_items = (replace_probability * sample_items).sum(dim=-1).long()

            # 获得新的item的特征信息
            replace_probability = replace_probability.view(-1, replace_probability.shape[1], 1)
            replaceable_items_feature = torch.mul(replace_probability, sample_items_emb)
            replaceable_items_feature = replaceable_items_feature.sum(dim=1)
            # 原始的item 和 选择出来的item 做相似度loss计算
            similarity_loss, similarity \
                = self.calculate_similar_loss(items_emb, replaceable_items_feature, privacy_settings)

        else:
            sample_item_index = F.softmax(replace_score, dim=1)
            replace_probability = sample_item_index.view(-1, sample_item_index.shape[1], 1)
            replaceable_items_feature = torch.mul(replace_probability, sample_items_emb)
            replaceable_items_feature = replaceable_items_feature.sum(dim=1)
            # 获取对应item的特征
            replaceable_items = (sample_item_index * sample_items).sum(dim=-1).long()
            similarity_loss = 0.
            similarity = 0.

        return replaceable_items, replaceable_items_feature, similarity_loss, similarity

    def regularize_similarity(self, replace_scores):
        return (replace_scores + 1) / 2
