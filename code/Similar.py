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

    def __init__(self, latent_dim):
        super(RegularSimilar, self).__init__()
        self.latent_dim = latent_dim
        # 计算相相似度
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
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

    # 用来生成原始item的遮挡mask
    def generate_original_item_mask(self, replace_scores, item_ids):
        mask = torch.ones(size=replace_scores.shape)
        item_ids = torch.tensor(item_ids).cuda()
        # 获取最长的一个proposal的长度
        item_matrix = torch.arange(0, replace_scores.shape[1]).long().cuda()
        mask_expand = item_matrix.unsqueeze(0).expand(replace_scores.shape[0], replace_scores.shape[1])
        item_expand = item_ids.unsqueeze(1).expand_as(mask_expand)
        mask = (item_expand != mask_expand).int()
        return mask

    def choose_replaceable_item(self, need_replace, union_feature, all_items, privacy_settings):
        item_ids = need_replace[:, 1]
        # 原始的item特征
        items_emb = all_items[item_ids]
        # 基于用户和item的联合特征 生成一个新的特征Z
        union_feature = torch.cat([union_feature, privacy_settings.view(-1, 1)], dim=-1)
        user_item_feature = self.user_item_feature(union_feature)
        # 计算新特征和所有采样item的得分
        replace_score = torch.mm(user_item_feature, all_items.T)
        mask = self.generate_original_item_mask(replace_score, item_ids)
        # 遮挡住原先的item得分
        replace_score = replace_score * mask
        # 采用得分最高的那个元素用于替换
        if world.is_train:
            replace_probability = F.gumbel_softmax(replace_score, tau=1e-4, hard=True)

            item_sequence = torch.arange(0, all_items.shape[0]).view(1, -1).cuda()
            replaceable_items = (replace_probability * item_sequence).sum(dim=-1).long()
            # 获得新的item的特征信息
            replaceable_items_feature = torch.mm(replace_probability, all_items)

            # 原始的item 和 选择出来的item 做相似度loss计算
            similarity_loss, similarity \
                = self.calculate_similar_loss(items_emb, replaceable_items_feature, privacy_settings)

        else:
            replaceable_items = torch.argmax(replace_score, dim=1)
            replaceable_items_feature = all_items[replaceable_items]
            similarity_loss = 0.
            similarity = 0.

        return replaceable_items, replaceable_items_feature, similarity_loss, similarity

    def regularize_similarity(self, replace_scores):
        return (replace_scores + 1) / 2
