"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import utils
from time import time
from Similar import RegularSimilar


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg, unique_user=None, pos_item_index=None, pos_item_mask=None):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
            unique_user: 本次训练中唯一的用户列表
            pos_item_index: positive items for corresponding users
            pos_item_mask: positive items mask for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg, unique_user=None, pos_item_index=None, pos_item_mask=None):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


class LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.replace_ratio = self.config['replace_ratio']
        self.similarity_ratio = self.config['similarity_ratio']

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.regularSimilar = RegularSimilar(self.similarity_ratio, self.latent_dim)


        if self.config['pretrain'] == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    # 计算每个正样本和用户的得分
    # 唯一用户标识
    # 每个用户的pos items
    # 每个用户pos items的mask
    # 当前批次要训练的item数据
    def computer_pos_score(self, users, pos_item_index, pos_item_mask, train_pos):
        all_users, all_items = self.computer()

        # 获取用户的特征信息
        users = torch.tensor(list(users)).long()
        users_emb = all_users[users]

        # 根据阈值获取每个用户需要替换的items
        user_list = users.detach().cpu().numpy()
        need_replace = []
        for index, user_id in enumerate(user_list):
            pos_items = pos_item_index[index]
            items_mask = (pos_item_mask[index] == 1)
            items = pos_items[items_mask].astype(np.int64)

            need_replace_item_start = len(items) - round(len(items) * self.replace_ratio)
            need_replace_items = items[need_replace_item_start:]
            # 循环 压入数据
            for item_id in need_replace_items:
                if item_id in train_pos:
                    need_replace.append([user_id, item_id])

        # 准备需要替换的信息
        need_replace = np.array(need_replace)
        # 获取所有的用户和item id的集合
        users_index = need_replace[:, 0]
        items_index = need_replace[:, 1]
        # 获取对应的特征
        users_emb = all_users[users_index]
        items_emb = all_items[items_index]
        need_replace_feature = torch.cat([users_emb, items_emb], dim=1)
        # 删除冗余数据
        del users_emb
        del items_emb
        del all_users

        # 获取每个需要替换的item 对应的相似item
        replaceable_items, similarity_loss, similarity = \
            self.regularSimilar.choose_replaceable_item(need_replace, need_replace_feature, all_items)

        return need_replace, replaceable_items, similarity_loss, similarity

    def replace_pos_items(self, users, pos_items, need_replace, replaceable_items):
        users = users.detach().cpu().numpy()
        pos_items = pos_items.detach().cpu().numpy()
        replaceable_items = replaceable_items.cpu().numpy()
        need_replace = np.array(need_replace)
        pos_items = utils.replace_original_to_replaceable(users, pos_items, need_replace, replaceable_items)
        return torch.from_numpy(pos_items).cuda()
    

    def bpr_loss(self, users, pos, neg, unique_user, pos_item_index=None, pos_item_mask=None):

        # 计算训练用户的正向样本的得分
        # 得分最低的样本需要被替换
        # 在所有的节点里面挑选相似度在阈值范围的节点
        # start_time = time()
        need_replace, replaceable_items, similarity_loss, similarity = \
            self.computer_pos_score(unique_user, pos_item_index, pos_item_mask, pos)
        # 替换所有要替换的节点
        pos = self.replace_pos_items(users, pos, need_replace, replaceable_items)
        # center_time = time()
        # print('计算替换节点的时间', center_time - start_time)

        # 进行loss 计算
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # self.replace_privacy_to_similar(pos_scores, neg_scores)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        # end_time = time()
        # print('loss 计算的时间间隔', end_time - center_time)

        return loss, reg_loss, similarity_loss, similarity

    def forward(self, users, items):
        # 这里代码没有运行
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
