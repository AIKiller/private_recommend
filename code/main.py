import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
# 加载预训练模型
if world.LOAD:
    try:
        # pretrain_file_path = './checkpoints/mf-gowalla-64.pth.tar'
        pretrain_file_path = './checkpoints/mf-Office-64.pth.tar'
        # pretrain_file_path = './checkpoints/mf-Clothing-64.pth.tar'
        # pretrain_file_path = '/disk/lf/light-gcn/code/checkpoints/similarity0.99_Clothing_max_min-mf-Clothing-64-0.4.pth.tar'
        pretrain_dict = torch.load(pretrain_file_path, map_location=torch.device('cpu'))
        original_model_dict = Recmodel.state_dict()
        # 截取两个模型名称相同的层
        pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in original_model_dict}
        original_model_dict.update(pretrained_dict)
        Recmodel.load_state_dict(original_model_dict)
        world.cprint(f"loaded model weights from {pretrain_file_path}")
    except FileNotFoundError:
        print(f"pretrain_file_path not exists, please check it")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

best_recall = 0.
best_precision = 0.
best_ndcg = 0.
best_loss = 999.
count = 1
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        # if epoch % 10 == 0:
        #     cprint("[TEST]")
        #     result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        #     print(result)
        #     if best_recall < result['recall'][0]:
        #         best_recall = result['recall'][0]
        #         best_precision = result['precision'][0]
        #         best_ndcg = result['ndcg'][0]
        #         count = 1
        #         torch.save(Recmodel.state_dict(), weight_file)
        #     else:
        #         count += 1
        #     if count > 30:
        #         # 训练数据的recall没有提升 出发early stop 策略
        #         print('best precision:{0:.4f}\t best recall:{1:.4f}\t best ndcg:{2:.4f}'.format(
        #             best_precision, best_recall, best_ndcg
        #         ))
        #         cprint("[Train END]")
        #         break
        aver_loss, time_info, bpr_loss, similarity_loss, std_loss, aver_similarity = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] loss{aver_loss:.3f}-bpr{bpr_loss:.3f}"
              f"-similarity_loss:{similarity_loss:.3f}-std{std_loss:.3f}"
              f"-{time_info}-similarity{aver_similarity:.3f}")
        # userSimMax = torch.stack(dataset.userSimMax)
        # userSimMin = torch.stack(dataset.userSimMin)
        # print(torch.sum(userSimMin)/ userSimMin.shape[0], torch.sum(userSimMax) / userSimMax.shape[0])

        if epoch % 10 == 0 and epoch > 1:
            if best_loss > aver_loss:
                best_loss = aver_loss
                torch.save(Recmodel.state_dict(), weight_file)
                count = 1
            else:
                count += 1
            if count > 40:
                cprint("[Train END]")
                break

finally:
    if world.tensorboard:
        w.close()

Procedure.output_generative_data(dataset, Recmodel, weight_file)