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
        pretrain_file_path = './checkpoints/mf-gowalla-64.pth.tar'
        pretrain_dict = torch.load(pretrain_file_path, map_location=torch.device('cpu'))
        original_model_dict = Recmodel.state_dict()
        # 截取两个模型名称相同的层
        pretrained_dict = {k: v for k, v in pretrain_dict.items() if k in original_model_dict}
        original_model_dict.update(pretrained_dict)
        Recmodel.load_state_dict(original_model_dict)
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

best_loss = 999.
count = 1
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        # if epoch % 10 == 0:
        #     cprint("[TEST]")
            # result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            # if best_recall < result['recall'][0]:
            #     best_recall = result['recall'][0]
            #     best_precision = result['precision'][0]
            #     best_ndcg = result['ndcg'][0]
            #     count = 1
            #     torch.save(Recmodel.state_dict(), weight_file)
            # else:
            #     count += 1
            # if count > 10:
            #     # 训练数据的recall没有提升 出发early stop 策略
            #     print('best precision:{0:.4f}\t best recall:{1:.4f}\t best ndcg:{2:.4f}'.format(
            #         best_precision, best_recall, best_ndcg
            #     ))
            #     cprint("[Train END]")
            #     break
            # print(result)
        aver_loss, time_info, aver_similarity = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f"EPOCH[{epoch+1}/{world.TRAIN_epochs}] loss{aver_loss:.3f}-{time_info}-similarity{aver_similarity:.3f}")
        if epoch % 10 == 0 and epoch > 1:
            if best_loss > aver_loss:
                best_loss = aver_loss
                torch.save(Recmodel.state_dict(), weight_file)
                count = 1
            else:
                count += 1
            if count > 20:
                cprint("[Train END]")
                break

finally:
    if world.tensorboard:
        w.close()

Procedure.output_generative_data(dataset, Recmodel, weight_file)