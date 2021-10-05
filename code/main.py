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
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
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

best_recall = .0
best_precision = .0
best_ndcg = 0.
count = 1
best_similarity = 0.
total_similarity = 0.
total_similar_distribution = np.zeros(10, dtype=np.float32)
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            if best_recall < result['recall'][0]:
                best_recall = result['recall'][0]
                best_precision = result['precision'][0]
                best_ndcg = result['ndcg'][0]
                best_similarity = total_similarity / 10
                count = 1
            else:
                count += 1
            if count > 10:
                # 训练数据的recall没有提升 出发early stop 策略
                print('best precision:{0:.4f}\t best recall:{1:.4f}\t '
                      'best ndcg:{2:.4f}\t best similarity:{3:.4f}\t '.format(
                    best_precision, best_recall, best_ndcg, best_similarity
                ))
                cprint("[Train END]")
                break
            print(total_similar_distribution / 10)
            total_similar_distribution = np.zeros(10, dtype=np.float32)
            result['similarity'] = total_similarity/10
            print(result)
            total_similarity = 0.

        aver_loss, time_info, aver_similarity, batch_similar_distribution = \
            Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        total_similar_distribution += batch_similar_distribution
        total_similarity += aver_similarity.item()
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] '
              f'loss{aver_loss:.3f}-{time_info}-similarity{aver_similarity:.3f}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()