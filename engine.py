import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
import horovod.torch as hvd


class Metric:
    """Horovod allreduce를 이용해서 분산 환경에서 평균을 계산하는 metric"""

    def __init__(self, name: str):
        self.name = name
        self.sum = torch.tensor(0.0)
        self.n = torch.tensor(0.0)

    def update(self, val: torch.Tensor):
        # 각 worker의 값을 allreduce로 모아서 합을 만들고 스텝 수로 나눠서 계산함
        reduced = hvd.allreduce(val.detach().cpu(), name=self.name)
        self.sum += reduced
        self.n += 1.0

    @property
    def avg(self) -> torch.Tensor:
        return self.sum / self.n


def accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    batch_idx: int,
    args,
    train_loader_len: int,
):
    """
    Horovod 논문에서 사용하는 warmup + step decay 스케줄을 구현.
    - warmup 동안: base_lr -> base_lr * hvd.size()로 선형 증가
    - 후반부: 30, 60, 80 epoch에서 10배씩 감소
    """
    if epoch < args.warmup_epochs:
        epoch = epoch + float(batch_idx + 1) / float(train_loader_len)
        lr_adj = (
            1.0
            / hvd.size()
            * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1.0)
        )
    elif epoch < 30:
        lr_adj = 1.0
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group["lr"] = (
            args.base_lr
            * hvd.size()
            * args.batches_per_allreduce
            * lr_adj
        )


def train_one_epoch(
    model,
    optimizer,
    train_loader,
    train_sampler,
    epoch: int,
    args,
    log_writer,
    verbose: int = 1,
):
    """한 epoch 동안 학습 수행."""
    model.train()
    train_sampler.set_epoch(epoch)

    train_loss = Metric("train_loss")
    train_acc = Metric("train_accuracy")

    with tqdm(
        total=len(train_loader),
        desc=f"Train Epoch     #{epoch + 1}",
        disable=not bool(verbose),
    ) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(
                optimizer=optimizer,
                epoch=epoch,
                batch_idx=batch_idx,
                args=args,
                train_loader_len=len(train_loader),
            )

            if args.cuda:
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            for i in range(0, len(data), args.batch_size):
                data_batch = data[i : i + args.batch_size]
                target_batch = target[i : i + args.batch_size]

                output = model(data_batch)
                batch_acc = accuracy(output, target_batch)
                loss = F.cross_entropy(output, target_batch)

                train_acc.update(batch_acc)
                train_loss.update(loss)

                # 여러 sub-batch에 대해 나눠서 backward를 호출하므로,
                # 전체 gradient가 동일하도록 loss를 sub-batch 개수로 나눔
                loss = loss.div(
                    math.ceil(float(len(data)) / float(args.batch_size))
                )
                loss.backward()

            # Horovod DistributedOptimizer가 각 worker의 gradient를 평균내고 step 수행
            optimizer.step()

            t.set_postfix(
                {
                    "loss": train_loss.avg.item(),
                    "accuracy": 100.0 * train_acc.avg.item(),
                }
            )
            t.update(1)

    if log_writer is not None:
        log_writer.add_scalar("train/loss", train_loss.avg, epoch)
        log_writer.add_scalar("train/accuracy", train_acc.avg, epoch)


def validate_one_epoch(
    model,
    val_loader,
    epoch: int,
    args,
    log_writer,
    verbose: int = 1,
):
    model.eval()

    val_loss = Metric("val_loss")
    val_acc = Metric("val_accuracy")

    with tqdm(
        total=len(val_loader),
        desc=f"Validate Epoch  #{epoch + 1}",
        disable=not bool(verbose),
    ) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data = data.cuda()
                    target = target.cuda()

                output = model(data)

                loss = F.cross_entropy(output, target)
                acc = accuracy(output, target)

                val_loss.update(loss)
                val_acc.update(acc)

                t.set_postfix(
                    {
                        "loss": val_loss.avg.item(),
                        "accuracy": 100.0 * val_acc.avg.item(),
                    }
                )
                t.update(1)

    if log_writer is not None:
        log_writer.add_scalar("val/loss", val_loss.avg, epoch)
        log_writer.add_scalar("val/accuracy", val_acc.avg, epoch)


def save_checkpoint(epoch: int, model, optimizer, args):
    """rank 0에서만 체크포인트 저장"""
    if hvd.rank() != 0:
        return

    filepath = args.checkpoint_format.format(epoch=epoch + 1)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, filepath)
