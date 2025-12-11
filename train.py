import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import horovod.torch as hvd

from data import build_dataloaders
from engine import (
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
)


def parse_args():
    """커맨드라인 인자 정의."""
    parser = argparse.ArgumentParser(
        description="PyTorch CIFAR-10 Horovod 예제",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 데이터 경로
    parser.add_argument(
        "--train-dir",
        default="./data",
        help="CIFAR-10 학습 데이터를 저장할 경로 (자동 다운로드)",
    )
    parser.add_argument(
        "--val-dir",
        default="./data",
        help="CIFAR-10 검증 데이터를 저장할 경로 (자동 다운로드)",
    )

    # 로그/체크포인트 관련
    parser.add_argument(
        "--log-dir",
        default="./logs",
        help="TensorBoard 로그를 저장할 디렉토리",
    )
    parser.add_argument(
        "--checkpoint-format",
        default="./checkpoint-{epoch}.pth.tar",
        help="체크포인트 파일 이름 포맷 (예: ./checkpoint-{epoch}.pth.tar)",
    )

    # Horovod / 통신 관련
    parser.add_argument(
        "--fp16-allreduce",
        action="store_true",
        default=False,
        help="allreduce 시 fp16 압축 사용 여부",
    )
    parser.add_argument(
        "--batches-per-allreduce",
        type=int,
        default=1,
        help="allreduce 전에 로컬에서 처리할 배치 수 (배치 축적)",
    )
    parser.add_argument(
        "--use-adasum",
        action="store_true",
        default=False,
        help="Horovod Adasum 알고리즘 사용 여부",
    )
    parser.add_argument(
        "--gradient-predivide-factor",
        type=float,
        default=1.0,
        help="gradient predivide factor (일반적으로 1.0)",
    )

    # 학습 하이퍼파라미터
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="학습 배치 크기 (GPU 하나 기준)",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=256,
        help="검증 배치 크기",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="총 학습 epoch 수",
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=0.1,
        help="GPU 하나 기준 learning rate (Horovod에서 size에 맞게 스케일링)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=5,
        help="learning rate warmup에 사용할 epoch 수",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.00005,
        help="weight decay",
    )

    # 기타 설정
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="CUDA 사용 끄기",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="난수 시드",
    )

    return parser.parse_args()


def main(args):
    hvd.init()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # rank 0에서 찾은 resume epoch를 다른 worker들로 브로드캐스트
    resume_from_epoch = hvd.broadcast(
        torch.tensor(resume_from_epoch),
        root_rank=0,
        name="resume_from_epoch",
    ).item()

    verbose = 1 if hvd.rank() == 0 else 0

    # TensorBoard writer는 rank 0만 생성
    log_writer = SummaryWriter(args.log_dir) if hvd.rank() == 0 else None

    # worker별 GPU thread 수 
    torch.set_num_threads(4)

    train_loader, val_loader, train_sampler, val_sampler = build_dataloaders(
        args, use_cuda=args.cuda
    )
    model = models.resnet18()

    # Horovod 설정에 따른 learning rate 스케일링
    if args.use_adasum and hvd.nccl_built():
        # GPU Adasum 사용 시 local_size 기준 스케일
        lr_scaler = args.batches_per_allreduce * hvd.local_size()
    else:
        # 일반 allreduce일 때는 전체 GPU 수 기준 스케일
        lr_scaler = args.batches_per_allreduce * hvd.size()

    if args.cuda:
        model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr * lr_scaler,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    compression = (
        hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    )

    # Horovod DistributedOptimizer 래핑
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor,
    )

    # 체크포인트가 있으면 rank 0에서만 로드
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 파라미터와 옵티마이저 state를 전체 worker로 브로드캐스트
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # train
    for epoch in range(resume_from_epoch, args.epochs):
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            train_sampler=train_sampler,
            epoch=epoch,
            args=args,
            log_writer=log_writer,
            verbose=verbose,
        )

        validate_one_epoch(
            model=model,
            val_loader=val_loader,
            epoch=epoch,
            args=args,
            log_writer=log_writer,
            verbose=verbose,
        )

        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            args=args,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
