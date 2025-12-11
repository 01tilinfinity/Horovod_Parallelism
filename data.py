import torch
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torchvision import datasets, transforms
import horovod.torch as hvd


def build_dataloaders(args, use_cuda: bool):
    """CIFAR-10 학습/검증용 DataLoader 생성 함수."""

    kwargs = {"num_workers": 4, "pin_memory": True} if use_cuda else {}

    if (
        kwargs.get("num_workers", 0) > 0
        and hasattr(mp, "_supports_context")
        and mp._supports_context
        and "forkserver" in mp.get_all_start_methods()
    ):
        kwargs["multiprocessing_context"] = "forkserver"

    train_transform = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    train_dataset = datasets.CIFAR10(
        args.train_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
    )
    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=allreduce_batch_size,
        sampler=train_sampler,
        **kwargs,
    )

    val_dataset = datasets.CIFAR10(
        args.val_dir,
        train=False,
        download=True,
        transform=val_transform,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=hvd.size(),
        rank=hvd.rank(),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        **kwargs,
    )

    return train_loader, val_loader, train_sampler, val_sampler
