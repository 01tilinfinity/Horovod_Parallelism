## CIFAR-10 ResNet-18 (Horovod + PyTorch)

2023 한국 정보처리학회 추계학술대회 클라우드컴퓨팅 트랙 논문 **<동기식 분산 딥러닝 환경에서 배치 사이즈 변화에 따른 모델 학습 성능 분석> (NIPA 원장상)** 의 코드입니다.
Horovod를 사용해 CIFAR-10 데이터셋에서 ResNet-18 모델을 학습할 수 있습니다.

## Settings

### Install packages

```bash
pip install -r requirements.txt
```

### Packages to be installed

-   torch
-   torchvision
-   horovod
-   tensorboard
-   tqdm (optional)

```bash
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITHOUT_MXNET=1 pip install --no-cache-dir horovod[tensorflow,keras,pytorch]
```

## Data

torchvision.datasets.CIFAR10을 사용하며, 처음 실행 시 --train-dir, --val-dir에 지정된 경로로 자동 다운로드됩니다.
기본값은 둘 다 ./data 입니다.

## How to Train

### Single GPU

```bash
python train.py \
  --train-dir ./data \
  --val-dir ./data \
  --epochs 50 \
  --batch-size 256
```

### Multi GPU

```bash
horovodrun -np 4 -H localhost:4 \
  python train.py \
    --train-dir ./data \
    --val-dir ./data \
    --epochs 50 \
    --batch-size 256
```

--batch-size는 GPU 1개 기준입니다.
전체 batch size = batch*size * hvd.size() \_ batches_per_allreduce

## Using Tensorboard

```bash
tensorboard --logdir ./logs
```

## Parameters

| parameter                  | description                                               |
| -------------------------- | --------------------------------------------------------- |
| `--train-dir`, `--val-dir` | CIFAR-10 데이터 저장 디렉토리                             |
| `--log-dir`                | TensorBoard 로그 경로 (기본: `./logs`)                    |
| `--checkpoint-format`      | 체크포인트 파일 포맷 (예: `./checkpoint-{epoch}.pth.tar`) |
| `--epochs`                 | 학습 epoch 수 (기본: 50)                                  |
| `--batch-size`             | GPU 하나 기준 학습 batch size (기본: 256)                 |
| `--val-batch-size`         | 검증 batch size (기본: 256)                               |
| `--base-lr`                | GPU 하나 기준 learning rate (기본: 0.1)                   |
| `--warmup-epochs`          | warmup 기간(epoch)                                        |
| `--batches-per-allreduce`  | allreduce 전에 로컬에서 누적할 batch 수                   |
| `--fp16-allreduce`         | fp16 gradient compression 사용 여부                       |
| `--use-adasum`             | Horovod Adasum 알고리즘 사용 여부                         |
