# CIFAR-100 với Wide ResNet-16-4

Project chính là bài toán phân loại ảnh CIFAR-100 bằng Wide ResNet-16-4. Mã train và đánh giá nằm trong package Python `cifar_classifier`, được chuyển từ `build_wrn_cifar_100.ipynb`. Các notebook và thí nghiệm cũ vẫn ở vị trí hiện tại.

Sơ đồ mô hình: [Wide ResNet architecture](wide_resnet_architecture.pdf).

## Cài đặt

Dùng Python 3.10 trở lên. Từ thư mục chứa project, chạy:

```bash
cd cifar-100-deep-learning
python -m pip install -e .
```

Dữ liệu CIFAR-100 dạng Python pickle cần nằm ở `cifar-100/`, gồm ba file `train`, `test`, `meta`. Có thể chỉ định vị trí khác bằng `--data-dir`.

## Thứ tự chạy pipeline

**Dùng checkpoint đã có** — không cần train lại:

```bash
python -m cifar_classifier plot-history
python -m cifar_classifier evaluate
```

**Train lại từ đầu** — chạy train, vẽ history, rồi đánh giá checkpoint vừa tạo:

```bash
python -m cifar_classifier train
python -m cifar_classifier plot-history --history artifacts/runs/TEN_RUN/history.csv
python -m cifar_classifier evaluate \
  --checkpoint artifacts/runs/TEN_RUN/best.keras \
  --output-dir artifacts/runs/TEN_RUN/test-evaluation
```

Sau bước train, thay `TEN_RUN` bằng tên thư mục mới được in ra và tạo trong `artifacts/runs/`. Lệnh `evaluate` không có `--checkpoint` luôn dùng checkpoint cũ trong `checkpoints/`.

## Đánh giá checkpoint hiện có

```bash
python -m cifar_classifier evaluate
```

Lệnh này dùng `checkpoints/cifar100_wrn16_4_aug200_best.keras`, tái tạo phép chia train/validation với seed 42, rồi chuẩn hóa test bằng mean/std của 45.000 ảnh train. Kết quả được lưu vào `artifacts/evaluation/legacy-wrn200/`: `test_metrics.json`, báo cáo phân loại CSV và ma trận nhầm lẫn CSV/PNG. Bộ nạp checkpoint xử lý định dạng Keras 2.15/Windows của file cũ khi chạy trên Keras 3.

Mốc đối chiếu trên 10.000 ảnh test: top-1 accuracy **74,05%**, top-5 accuracy **93,13%**, loss khoảng **0,9530**.

## Vẽ training history

```bash
python -m cifar_classifier plot-history
```

Lệnh đọc hai CSV WRN trong `checkpoints/` và tạo một hình accuracy/loss riêng cho mỗi lượt train tại `artifacts/plots/`. Với một run mới:

```bash
python -m cifar_classifier plot-history --history artifacts/runs/TEN_RUN/history.csv
```

Để xem trực quan các phép augment, mở `build_wrn_cifar_100.ipynb` và chạy từ đầu đến hai cell **Preview** trước phần xây dựng model; không cần chạy các cell train. Notebook hiển thị cùng ảnh trước/sau augmentation và so sánh MixUp với CutMix. Hình được lưu tại `artifacts/plots/wrn_before_after_augmentation.png` và `artifacts/plots/wrn_mixup_cutmix_preview.png`.

## Train một model mới

```bash
python -m cifar_classifier train
```

Mặc định tương ứng cấu hình 200 epoch trong notebook: batch size 64, seed 42, 5.000 ảnh validation, AdamW, giảm learning rate khi `val_loss` chững lại, early stopping, và lưu checkpoint có `val_loss` thấp nhất. Quá trình có thể dừng trước epoch 200.

Mỗi lượt train ghi vào một thư mục mới theo thời gian trong `artifacts/runs/`, gồm `best.keras`, `history.csv` và `run.json`. `run.json` lưu cấu hình, mean/std, phiên bản TensorFlow và chỉ số validation. Checkpoint cũ trong `checkpoints/` không bị ghi đè.

Để dùng cấu hình thử 20 epoch của notebook:

```bash
python -m cifar_classifier train --epochs 20 --schedule cosine
```

Đánh giá checkpoint của một run mới:

```bash
python -m cifar_classifier evaluate \
  --checkpoint artifacts/runs/TEN_RUN/best.keras \
  --output-dir artifacts/runs/TEN_RUN/test-evaluation
```

Khi có `run.json` cạnh checkpoint, lệnh đánh giá dùng mean/std đã lưu trong run đó. Tập test chỉ dùng ở bước đánh giá.

## Cấu trúc mã nguồn

- `src/cifar_classifier/data.py`: đọc CIFAR-100, chia train/validation, chuẩn hóa và augmentation.
- `src/cifar_classifier/model.py`: kiến trúc WRN-16-4.
- `src/cifar_classifier/train.py`: train và callback checkpoint.
- `src/cifar_classifier/checkpoints.py`: nạp checkpoint mới và checkpoint cũ.
- `src/cifar_classifier/evaluate.py`: chỉ số test và báo cáo phân loại.
- `src/cifar_classifier/plots.py`: biểu đồ training history.
- `src/cifar_classifier/__main__.py`: ba lệnh CLI.

Chạy kiểm tra bằng `python -m unittest discover -s tests` sau khi cài package. Các output mới trong `artifacts/` được Git bỏ qua.

## EfficientNetV2-B0 pretrained (PyTorch)

Package `.py` tại `src/cifar_classifier/pretrained/` là nơi thực hiện train và
đánh giá. `build_efficientnet.ipynb` gọi lại chính các hàm này để xem từng bước
và biểu đồ trong Jupyter. Chạy lệnh hoặc mở notebook từ thư mục gốc project.

```bash
python -m pip install -e '.[pretrained]'
python -m cifar_classifier.pretrained train
```

### Chạy bằng notebook

Sau khi cài `.[pretrained]`, mở `build_efficientnet.ipynb` trong Jupyter/VS Code
và chọn **Run All**. Các cell lần lượt kiểm tra dữ liệu local, train, vẽ history,
đánh giá checkpoint tốt nhất và hiển thị ảnh dự đoán cùng confusion matrix.
Notebook mới không mang theo output 84,95% của lượt chạy cũ vì đó là kết quả
của cấu hình trước khi đồng bộ augmentation.

Để kiểm tra nhanh mà không train toàn bộ dữ liệu, đặt biến môi trường
`CIFAR_PRETRAINED_SMOKE=1` trước khi khởi động Jupyter rồi chạy notebook.
Chế độ này dùng 4 ảnh train, 2 ảnh validation và 4 ảnh test ở 64×64;
checkpoint và hình chỉ nhằm xác nhận pipeline chạy, không phải kết quả mô hình.

Lệnh train dùng `cifar-100/train`, chia 45.000 ảnh train và 5.000 ảnh validation
bằng `default_rng(42)`, khởi tạo EfficientNetV2-B0 bằng trọng số ImageNet từ
`timm`, và train tối đa 30 epoch. Ảnh train được flip/crop/jitter/Cutout hoặc
Random Erasing ở 32×32, sau đó resize 224×224, chuẩn hóa ImageNet, và chọn
MixUp/CutMix theo batch. Checkpoint tốt nhất được chọn theo `val_loss`.

Trọng số ImageNet đã được tải vào `artifacts/pretrained/cache/` trên máy này.
Máy khác cần truy cập nguồn pretrained ở lần chạy đầu. Nếu đã có file trọng số ImageNet tương ứng,
dùng `--pretrained-weights /duong/dan/model.safetensors` để chạy offline.
`--no-pretrained` chỉ dành cho kiểm tra pipeline hoặc thí nghiệm khởi tạo ngẫu nhiên.

Mỗi lần train tạo một thư mục mới `artifacts/pretrained/runs/efficientnetv2-b0-...`
chứa `best.pth`, `history.csv`, và `run.json`. Lệnh train in đường dẫn thư mục
run khi bắt đầu và kết thúc. Thay `TEN_RUN` trong các lệnh sau bằng tên đó:

```bash
python -m cifar_classifier.pretrained plot-history \
  --history artifacts/pretrained/runs/TEN_RUN/history.csv
python -m cifar_classifier.pretrained evaluate \
  --checkpoint artifacts/pretrained/runs/TEN_RUN/best.pth
```

Lệnh evaluate dùng checkpoint đã lưu để đánh giá 10.000 ảnh test, không tải lại
trọng số ImageNet. Kết quả gồm `test_metrics.json`, báo cáo precision/recall/F1,
ma trận nhầm lẫn CSV/PNG và 9 ảnh dự đoán mẫu trong
`artifacts/pretrained/runs/TEN_RUN/test-evaluation/`.

Checkpoint `.pth` cũ do notebook tạo chỉ chứa `state_dict` cũng có thể đánh giá
bằng `--checkpoint`; nếu nó dùng kích thước đầu vào khác 224, truyền thêm
`--image-size`. Hiện project không có checkpoint EfficientNet đã train, nên
phải train hoặc cung cấp checkpoint trước khi đánh giá.

Để kiểm tra nhanh trên CPU mà không tải trọng số và không train toàn bộ dữ liệu:

```bash
python -m cifar_classifier.pretrained train \
  --no-pretrained --epochs 1 --image-size 64 --batch-size 2 \
  --limit-train 4 --limit-val 2 --device cpu
```

Lượt kiểm tra nhanh này chỉ chứng minh pipeline chạy và checkpoint nạp được;
không đo chất lượng mô hình. Muốn train thực tế, bỏ các cờ kiểm tra nhanh.
