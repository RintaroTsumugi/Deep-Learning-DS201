# LAB2 – Convolutional Neural Networks (CNN)

## 1. Giới thiệu

Repository này chứa các bài tập thực hành về **mạng nơ-ron tích chập (Convolutional Neural Networks – CNN)**, bao gồm các mô hình học sâu được triển khai bằng Python và PyTorch.

Nội dung được tổ chức thành các file yêu cầu bài tập, chương trình thực thi từng bài, các mô hình mạng nơ-ron và tiện ích xử lý dữ liệu.

## 2. Mục tiêu

- Tìm hiểu và triển khai các kiến trúc CNN phổ biến.
- Thực hành xây dựng và sử dụng các mô hình phân loại ảnh.
- Làm việc với bộ dữ liệu MNIST và VinaFood21 theo yêu cầu của từng bài tập.
- Tổ chức mã nguồn thành các module riêng biệt để thuận tiện cho việc quản lý và tái sử dụng.

## 3. Cấu trúc thư mục

```text
LAB2/
├── data_utils/
│   ├── __init__.py
│   └── mnist_dataset.py
│
├── models/
│   ├── ResnetBlock.py
│   ├── __init__.py
│   ├── googlenet.py
│   ├── inception.py
│   ├── lenet.py
│   ├── resnet18.py
│   └── resnet50.py
│
├── .gitignore
├── CNN_Requirements.ipynb
├── assignment-1.py
├── assignment-2.py
├── assignment-3.py
└── assignment-4.py
```

### Mô tả các thành phần

| Thành phần | Mô tả |
|---|---|
| `data_utils/` | Chứa các tiện ích xử lý và chuẩn bị dữ liệu. |
| `mnist_dataset.py` | Module liên quan đến bộ dữ liệu MNIST. |
| `models/` | Chứa các module triển khai kiến trúc mạng nơ-ron. |
| `ResnetBlock.py` | Module định nghĩa khối ResNet. |
| `googlenet.py` | Module triển khai kiến trúc GoogLeNet. |
| `inception.py` | Module định nghĩa các thành phần Inception. |
| `lenet.py` | Module triển khai kiến trúc LeNet. |
| `resnet18.py` | Module triển khai kiến trúc ResNet-18. |
| `resnet50.py` | Module triển khai kiến trúc ResNet-50. |
| `CNN_Requirements.ipynb` | Notebook chứa đề bài và yêu cầu thực hành. |
| `assignment-1.py` – `assignment-4.py` | Các file mã nguồn thực hiện bốn bài tập tương ứng. |
| `.gitignore` | Quy định các file và thư mục Git bỏ qua khi theo dõi mã nguồn. |

## 4. Công nghệ sử dụng

- **Python** – Ngôn ngữ lập trình.
- **PyTorch** – Xây dựng và huấn luyện mô hình học sâu.
- **Torchvision** – Hỗ trợ xử lý dữ liệu ảnh và các thành phần phục vụ thị giác máy tính.

## 5. Bộ dữ liệu

- **MNIST:** Bộ dữ liệu ảnh chữ số viết tay, thường được sử dụng trong bài toán phân loại ảnh.
- **VinaFood21:** Bộ dữ liệu ảnh thực phẩm gồm 21 lớp, được sử dụng trong các bài tập phân loại ảnh tương ứng.

## 6. Thực hiện bài tập

1. Mở `CNN_Requirements.ipynb` để xem yêu cầu của các bài tập.
2. Kiểm tra file `assignment-1.py` đến `assignment-4.py` để xác định nội dung triển khai tương ứng.
3. Đảm bảo các bộ dữ liệu và thư viện cần thiết đã được chuẩn bị.
4. Chạy file Python tương ứng với bài tập cần thực hiện.




