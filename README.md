ĐỒ ÁN TỐT NGHIỆP: 
NHẬN DIỆN BẠO LỰC SỬ DỤNG PHƯƠNG PHÁP TRÍCH XUẤT ĐẶC TRƯNG THEO CHUỖI THỜI GIAN

## Trước khi chạy code
- Tải trọng số đã được đào tạo trước của mạng 3D resnet 18 tại (https://drive.google.com/file/d/12FxrQY2hX-bINbmSrN9q2Z5zJguJhy6C/view?usp=drive_link) và lưu vào folder pretrained_weight
- Clone mạng 3D resnet 18 tại (git clone https://github.com/kenshohara/3D-ResNets-PyTorch.git ResNets_3D_CNN)
## Hướng dẫn chạy code
- Chạy file make_dataset.py để tạo các video nhỏ 32 khung hình từ video UAV-human
- File detect_people.py để demo phát hiện người bằng YOLOv8
- Chạy file human_violence_detector.py để huấn luyện mô hình.