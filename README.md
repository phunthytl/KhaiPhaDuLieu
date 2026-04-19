# Hệ thống gợi ý phim (Hybrid)


## 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## 2. Tiền xử lý dữ liệu
- Tạo các file _processed
```bash
python preprocess.py
```

Sinh các file: `movies_processed.csv`, `users_processed.csv`, `ratings_processed.csv`, `final_dataset.csv`, `user_segments.csv`.

## 3. Huấn luyện các mô hình thành phần (CF, Content, Demographic)
```bash
python model/train_models.py
```

Tạo (trong `model/`):

- `cf_model.pkl`
- `content_model.pkl`
- `demographic_model.pkl`

## 4. Trọng số hybrid và metrics đánh giá

```bash
python model/train_hybrid.py
```

Tạo / cập nhật:

- `model/hybrid_weights.json`
- `model/evaluation_metrics.json`

## 5. Chạy backend web (Flask)

Chạy python app.py để khởi động web