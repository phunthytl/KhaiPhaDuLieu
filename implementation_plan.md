# Kế hoạch nâng cấp Hệ thống Data Mining

Kế hoạch này dựa trên các định hướng ở **Phần 6** trong tài liệu báo cáo của bạn, nhằm chuyển đổi hệ thống từ các thuật toán heuristic (luật nhân tạo) sang các thuật toán Học máy (Machine Learning) bài bản, giúp nâng cao điểm số và tính khoa học của bài toán.

## User Review Required

> [!IMPORTANT]
> Toàn bộ các thay đổi dưới đây sẽ làm thay đổi bản chất của các mô hình đã được lưu (như `cf_model.pkl` và `hybrid_weights.json`).
> Bạn cần kiểm tra xem có cần cài đặt thêm thư viện nào ngoài `scikit-learn` không (hệ thống sẽ mặc định dùng `scikit-learn` và `scipy`).

## Open Questions

> [!WARNING]
> Đối với **CF bằng SVD**, hệ thống sẽ dùng `TruncatedSVD` của `scikit-learn` hoặc `svds` của `scipy`. Bạn muốn ma trận trả về được nén với bao nhiêu chiều (components)? Mặc định tôi sẽ thiết lập `k=50`.
>
> Đối với việc **Học trọng số Hybrid**, để đảm bảo trọng số mang ý nghĩa thực tế (không bị âm), tôi sẽ sử dụng mô hình Hồi quy tuyến tính có ràng buộc không âm (Non-Negative Least Squares hoặc `LinearRegression(positive=True)`), sau đó chuẩn hóa để tổng bằng 1. Cách này có phù hợp với mong muốn của bạn không?

## Proposed Changes

---

### Tiền xử lý & Phân khúc (Preprocessing)

Sẽ cập nhật cách tính toán để tự động hóa việc phân chia nhóm người dùng.

#### [MODIFY] [preprocess.py](file:///d:/Web/KhaiPhaDuLieu/preprocess.py)
- **Xóa**: Logic chia quantile (33%, 67%) rule-based trong `create_user_segments()`.
- **Thêm**: 
  - Trích xuất 2 đặc trưng cho mỗi user: `num_interactions` (số lần đánh giá) và `avg_rating` (điểm đánh giá trung bình).
  - Chuẩn hóa dữ liệu bằng `StandardScaler`.
  - Khởi tạo thuật toán `KMeans(n_clusters=3)` từ thư viện `sklearn.cluster` để phân nhóm.
  - Sau khi train xong, sắp xếp 3 cụm (clusters) theo `num_interactions` trung bình để gán lại các nhãn `Tier1_New`, `Tier2_Medium`, `Tier3_Old` một cách tự động và logic.

---

### Nâng cấp Mô hình (Model Training)

Đổi cốt lõi của thuật toán gợi ý cơ bản sang Matrix Factorization.

#### [MODIFY] [train_models.py](file:///d:/Web/KhaiPhaDuLieu/model/train_models.py)
- **Xóa**: Việc tính ma trận `cosine_similarity` giữa các user.
- **Thêm**: 
  - Áp dụng Matrix Factorization (SVD) trên ma trận `train_pivot` để khử nhiễu và lấp đầy các khoảng trống dữ liệu. 
  - Tái tạo lại ma trận dự đoán: $R_{predicted} = U \times \Sigma \times V^T$.
  - Lưu ma trận dự đoán hoàn chỉnh này vào `cf_model.pkl` thay vì lưu `user_similarity`. Điều này giúp quá trình dự đoán (inference) cực kỳ nhanh (O(1)) vì chỉ cần tra cứu bảng.

---

### Huấn luyện Trọng số Hybrid (Hybrid Training)

Tự động tìm kiếm hệ số kết hợp tối ưu thay vì đoán tay.

#### [MODIFY] [train_hybrid.py](file:///d:/Web/KhaiPhaDuLieu/model/train_hybrid.py)
- **Xóa**: Dictionary `hybrid_weights` đang bị "hardcode" (gán tĩnh).
- **Thêm**:
  - Dùng chính tập `train_set` (hoặc tách thêm 1 phần validation) để lấy nhãn $Y$ (`rating` thực tế).
  - Tính điểm từ 3 mô hình độc lập (CF, Content, Demographic) để làm 3 đặc trưng (Features $X$) cho tập dữ liệu học.
  - Với mỗi phân khúc (Tier1, Tier2, Tier3), khởi tạo một mô hình `LinearRegression(positive=True)` để học quan hệ giữa $X$ và $Y$.
  - Trích xuất các hệ số (`coef_`) của mô hình học được. Chuẩn hóa lại hệ số sao cho tổng w1 + w2 + w3 = 1.0.
  - Lưu trọng số học được ra `hybrid_weights.json`.
  - In ra trọng số học được cho mỗi Tier để dễ dàng báo cáo phân tích.

---

## Verification Plan

### Automated Tests
- Chạy toàn bộ pipeline theo thứ tự:
  1. `python preprocess.py` -> Kiểm tra file `user_segments.csv` xem nhãn KMeans tạo ra có hợp lý không.
  2. `python model/train_models.py` -> Theo dõi RMSE/MAE của mô hình CF mới (kỳ vọng tốt hơn Cosine Similarity truyền thống).
  3. `python model/train_hybrid.py` -> Xem trọng số thuật toán tự học ra là bao nhiêu, và so sánh các chỉ số (RMSE, MAE, NDCG) xem có tối ưu hơn trọng số gán bằng tay không.
- Chạy `python test_app.py` và `python test_hybrid.py` để đảm bảo API và thuật toán Inference (Dự đoán trực tiếp) vẫn hoạt động chính xác với mô hình mới.

### Manual Verification
- Bạn cần rà soát lại file báo cáo để sửa đổi lý thuyết tương ứng với Code (Thêm các phần nói về SVD, KMeans, LinearRegression).
