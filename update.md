# Hybrid Recommender System – Data Mining Summary

## 1. Tổng quan hệ thống

Hệ thống của bạn là một **Hybrid Recommender System**, kết hợp 3 phương pháp:

* Collaborative Filtering (CF)
* Content-Based Filtering
* Demographic Filtering

Mục tiêu: dự đoán rating và gợi ý phim phù hợp cho người dùng.

---

## 2. Các thành phần và thuật toán sử dụng

### 2.1 Collaborative Filtering (User-based CF)

* Dựa trên ma trận user–movie (rating)
* Tính độ tương đồng giữa user bằng cosine similarity
* Dự đoán rating bằng trung bình có trọng số từ các user tương tự

**Vai trò Data Mining:**

* Khai phá hành vi người dùng
* Tìm pattern tương đồng giữa user

---

### 2.2 Content-Based Filtering

* Sử dụng TF-IDF cho:

  * genres
  * overview
* Kết hợp thêm feature year
* Tạo vector đặc trưng cho từng phim
* Tạo user profile bằng trung bình các phim đã xem

**Vai trò Data Mining:**

* Text mining
* Khai phá nội dung

---

### 2.3 Demographic Filtering

* Dựa trên:

  * gender
  * age group
* Tính trung bình rating theo nhóm

**Vai trò:**

* Phân tích thống kê đơn giản
* Không phải thuật toán học mạnh

---

### 2.4 Hybrid Model

Kết hợp 3 mô hình:

```
pred = w1 * CF + w2 * Content + w3 * Demographic
```

* Trọng số thay đổi theo từng nhóm user

---

## 3. User Segmentation (user_segments)

### 3.1 Cách hiện tại

Phân nhóm user dựa trên số lượng rating:

* Tier1_New: ít rating (user mới)
* Tier2_Medium: trung bình
* Tier3_Old: nhiều rating

### 3.2 Bản chất

* Đây là **rule-based segmentation**
* Không phải thuật toán Data Mining

### 3.3 Vai trò

* Giải quyết cold-start
* Điều chỉnh trọng số hybrid

Ví dụ:

* User mới → ưu tiên Content + Demographic
* User lâu năm → ưu tiên CF

---

## 4. Đánh giá mô hình

Sử dụng các metric:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* Precision@K
* Recall@K
* NDCG@K

---

## 5. Đánh giá tổng thể bài toán Data Mining

### 5.1 Những phần là Data Mining

* Collaborative Filtering
* TF-IDF (Text Mining)

### 5.2 Những phần không phải Data Mining

* User segmentation (rule-based)
* Hybrid weighting (set tay)

---

## 6. Hướng cải thiện (nếu cần nâng điểm)

### 6.1 Dùng KMeans cho user segmentation

Thay vì rule-based, dùng clustering:

* Feature:

  * num_ratings
  * avg_rating

→ Tạo segment bằng thuật toán học

---

### 6.2 Dùng Matrix Factorization cho CF

Thay cosine similarity bằng:

* SVD

→ Tăng chất lượng dự đoán

---

### 6.3 Học trọng số Hybrid

Thay vì set tay:

* Dùng Linear Regression để học w1, w2, w3

---

## 7. Kết luận

* Hệ thống là **Hybrid Recommender System**
* Có áp dụng Data Mining ở mức cơ bản
* User segmentation hiện tại là heuristic, không phải học máy
* Có thể nâng cấp để tăng tính học thuật và điểm số

---

## 8. Câu trả lời ngắn gọn (để trình bày)

"Hệ thống sử dụng các kỹ thuật Data Mining gồm Collaborative Filtering, TF-IDF cho content-based filtering, và phân tích demographic. Các phương pháp này được kết hợp trong mô hình Hybrid Recommender System để cải thiện độ chính xác gợi ý."
