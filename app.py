"""
Flask Application for Hybrid Movie Recommendation System
Sử dụng Model Manager để cung cấp API gợi ý phim
"""

import sys
import os
import json
import logging
from datetime import datetime

# Thêm model folder vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

from flask import Flask, request, jsonify
from flask_cors import CORS
from model_manager_sql import HybridRecommendationManager  # SQLite version

# =========================================================
# CẤU HÌNH LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# KHỞI TẠO FLASK APP
# =========================================================
app = Flask(__name__)
CORS(app)  # Cho phép CORS cho tất cả routes

# =========================================================
# KHỞI TẠO MODEL MANAGER (Global)
# =========================================================
try:
    logger.info("[App] Khởi tạo Model Manager...")
    manager = HybridRecommendationManager()
    logger.info("[App] ✓ Model Manager khởi tạo thành công!")
except Exception as e:
    logger.error(f"[App] ✗ Lỗi khởi tạo Model Manager: {str(e)}")
    manager = None

# =========================================================
# HEALTH CHECK ENDPOINT
# =========================================================
@app.route('/health', methods=['GET'])
def health_check():
    """
    Kiểm tra trạng thái của API
    """
    return jsonify({
        "status": "OK" if manager else "ERROR",
        "timestamp": datetime.now().isoformat(),
        "message": "Model Manager is ready" if manager else "Model Manager not initialized"
    }), 200 if manager else 503

# =========================================================
# RECOMMENDATIONS API
# =========================================================
@app.route('/api/recommendations/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    """
    Lấy danh sách gợi ý cho một người dùng
    
    Parameters:
        user_id (int): ID của người dùng
        k (int, optional): Số gợi ý (mặc định: 10)
        exclude_watched (bool, optional): Loại bỏ phim đã xem (mặc định: true)
    
    Returns:
        JSON: Danh sách phim được gợi ý
    """
    
    if not manager:
        return jsonify({"error": "Model Manager not initialized"}), 503
    
    try:
        # Lấy parameters
        k = request.args.get('k', default=10, type=int)
        exclude_watched = request.args.get('exclude_watched', default=True, type=bool)
        
        # Validate
        if k < 1 or k > 100:
            return jsonify({"error": "k must be between 1 and 100"}), 400
        
        logger.info(f"[API] Lấy gợi ý cho user {user_id} (k={k})")
        
        # Lấy gợi ý
        recommendations = manager.get_recommendations(
            user_id=user_id,
            k=k,
            exclude_watched=exclude_watched
        )
        
        # Format response
        response = {
            "user_id": user_id,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations[[
                "movie_id", "title", "clean_title", "genres", "year", 
                "predicted_rating", "score"
            ]].to_dict(orient='records')
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"[API] Lỗi lấy gợi ý: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# USER PROFILE API
# =========================================================
@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user_profile(user_id):
    """
    Lấy thông tin hồ sơ của một người dùng
    
    Parameters:
        user_id (int): ID của người dùng
    
    Returns:
        JSON: Thông tin người dùng (tuổi, giới tính, quốc gia, tier, số tương tác)
    """
    
    if not manager:
        return jsonify({"error": "Model Manager not initialized"}), 503
    
    try:
        logger.info(f"[API] Lấy profil user {user_id}")
        
        profile = manager.get_user_profile(user_id)
        
        if profile is None:
            return jsonify({"error": f"User {user_id} not found"}), 404
        
        return jsonify(profile), 200
    
    except Exception as e:
        logger.error(f"[API] Lỗi lấy profil user: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# EVALUATION METRICS API
# =========================================================
@app.route('/api/evaluation-metrics', methods=['GET'])
def get_evaluation_metrics():
    """
    Lấy các chỉ số đánh giá mô hình
    
    Returns:
        JSON: RMSE, MAE, Precision, Recall, NDCG, MAP per tier
    """
    
    try:
        logger.info("[API] Lấy evaluation metrics")
        
        with open("model/evaluation_metrics.json", "r") as f:
            metrics = json.load(f)
        
        return jsonify(metrics), 200
    
    except FileNotFoundError:
        return jsonify({"error": "Evaluation metrics not found. Run phase 3-4 first."}), 404
    except Exception as e:
        logger.error(f"[API] Lỗi lấy metrics: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# HYBRID WEIGHTS API
# =========================================================
@app.route('/api/hybrid-weights', methods=['GET'])
def get_hybrid_weights():
    """
    Lấy trọng số cho từng tầng người dùng
    
    Returns:
        JSON: Weights cho CF, Content-based, Demographic-based per tier
    """
    
    try:
        logger.info("[API] Lấy hybrid weights")
        
        with open("model/hybrid_weights.json", "r") as f:
            weights = json.load(f)
        
        return jsonify(weights), 200
    
    except FileNotFoundError:
        return jsonify({"error": "Hybrid weights not found. Run phase 3 first."}), 404
    except Exception as e:
        logger.error(f"[API] Lỗi lấy weights: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# BATCH RECOMMENDATIONS API
# =========================================================
@app.route('/api/batch-recommendations', methods=['POST'])
def batch_recommendations():
    """
    Lấy gợi ý cho nhiều người dùng cùng lúc
    
    Body (JSON):
        {
            "user_ids": [1, 2, 3, ...],
            "k": 10
        }
    
    Returns:
        JSON: Gợi ý cho tất cả users
    """
    
    if not manager:
        return jsonify({"error": "Model Manager not initialized"}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'user_ids' not in data:
            return jsonify({"error": "user_ids required in request body"}), 400
        
        user_ids = data['user_ids']
        k = data.get('k', 10)
        
        if not isinstance(user_ids, list) or len(user_ids) == 0:
            return jsonify({"error": "user_ids must be a non-empty list"}), 400
        
        logger.info(f"[API] Lấy gợi ý cho {len(user_ids)} users")
        
        results = {}
        for user_id in user_ids:
            try:
                recommendations = manager.get_recommendations(user_id, k=k)
                results[str(user_id)] = {
                    "total": len(recommendations),
                    "recommendations": recommendations[[
                        "movie_id", "title", "clean_title", "genres", "year", 
                        "predicted_rating", "score"
                    ]].to_dict(orient='records')
                }
            except Exception as e:
                results[str(user_id)] = {"error": str(e)}
        
        return jsonify(results), 200
    
    except Exception as e:
        logger.error(f"[API] Lỗi batch recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# CACHE STATISTICS API
# =========================================================
@app.route('/api/cache-stats', methods=['GET'])
def cache_stats():
    """
    Lấy thống kê về cache
    
    Returns:
        JSON: Số lượng cache entries, memory usage
    """
    
    if not manager:
        return jsonify({"error": "Model Manager not initialized"}), 503
    
    try:
        logger.info("[API] Lấy cache stats")
        
        stats = manager.get_cache_stats()
        
        return jsonify(stats), 200
    
    except Exception as e:
        logger.error(f"[API] Lỗi lấy cache stats: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# CLEAR CACHE API
# =========================================================
@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """
    Xóa toàn bộ cache (dùng cho maintenance)
    
    Returns:
        JSON: Confirmation message
    """
    
    if not manager:
        return jsonify({"error": "Model Manager not initialized"}), 503
    
    try:
        logger.info("[API] Xóa cache")
        manager.clear_cache()
        
        return jsonify({"message": "Cache cleared successfully"}), 200
    
    except Exception as e:
        logger.error(f"[API] Lỗi xóa cache: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# SYSTEM INFO API
# =========================================================
@app.route('/api/system-info', methods=['GET'])
def system_info():
    """
    Lấy thông tin hệ thống
    
    Returns:
        JSON: Thông tin về mô hình, dữ liệu, performance
    """
    
    try:
        logger.info("[API] Lấy system info")
        
        # Load evaluation metrics để lấy thông tin tổng
        with open("model/evaluation_metrics.json", "r") as f:
            metrics = json.load(f)
        
        info = {
            "system": "Hybrid Movie Recommendation System",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "model_status": "READY" if manager else "ERROR",
            "data": {
                "total_ratings": metrics['global']['train_set_size'] + metrics['global']['test_set_size'],
                "train_set": metrics['global']['train_set_size'],
                "test_set": metrics['global']['test_set_size']
            },
            "performance": {
                "rmse": metrics['global']['rmse'],
                "mae": metrics['global']['mae']
            },
            "cache": manager.get_cache_stats() if manager else None
        }
        
        return jsonify(info), 200
    
    except Exception as e:
        logger.error(f"[API] Lỗi lấy system info: {str(e)}")
        return jsonify({"error": str(e)}), 500

# =========================================================
# ERROR HANDLERS
# =========================================================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

# =========================================================
# CLI ENDPOINT (DEV)
# =========================================================
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """
    Test endpoint để verify hệ thống hoạt động
    """
    if not manager:
        return jsonify({"error": "Model Manager not initialized"}), 503
    
    try:
        # Test với user đầu tiên
        test_user = 1
        profile = manager.get_user_profile(test_user)
        recommendations = manager.get_recommendations(test_user, k=3)
        
        return jsonify({
            "message": "System is working correctly",
            "test_user": test_user,
            "user_profile": profile,
            "sample_recommendations": recommendations[[
                "title", "year", "predicted_rating"
            ]].to_dict(orient='records')
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("HYBRID MOVIE RECOMMENDATION SYSTEM - FLASK API")
    logger.info("=" * 70)
    
    if not manager:
        logger.error("⚠️  Model Manager không thể khởi tạo!")
        logger.error("   Đã chạy phase 1-2 chưa? (tạo mô hình đã chưa?)")
        sys.exit(1)
    
    logger.info("\n✓ Hệ thống sẵn sàng!")
    logger.info("\n📚 API Endpoints:")
    logger.info("  GET  /health                          - Health check")
    logger.info("  GET  /api/recommendations/<user_id>   - Get recommendations")
    logger.info("  GET  /api/user/<user_id>              - Get user profile")
    logger.info("  GET  /api/evaluation-metrics          - Get metrics")
    logger.info("  GET  /api/hybrid-weights              - Get weights")
    logger.info("  POST /api/batch-recommendations       - Batch recommendations")
    logger.info("  GET  /api/cache-stats                 - Cache statistics")
    logger.info("  POST /api/cache/clear                 - Clear cache")
    logger.info("  GET  /api/system-info                 - System info")
    logger.info("  GET  /api/test                        - Test endpoint")
    logger.info("\n🚀 Starting server on http://127.0.0.1:5000")
    logger.info("=" * 70 + "\n")
    
    # Chạy Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Tắt debug mode trong production
        use_reloader=False
    )
