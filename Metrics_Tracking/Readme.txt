# Yêu cầu hệ thống
Python: Phiên bản 3.7

# Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết từ file:
pip install -r requirements.txt

# Eval với MOT17
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL test --TRACKERS_TO_EVAL MPNTrack --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 1

