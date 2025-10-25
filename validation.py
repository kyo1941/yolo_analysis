import os
from ultralytics import YOLO
from pathlib import Path
import yaml # YAMLを扱うために追加

print("YOLOv12l (標準モデル) による検証（推論）を Mac / MPS で開始します。")
print("="*50)

# --- 1. データセットYAMLの準備 ---
# make_yaml を使わず、手動で作成したファイルのパスを直接指定する
data_yaml = "./dataset_extracted/data.yaml"

if not os.path.exists(data_yaml):
    print(f"エラー: data.yaml が見つかりません: {data_yaml}")
    print("指定されたパスに data.yaml を作成してください。")
    exit()
else:
    print(f"データセットYAMLを使用します: {data_yaml}")


# --- 2. 推論パラメータの設定 ---
imgsz = 1024 
batch_size = 4 # Macでも 4 や 8 が可能 (OOMになったら 1 に減らしてください)

# --- 3. モデルの初期化 (標準の yolo12l.pt) ---
weights_path = "yolo12l.pt" 
print(f"標準の事前学習済みモデル '{weights_path}' をロードします。")
try:
    model = YOLO(weights_path)
    print("モデルのロードが完了しました。")
except Exception as e:
    print(f"エラー: モデルのダウンロードまたはロードに失敗しました: {e}")
    exit()

print("="*50)

# --- 4. 検証の実行 (推論とメトリクス計算) ---
print(f"検証を開始します... [Device: MPS, Image Size: {imgsz}]")

try:
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch_size,
        device='mps',             # MacのGPU(MPS)を明示的に指定
        project="runs/validation",# 学習(train)とは別のフォルダに保存
        name="yolo12l_standard_val",
        save_json=True,           # COCO形式で結果をjson保存
        save_txt=True,            # ラベル形式で結果をtxt保存
        plots=True                # 混同行列(FP/FN)やPR曲線を画像として保存
    )
    
    print("="*50)
    print("検証が完了しました。")
    print(f"結果は 'runs/validation/yolo12l_standard_val' フォルダに保存されました。")
    
    print("\n--- 主要メトリクス ---")
    print(f"mAP50-95 (B): {metrics.box.map:.4f}")
    print(f"mAP50 (B):    {metrics.box.map50:.4f}")
    print(f"Precision (B):  {metrics.box.mp:.4f}")
    print(f"Recall (B):     {metrics.box.mr:.4f}")
    print("="*50)

except Exception as e:
    print(f"エラー: モデルの検証（推論）中に問題が発生しました。 {e}")
    print("data.yaml の 'val:' パスが正しいか確認してください。")