import os
import shutil
import pandas as pd
from pathlib import Path
import json
import cv2  # OpenCV をインポート
import numpy as np
import yaml

# --- ユーザー設定 ---
# 1. 分析結果のCSVファイル
DIFFICULTY_CSV_PATH = "./difficulty_analysis.csv"

# 2. 元の画像があるディレクトリ
SOURCE_IMAGE_DIR = "./dataset_extracted/images/val/"

# 3. 正解ラベル (Ground Truth) のディレクトリ
GT_LABELS_DIR = "./dataset_extracted/labels/val/"

# 4. 予測結果 (Prediction) のJSONファイル
PREDICTIONS_JSON_PATH = "./runs/validation/yolo12l_standard_val3/predictions.json" # analyze_difficulty.py と同じパスを指定

# 5. data.yaml のパス (クラス名を取得するため)
DATA_YAML_PATH = "./dataset_extracted/data.yaml"

# 6. コピー先のディレクトリ (新規作成されます)
DESTINATION_DIR = "./difficult_images_annotated/"

# 7. コピーする上位画像の数
NUM_TOP_IMAGES_TO_COPY = 50 # 例: 困難度上位50枚

# 8. 描画設定
CONF_THRESHOLD = 0.001  # この信頼度未満の「予測」は描画しない
IOU_THRESHOLD = 0.75    # TP/FP/FN判定の閾値（analyze_difficulty.py と合わせる）
IMAGE_WIDTH = 512     # 画像サイズ
IMAGE_HEIGHT = 512    # 画像サイズ

# 色 (B, G, R)
COLOR_GT = (0, 255, 0)     # 正解 (Green) - 常に描画
COLOR_TP = (0, 255, 255)   # 正しく検出 (Yellow) - 予測が正解とマッチ
COLOR_FP = (0, 0, 255)     # 誤検出 (Red) - 予測が正解とマッチしない
# --- 設定ここまで ---


# --- analyze_difficulty.py から移植したヘルパー関数 ---

def load_class_names(yaml_path):
    """data.yaml からクラス名をロード"""
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            class_names = yaml_data.get('names', {})
            if not class_names:
                raise ValueError(f"'names:' が {yaml_path} に見つかりません。")
            print(f"クラス名をロードしました: {class_names}")
            return class_names
    except Exception as e:
        print(f"エラー: {yaml_path} の読み込みに失敗しました。 {e}")
        exit()

def yolo_to_xyxy(yolo_bbox, img_w, img_h):
    """YOLO形式 (center_x, center_y, w, h) [0-1] を xyxy [pixel] に変換"""
    x_center, y_center, w, h = yolo_bbox
    x_min = int((x_center - w / 2) * img_w)
    y_min = int((y_center - h / 2) * img_h)
    x_max = int((x_center + w / 2) * img_w)
    y_max = int((y_center + h / 2) * img_h)
    return [x_min, y_min, x_max, y_max]

def coco_to_xyxy(coco_bbox):
    """COCO形式 (x_min, y_min, w, h) [pixel] を xyxy [pixel] に変換"""
    x_min, y_min, w, h = coco_bbox
    return [int(x_min), int(y_min), int(x_min + w), int(y_min + h)]

def calculate_iou(boxA, boxB):
    """2つのBBox (xyxy形式) の IoU を計算"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def load_predictions_dict(json_path, conf_thresh):
    """予測JSONを読み込み、画像IDごとの辞書にする"""
    print(f"予測JSONを読み込み中: {json_path}")
    if not Path(json_path).exists():
        print(f"エラー: 予測JSONファイルが見つかりません: {json_path}")
        return {}
        
    with open(json_path, 'r') as f:
        preds = json.load(f)
    
    pred_data = {}
    for p in preds:
        if p['score'] < conf_thresh:
            continue # 信頼度が低い予測は無視
        
        image_id = Path(p['file_name']).stem
        
        if image_id not in pred_data:
            pred_data[image_id] = []
            
        pred_data[image_id].append({
            "class_id": p['category_id'], 
            "bbox_coco": p['bbox'],
            "bbox_xyxy": coco_to_xyxy(p['bbox']),
            "score": p['score'],
            "matched": False # TP判定用フラグ
        })
    print(f"{len(preds)} 件の予測（閾値 {conf_thresh} 以上）をロードしました。")
    return pred_data

def find_image_path(image_id, source_dir):
    """画像IDから実際の画像ファイルパスを探す"""
    base_path = Path(source_dir) / image_id
    for ext in ['.png', '.jpg', '.jpeg']:
        if (base_path.with_suffix(ext)).exists():
            return base_path.with_suffix(ext)
    return None

def draw_bbox(image, bbox, label, color, thickness=2):
    """画像にBBoxとラベルを描画する"""
    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[2], bbox[3])
    cv2.rectangle(image, pt1, pt2, color, thickness)
    
    # ラベルテキストの背景
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (pt1[0], pt1[1] - text_h - baseline), (pt1[0] + text_w, pt1[1]), color, -1)
    # ラベルテキスト
    cv2.putText(image, label, (pt1[0], pt1[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1) # 黒文字

# --- メイン処理 ---
if __name__ == "__main__":

    # 0. クラス名と予測をロード
    CLASS_NAMES = load_class_names(DATA_YAML_PATH)
    PRED_DATA = load_predictions_dict(PREDICTIONS_JSON_PATH, CONF_THRESHOLD)

    # 1. CSVファイルを読み込む
    try:
        df = pd.read_csv(DIFFICULTY_CSV_PATH)
        df = df.sort_values(by="difficulty", ascending=False)
        print(f"{DIFFICULTY_CSV_PATH} を読み込みました。")
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {DIFFICULTY_CSV_PATH}")
        exit()
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました。 {e}")
        exit()

    # 2. コピー先ディレクトリを作成
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    print(f"コピー先ディレクトリを作成（または確認）しました: {DESTINATION_DIR}")

    # 3. 上位画像をコピー & 描画
    copied_count = 0
    print(f"困難度上位 {NUM_TOP_IMAGES_TO_COPY} 枚の画像を処理します...")

    for index, row in df.head(NUM_TOP_IMAGES_TO_COPY).iterrows():
        image_id = row['image_id']
        difficulty = row['difficulty']

        # --- 元画像のパスを探す ---
        source_path = find_image_path(image_id, SOURCE_IMAGE_DIR)
        if not source_path:
            print(f"  警告: 元画像が見つかりません: {image_id}")
            continue

        # --- A. 元の画像（天然）をコピー ---
        dest_path_raw = Path(DESTINATION_DIR) / f"{image_id}_raw{source_path.suffix}"
        try:
            shutil.copy2(source_path, dest_path_raw)
        except Exception as e:
            print(f"  コピー失敗 (Raw): {source_path.name} - {e}")
            continue # 元画像がコピーできないなら描画もスキップ

        # --- B. BBox描画版の画像を作成 ---
        try:
            img = cv2.imread(str(source_path))
            if img is None:
                print(f"  エラー: 画像ファイルの読み込みに失敗: {source_path.name}")
                continue

            # BBox描画用のリストを取得
            gt_boxes = []
            gt_txt_path = Path(GT_LABELS_DIR) / f"{image_id}.txt"
            if gt_txt_path.exists():
                with open(gt_txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        xyxy_bbox = yolo_to_xyxy([float(p) for p in parts[1:]], IMAGE_WIDTH, IMAGE_HEIGHT)
                        gt_boxes.append({"class_id": class_id, "bbox": xyxy_bbox, "matched": False})

            pred_boxes = PRED_DATA.get(image_id, [])
            for p in pred_boxes: p['matched'] = False # マッチングフラグをリセット

            # --- TP/FP判定 (analyze_difficulty.py のロジック) ---
            for p_box in sorted(pred_boxes, key=lambda x: x['score'], reverse=True):
                best_iou = 0
                best_gt_match_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if p_box['class_id'] == gt_box['class_id'] and not gt_box['matched']:
                        iou = calculate_iou(p_box['bbox_xyxy'], gt_box['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_match_idx = i
                
                if best_iou >= IOU_THRESHOLD:
                    gt_boxes[best_gt_match_idx]['matched'] = True
                    p_box['matched'] = True

            # --- 描画レイヤー1: 正解を全て緑で描画 (ベースレイヤー) ---
            for gt_box in gt_boxes:
                label = f"GT: {CLASS_NAMES.get(gt_box['class_id'], gt_box['class_id'])}"
                draw_bbox(img, gt_box['bbox'], label, COLOR_GT, thickness=2)

            # --- 描画レイヤー2: 予測をTP(黄)/FP(赤)で描画 (オーバーレイ) ---
            for p_box in pred_boxes:
                if p_box['matched']:
                    # TP: 正しく検出された予測 (黄色)
                    label = f"TP: {CLASS_NAMES.get(p_box['class_id'], p_box['class_id'])} {p_box['score']:.2f}"
                    draw_bbox(img, p_box['bbox_xyxy'], label, COLOR_TP, thickness=2)
                else:
                    # FP: 誤検出 (赤色)
                    label = f"FP: {CLASS_NAMES.get(p_box['class_id'], p_box['class_id'])} {p_box['score']:.2f}"
                    draw_bbox(img, p_box['bbox_xyxy'], label, COLOR_FP, thickness=2)

            # --- 描画した画像を保存 ---
            dest_path_annotated = Path(DESTINATION_DIR) / f"{image_id}_annotated{source_path.suffix}"
            cv2.imwrite(str(dest_path_annotated), img)
            
            print(f"  処理成功: {image_id} (Difficulty: {difficulty}) -> Raw + Annotated")
            copied_count += 1

        except Exception as e:
            print(f"  描画エラー: {source_path.name} - {e}")


    print("\n" + "="*50)
    print(f"処理が完了しました。")
    print(f"合計 {copied_count} / {NUM_TOP_IMAGES_TO_COPY} ペアの画像を {DESTINATION_DIR} に保存しました。")
    print("="*50)
    print("描画ルール:")
    print(f"  緑 (GT): 正解 - 常に表示")
    print(f"  黄 (TP): 正しく検出された予測 - 正解と重なる")
    print(f"  赤 (FP): 誤検出 - 正解と重ならない")
    print(f"\n※ 緑のみ = FN（見逃し）")
    print(f"※ 緑+黄 = TP（正検出）")
    print(f"※ 赤のみ = FP（誤検出）")