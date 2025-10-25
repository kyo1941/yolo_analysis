import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# --- ユーザー設定 ---

# 1. 分析対象の JSON ファイル
# (サーバーで学習させたモデルで val() を実行した結果のパスを指定)
RESULT_DIR = "yolo12l_standard_val3"
PREDICTIONS_JSON_PATH = f"./runs/validation/{RESULT_DIR}/predictions.json"

# 2. 正解ラベル (Ground Truth) のディレクトリ
GT_LABELS_DIR = "./dataset_extracted/labels/val/"

# 3. data.yaml のパス (クラス名を取得するため)
DATA_YAML_PATH = "./dataset_extracted/data.yaml"

# 4. 分析の閾値
CONF_THRESHOLD = 0.001  # この信頼度未満の予測は「背景」として無視
IOU_THRESHOLD = 0.75    # 予測と正解の重複がこの値以上で「TP (正解)」とみなす

# 5. 画像サイズ (YOLO座標の変換に必要)
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# --- 設定ここまで ---


def yolo_to_xyxy(yolo_bbox, img_w, img_h):
    """YOLO形式 (center_x, center_y, w, h) [0-1] を xyxy [pixel] に変換"""
    x_center, y_center, w, h = yolo_bbox
    x_min = (x_center - w / 2) * img_w
    y_min = (y_center - h / 2) * img_h
    x_max = (x_center + w / 2) * img_w
    y_max = (y_center + h / 2) * img_h
    return [x_min, y_min, x_max, y_max]

def coco_to_xyxy(coco_bbox):
    """COCO形式 (x_min, y_min, w, h) [pixel] を xyxy [pixel] に変換"""
    x_min, y_min, w, h = coco_bbox
    return [x_min, y_min, x_min + w, y_min + h]

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

def load_ground_truth(gt_dir, img_w, img_h):
    """正解ラベル (YOLO .txt) を読み込み、画像IDごとに辞書化する"""
    print(f"正解ラベルを読み込み中: {gt_dir}")
    gt_data = {}
    gt_files = list(Path(gt_dir).glob("*.txt"))
    if not gt_files:
        print(f"警告: 正解ラベルが {gt_dir} に見つかりません。")
        return {}
        
    for txt_file in gt_files:
        image_id = txt_file.stem # 例: "001"
        bboxes = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                yolo_bbox = [float(p) for p in parts[1:]]
                xyxy_bbox = yolo_to_xyxy(yolo_bbox, img_w, img_h)
                bboxes.append({"class_id": class_id, "bbox": xyxy_bbox, "matched": False})
        gt_data[image_id] = bboxes
    print(f"{len(gt_data)} 件の正解ラベルをロードしました。")
    return gt_data

def load_predictions(json_path, conf_thresh):
    """予測JSONを読み込み、画像IDごとに辞書化する"""
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
            "bbox": coco_to_xyxy(p['bbox']),
            "score": p['score'],
            "matched": False
        })
    print(f"{len(preds)} 件の予測（閾値 {conf_thresh} 以上）をロードしました。")
    return pred_data

def analyze_image_difficulty(gt_data, pred_data, class_names, iou_thresh):
    """画像ごとにFP, FN, TPを集計し、困難な画像を特定する"""
    
    # 全ての画像IDのリスト (正解または予測のどちらかに存在する)
    all_image_ids = set(gt_data.keys()) | set(pred_data.keys())
    
    results = []
    
    print(f"全 {len(all_image_ids)} 画像の突合処理を開始します...")
    
    for image_id in all_image_ids:
        
        gt_boxes = gt_data.get(image_id, [])
        pred_boxes = pred_data.get(image_id, [])
        
        # マッチング前にリセット
        for box in gt_boxes: box['matched'] = False
        for box in pred_boxes: box['matched'] = False
        
        tp = 0
        fp = 0
        
        # 予測をスコア順にソート（重要）
        pred_boxes.sort(key=lambda x: x['score'], reverse=True)

        # マッチング処理 (TP と FP の計算)
        for p_box in pred_boxes:
            best_iou = 0
            best_gt_match_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                # クラスが一致し、まだマッチしていない正解ラベルが対象
                if p_box['class_id'] == gt_box['class_id'] and not gt_box['matched']:
                    iou = calculate_iou(p_box['bbox'], gt_box['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_match_idx = i
            
            if best_iou >= iou_thresh:
                # TP (True Positive)
                gt_boxes[best_gt_match_idx]['matched'] = True
                p_box['matched'] = True
                tp += 1
            else:
                # FP (False Positive)
                fp += 1
                
        # FN (False Negative) の計算
        # マッチしなかった正解ラベルの数
        fn = sum(1 for gt_box in gt_boxes if not gt_box['matched'])
        
        # 困難度スコア (FPとFNの合計)
        difficulty_score = fp + fn
        
        # クラスごとのFN内訳
        fn_details = {}
        for gt_box in gt_boxes:
            if not gt_box['matched']:
                class_name = class_names.get(gt_box['class_id'], "Unknown")
                fn_details[class_name] = fn_details.get(class_name, 0) + 1
        
        results.append({
            "image_id": image_id,
            "difficulty": difficulty_score,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "FN_Details": str(fn_details) # 見逃したクラスの内訳
        })

    # 困難な順にソート
    df = pd.DataFrame(results)
    df = df.sort_values(by="difficulty", ascending=False)
    
    return df

# --- メイン処理 ---
if __name__ == "__main__":
    
    # 1. クラス名を取得
    class_names = {}
    try:
        with open(DATA_YAML_PATH, 'r') as f:
            yaml_data = yaml.safe_load(f)
            class_names = yaml_data.get('names', {})
            if not class_names:
                print(f"エラー: {DATA_YAML_PATH} に 'names:' が見つかりません。")
                exit()
            print(f"クラス名をロードしました: {class_names}")
    except Exception as e:
        print(f"エラー: {DATA_YAML_PATH} の読み込みに失敗しました。 {e}")
        exit()

    # 2. 正解と予測をロード
    gt_data = load_ground_truth(GT_LABELS_DIR, IMAGE_WIDTH, IMAGE_HEIGHT)
    pred_data = load_predictions(PREDICTIONS_JSON_PATH, CONF_THRESHOLD)
    
    if not gt_data or not pred_data:
        print("エラー: 正解または予測データがロードできませんでした。処理を終了します。")
        exit()

    # 3. 分析実行
    difficulty_df = analyze_image_difficulty(gt_data, pred_data, class_names, IOU_THRESHOLD)
    
    # 4. 結果をCSVファイルに出力
    output_csv_path = "./difficulty_analysis.csv"
    difficulty_df.to_csv(output_csv_path, index=False)
    
    print("\n" + "="*50)
    print(f"分析が完了しました。")
    print(f"結果を {output_csv_path} に保存しました。")
    print("--- 最も困難だった画像 TOP 20 ---")
    print(difficulty_df.head(20).to_string())
    print("="*50)