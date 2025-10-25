# YOLO Analysis
## 概要
YOLOの予測結果を分析して、FP+FNでソートしたCSVを生成し、対応する画像を保存するスクリプト

## 使い方
1. 分析結果を生成する
`difficulty_analysis.csv`が生成される
- `score`（信頼度）の閾値を変更 -> `CONF_THRESHOLD`
- BBoxのIoU閾値を変更 -> `IOU_THRESHOLD`
```
python analyze_difficulty.py
```

2. 上位の指定した件数の画像を出力する
- `difficult_images_annotated`フォルダに画像が保存される
    - suffix _annotated -> BBoxあり
    - suffix _raw -> 元画像

- ワースト画像の枚数を変更 -> `NUM_TOP_IMAGES_TO_COPY`
- `score`（信頼度）の閾値を変更 -> `CONF_THRESHOLD`
- BBoxのIoU閾値を変更 -> `IOU_THRESHOLD`
```
python save_difficulty_images.py
```

### 仮想環境
自分用
```
source /Users/yourName/yourDirectory/venv/bin/activate
```

```
deactivate
```
