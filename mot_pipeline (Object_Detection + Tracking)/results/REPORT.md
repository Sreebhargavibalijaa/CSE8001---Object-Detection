# MOT17 seven-sequence profiling

One copy of each training video (`*-DPM`). DPM / FRCNN / SDP folders share frames.

Chosen configuration: **YOLO26n imgsz=1280 + ByteTrack** (HOTA 41.3, recall 55.0%).
OC-SORT is tied on HOTA (41.4) with slightly lower recall. Faster R-CNN
MobileNetV3-Large 320-FPN is included as a CPU-practical two-stage baseline (16 FPS)
replacing ResNet-50 FPN (~0.13 FPS); its 320 px input loses small pedestrians.

Detection recall by height (IoU 0.5, all seven videos): YOLO 640 is 12.9% / 39.9% on
`<40px` / `40–80px`; YOLO 1280 is 47.4% / 61.3%. Tracker-threshold variants change
TrackEval recall by about one point.

Motion-assertion pass rate is 96.9–97.5% for every configuration, including Faster
R-CNN at HOTA 23, so it is not a substitute for HOTA / MOTA / IDF1 / Recall.

## Configuration means (unweighted over 7 sequences)

| detector | tracker | HOTA | MOTA | IDF1 | Recall | Precision | assertion_pass_rate | assertions_failed | assertions_checked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26n_imgsz1280 | ocsort | 41.4 | 44.6 | 51.0 | 53.6 | 87.1 | 96.9 | 1094 | 69262 |
| yolo26n_imgsz1280 | bytetrack | 41.3 | 44.7 | 50.2 | 55.0 | 85.6 | 97.0 | 1062 | 73281 |
| yolo26n_imgsz1280 | bytetrack_low | 40.8 | 44.3 | 49.4 | 54.0 | 86.4 | 97.0 | 1054 | 70491 |
| yolo26n_imgsz960 | bytetrack | 39.9 | 43.2 | 49.0 | 50.7 | 88.3 | 97.4 | 879 | 60304 |
| yolo26n_imgsz960 | ocsort | 39.6 | 42.3 | 48.8 | 48.9 | 89.6 | 97.3 | 883 | 55777 |
| yolo26n_imgsz960 | bytetrack_low | 39.3 | 42.2 | 48.0 | 49.4 | 88.9 | 97.4 | 854 | 57077 |
| yolo26n_imgsz640 | bytetrack | 36.3 | 37.1 | 43.4 | 43.1 | 89.9 | 97.2 | 858 | 43636 |
| yolo26n_imgsz640 | bytetrack_low | 35.7 | 36.3 | 42.3 | 42.0 | 90.5 | 97.1 | 833 | 41520 |
| yolo26n_imgsz640 | ocsort | 35.6 | 36.0 | 42.5 | 41.5 | 90.6 | 97.0 | 881 | 40899 |
| fasterrcnn_mv3_320 | bytetrack | 23.2 | 8.9 | 26.4 | 35.2 | 56.2 | 97.5 | 870 | 55253 |
| fasterrcnn_mv3_320 | ocsort | 23.1 | 11.5 | 25.4 | 34.4 | 59.1 | 97.3 | 886 | 49746 |
| fasterrcnn_mv3_320 | bytetrack_low | 22.9 | 10.6 | 25.9 | 34.6 | 58.1 | 97.5 | 824 | 50838 |

Motion-assertion pass rate is reported next to HOTA / MOTA / IDF1 / Recall so a
kinematically smooth tracker is not mistaken for an accurate one. High pass rate
with low Recall means the tracks that exist move like people, but most pedestrians
were never detected.

## Per-sequence motion assertion vs tracking accuracy

| sequence | detector | tracker | HOTA | MOTA | IDF1 | Recall | assertion_pass_rate | assertions_failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-DPM | yolo26n_imgsz640 | bytetrack | 29.9 | 21.2 | 32.7 | 23.8 | 99.0 | 50 |
| MOT17-02-DPM | yolo26n_imgsz640 | bytetrack_low | 29.0 | 20.5 | 30.5 | 22.5 | 99.0 | 48 |
| MOT17-02-DPM | yolo26n_imgsz640 | ocsort | 27.1 | 20.2 | 28.0 | 22.0 | 99.0 | 47 |
| MOT17-02-DPM | yolo26n_imgsz960 | bytetrack | 30.9 | 25.1 | 36.5 | 32.6 | 99.4 | 51 |
| MOT17-02-DPM | yolo26n_imgsz960 | bytetrack_low | 29.9 | 24.8 | 34.0 | 30.9 | 99.3 | 51 |
| MOT17-02-DPM | yolo26n_imgsz960 | ocsort | 31.9 | 25.6 | 36.9 | 30.6 | 99.2 | 54 |
| MOT17-02-DPM | yolo26n_imgsz1280 | bytetrack | 32.6 | 29.6 | 39.3 | 38.2 | 99.5 | 46 |
| MOT17-02-DPM | yolo26n_imgsz1280 | bytetrack_low | 30.6 | 30.0 | 36.8 | 36.6 | 99.4 | 50 |
| MOT17-02-DPM | yolo26n_imgsz1280 | ocsort | 33.2 | 29.8 | 40.0 | 35.9 | 99.4 | 51 |
| MOT17-02-DPM | fasterrcnn_mv3_320 | bytetrack | 15.0 | -4.9 | 13.9 | 18.8 | 99.5 | 41 |
| MOT17-02-DPM | fasterrcnn_mv3_320 | bytetrack_low | 15.8 | -2.7 | 15.0 | 17.9 | 99.5 | 36 |
| MOT17-02-DPM | fasterrcnn_mv3_320 | ocsort | 17.1 | -1.7 | 16.1 | 17.8 | 99.5 | 35 |
| MOT17-04-DPM | yolo26n_imgsz640 | bytetrack | 32.8 | 26.4 | 35.1 | 28.6 | 100.0 | 3 |
| MOT17-04-DPM | yolo26n_imgsz640 | bytetrack_low | 31.9 | 24.6 | 33.6 | 26.8 | 100.0 | 3 |
| MOT17-04-DPM | yolo26n_imgsz640 | ocsort | 32.2 | 24.0 | 34.4 | 26.0 | 100.0 | 4 |
| MOT17-04-DPM | yolo26n_imgsz960 | bytetrack | 42.7 | 41.9 | 50.5 | 46.0 | 100.0 | 6 |
| MOT17-04-DPM | yolo26n_imgsz960 | bytetrack_low | 41.2 | 39.3 | 48.9 | 43.2 | 100.0 | 7 |
| MOT17-04-DPM | yolo26n_imgsz960 | ocsort | 42.2 | 38.3 | 50.8 | 41.9 | 100.0 | 9 |
| MOT17-04-DPM | yolo26n_imgsz1280 | bytetrack | 46.3 | 49.0 | 54.3 | 58.7 | 100.0 | 3 |
| MOT17-04-DPM | yolo26n_imgsz1280 | bytetrack_low | 45.5 | 48.3 | 52.8 | 57.3 | 100.0 | 3 |
| MOT17-04-DPM | yolo26n_imgsz1280 | ocsort | 47.0 | 48.8 | 56.0 | 56.8 | 100.0 | 4 |
| MOT17-04-DPM | fasterrcnn_mv3_320 | bytetrack | 16.1 | 3.5 | 17.1 | 22.6 | 100.0 | 9 |
| MOT17-04-DPM | fasterrcnn_mv3_320 | bytetrack_low | 15.0 | 5.0 | 15.7 | 21.0 | 100.0 | 5 |
| MOT17-04-DPM | fasterrcnn_mv3_320 | ocsort | 15.2 | 5.5 | 15.5 | 20.7 | 100.0 | 5 |
| MOT17-05-DPM | yolo26n_imgsz640 | bytetrack | 43.5 | 47.2 | 59.5 | 59.6 | 89.3 | 506 |
| MOT17-05-DPM | yolo26n_imgsz640 | bytetrack_low | 43.8 | 47.3 | 60.2 | 59.0 | 89.3 | 496 |
| MOT17-05-DPM | yolo26n_imgsz640 | ocsort | 41.9 | 46.6 | 56.8 | 58.5 | 89.0 | 514 |
| MOT17-05-DPM | yolo26n_imgsz960 | bytetrack | 41.8 | 47.8 | 57.0 | 60.8 | 89.5 | 507 |
| MOT17-05-DPM | yolo26n_imgsz960 | bytetrack_low | 42.0 | 47.9 | 56.9 | 60.4 | 89.7 | 487 |
| MOT17-05-DPM | yolo26n_imgsz960 | ocsort | 41.3 | 48.6 | 56.0 | 60.2 | 89.4 | 505 |
| MOT17-05-DPM | yolo26n_imgsz1280 | bytetrack | 40.4 | 36.7 | 53.6 | 54.4 | 85.8 | 670 |
| MOT17-05-DPM | yolo26n_imgsz1280 | bytetrack_low | 40.7 | 36.6 | 53.9 | 54.6 | 86.0 | 659 |
| MOT17-05-DPM | yolo26n_imgsz1280 | ocsort | 39.3 | 37.3 | 53.9 | 54.3 | 86.0 | 657 |
| MOT17-05-DPM | fasterrcnn_mv3_320 | bytetrack | 34.0 | 27.1 | 43.5 | 53.7 | 89.8 | 526 |
| MOT17-05-DPM | fasterrcnn_mv3_320 | bytetrack_low | 33.4 | 27.5 | 42.7 | 53.6 | 90.2 | 496 |
| MOT17-05-DPM | fasterrcnn_mv3_320 | ocsort | 33.6 | 29.0 | 43.1 | 53.3 | 89.3 | 541 |
| MOT17-09-DPM | yolo26n_imgsz640 | bytetrack | 44.8 | 55.6 | 55.4 | 66.7 | 98.9 | 49 |
| MOT17-09-DPM | yolo26n_imgsz640 | bytetrack_low | 43.3 | 54.8 | 52.5 | 65.8 | 99.0 | 43 |
| MOT17-09-DPM | yolo26n_imgsz640 | ocsort | 42.1 | 55.1 | 51.8 | 65.2 | 98.9 | 48 |
| MOT17-09-DPM | yolo26n_imgsz960 | bytetrack | 49.5 | 61.3 | 61.5 | 70.4 | 99.1 | 45 |
| MOT17-09-DPM | yolo26n_imgsz960 | bytetrack_low | 48.7 | 61.1 | 60.5 | 69.7 | 99.2 | 41 |
| MOT17-09-DPM | yolo26n_imgsz960 | ocsort | 49.6 | 61.7 | 62.2 | 69.6 | 99.2 | 40 |
| MOT17-09-DPM | yolo26n_imgsz1280 | bytetrack | 49.3 | 61.7 | 59.5 | 71.7 | 99.0 | 51 |
| MOT17-09-DPM | yolo26n_imgsz1280 | bytetrack_low | 49.7 | 61.5 | 60.7 | 71.0 | 99.0 | 51 |
| MOT17-09-DPM | yolo26n_imgsz1280 | ocsort | 48.5 | 62.1 | 59.4 | 71.0 | 98.9 | 56 |
| MOT17-09-DPM | fasterrcnn_mv3_320 | bytetrack | 28.4 | 19.7 | 34.9 | 56.6 | 99.0 | 49 |
| MOT17-09-DPM | fasterrcnn_mv3_320 | bytetrack_low | 27.8 | 23.5 | 33.9 | 56.5 | 99.0 | 48 |
| MOT17-09-DPM | fasterrcnn_mv3_320 | ocsort | 29.3 | 24.3 | 33.3 | 56.2 | 99.0 | 48 |
| MOT17-10-DPM | yolo26n_imgsz640 | bytetrack | 28.5 | 33.4 | 35.4 | 37.5 | 99.8 | 13 |
| MOT17-10-DPM | yolo26n_imgsz640 | bytetrack_low | 28.3 | 32.4 | 35.3 | 36.4 | 99.8 | 12 |
| MOT17-10-DPM | yolo26n_imgsz640 | ocsort | 29.7 | 32.0 | 37.8 | 35.8 | 99.7 | 16 |
| MOT17-10-DPM | yolo26n_imgsz960 | bytetrack | 34.4 | 42.8 | 43.6 | 48.6 | 99.8 | 17 |
| MOT17-10-DPM | yolo26n_imgsz960 | bytetrack_low | 34.0 | 41.2 | 43.4 | 46.9 | 99.7 | 18 |
| MOT17-10-DPM | yolo26n_imgsz960 | ocsort | 33.4 | 41.2 | 42.9 | 46.6 | 99.7 | 20 |
| MOT17-10-DPM | yolo26n_imgsz1280 | bytetrack | 35.7 | 47.2 | 44.3 | 54.9 | 99.7 | 22 |
| MOT17-10-DPM | yolo26n_imgsz1280 | bytetrack_low | 34.2 | 46.2 | 41.7 | 53.8 | 99.7 | 22 |
| MOT17-10-DPM | yolo26n_imgsz1280 | ocsort | 35.5 | 46.0 | 44.9 | 53.0 | 99.7 | 23 |
| MOT17-10-DPM | fasterrcnn_mv3_320 | bytetrack | 16.9 | -6.7 | 15.8 | 26.1 | 99.9 | 8 |
| MOT17-10-DPM | fasterrcnn_mv3_320 | bytetrack_low | 17.1 | -4.4 | 15.7 | 25.6 | 99.9 | 5 |
| MOT17-10-DPM | fasterrcnn_mv3_320 | ocsort | 17.1 | -4.0 | 16.2 | 25.6 | 99.9 | 7 |
| MOT17-11-DPM | yolo26n_imgsz640 | bytetrack | 48.5 | 54.0 | 54.1 | 60.9 | 99.1 | 58 |
| MOT17-11-DPM | yolo26n_imgsz640 | bytetrack_low | 47.9 | 53.7 | 53.8 | 60.3 | 99.1 | 57 |
| MOT17-11-DPM | yolo26n_imgsz640 | ocsort | 50.3 | 53.3 | 58.3 | 59.9 | 98.9 | 68 |
| MOT17-11-DPM | yolo26n_imgsz960 | bytetrack | 48.4 | 54.0 | 53.2 | 63.8 | 99.2 | 57 |
| MOT17-11-DPM | yolo26n_imgsz960 | bytetrack_low | 48.4 | 53.9 | 53.2 | 63.5 | 99.1 | 58 |
| MOT17-11-DPM | yolo26n_imgsz960 | ocsort | 49.1 | 54.3 | 55.9 | 63.2 | 99.1 | 64 |
| MOT17-11-DPM | yolo26n_imgsz1280 | bytetrack | 49.8 | 51.7 | 54.2 | 65.1 | 99.0 | 70 |
| MOT17-11-DPM | yolo26n_imgsz1280 | bytetrack_low | 49.9 | 52.2 | 54.9 | 64.9 | 99.1 | 68 |
| MOT17-11-DPM | yolo26n_imgsz1280 | ocsort | 51.6 | 53.1 | 57.8 | 65.0 | 98.9 | 80 |
| MOT17-11-DPM | fasterrcnn_mv3_320 | bytetrack | 39.7 | 29.8 | 47.7 | 53.9 | 99.0 | 73 |
| MOT17-11-DPM | fasterrcnn_mv3_320 | bytetrack_low | 39.6 | 30.6 | 47.3 | 53.4 | 98.9 | 74 |
| MOT17-11-DPM | fasterrcnn_mv3_320 | ocsort | 37.6 | 32.1 | 42.5 | 53.5 | 98.8 | 79 |
| MOT17-13-DPM | yolo26n_imgsz640 | bytetrack | 26.0 | 21.8 | 31.3 | 24.4 | 94.0 | 179 |
| MOT17-13-DPM | yolo26n_imgsz640 | bytetrack_low | 25.9 | 20.9 | 30.1 | 23.1 | 93.8 | 174 |
| MOT17-13-DPM | yolo26n_imgsz640 | ocsort | 25.6 | 20.6 | 30.8 | 22.8 | 93.3 | 184 |
| MOT17-13-DPM | yolo26n_imgsz960 | bytetrack | 31.4 | 29.3 | 40.7 | 33.0 | 95.2 | 196 |
| MOT17-13-DPM | yolo26n_imgsz960 | bytetrack_low | 30.7 | 27.5 | 38.9 | 31.3 | 95.0 | 192 |
| MOT17-13-DPM | yolo26n_imgsz960 | ocsort | 29.7 | 26.8 | 37.1 | 30.3 | 94.9 | 191 |
| MOT17-13-DPM | yolo26n_imgsz1280 | bytetrack | 35.3 | 37.1 | 46.1 | 41.9 | 96.2 | 200 |
| MOT17-13-DPM | yolo26n_imgsz1280 | bytetrack_low | 35.0 | 35.4 | 44.9 | 39.7 | 95.9 | 201 |
| MOT17-13-DPM | yolo26n_imgsz1280 | ocsort | 34.5 | 35.4 | 45.2 | 39.4 | 95.4 | 223 |
| MOT17-13-DPM | fasterrcnn_mv3_320 | bytetrack | 12.5 | -6.3 | 11.5 | 14.5 | 95.3 | 164 |
| MOT17-13-DPM | fasterrcnn_mv3_320 | bytetrack_low | 11.9 | -5.6 | 10.7 | 14.1 | 95.1 | 160 |
| MOT17-13-DPM | fasterrcnn_mv3_320 | ocsort | 12.1 | -4.5 | 10.8 | 13.8 | 94.5 | 171 |

## Detection recall by pedestrian height (IoU 0.5, before tracking)

| detector | sequence | height_bin | gt | tp | recall |
| --- | --- | --- | --- | --- | --- |
| yolo26n_imgsz640 | MOT17-02-DPM | <40px | 222 | 0 | 0.0 |
| yolo26n_imgsz640 | MOT17-02-DPM | 40-80px | 5817 | 1560 | 26.8 |
| yolo26n_imgsz640 | MOT17-02-DPM | 80-160px | 8519 | 4378 | 51.4 |
| yolo26n_imgsz640 | MOT17-02-DPM | >=160px | 4023 | 3321 | 82.6 |
| yolo26n_imgsz960 | MOT17-02-DPM | <40px | 222 | 25 | 11.3 |
| yolo26n_imgsz960 | MOT17-02-DPM | 40-80px | 5817 | 2291 | 39.4 |
| yolo26n_imgsz960 | MOT17-02-DPM | 80-160px | 8519 | 4963 | 58.3 |
| yolo26n_imgsz960 | MOT17-02-DPM | >=160px | 4023 | 3391 | 84.3 |
| yolo26n_imgsz1280 | MOT17-02-DPM | <40px | 222 | 34 | 15.3 |
| yolo26n_imgsz1280 | MOT17-02-DPM | 40-80px | 5817 | 2702 | 46.5 |
| yolo26n_imgsz1280 | MOT17-02-DPM | 80-160px | 8519 | 5261 | 61.8 |
| yolo26n_imgsz1280 | MOT17-02-DPM | >=160px | 4023 | 3388 | 84.2 |
| fasterrcnn_mv3_320 | MOT17-02-DPM | <40px | 222 | 0 | 0.0 |
| fasterrcnn_mv3_320 | MOT17-02-DPM | 40-80px | 5817 | 248 | 4.3 |
| fasterrcnn_mv3_320 | MOT17-02-DPM | 80-160px | 8519 | 1504 | 17.7 |
| fasterrcnn_mv3_320 | MOT17-02-DPM | >=160px | 4023 | 3005 | 74.7 |
| yolo26n_imgsz640 | MOT17-04-DPM | 80-160px | 13396 | 4327 | 32.3 |
| yolo26n_imgsz640 | MOT17-04-DPM | >=160px | 34161 | 24015 | 70.3 |
| yolo26n_imgsz960 | MOT17-04-DPM | 80-160px | 13396 | 5895 | 44.0 |
| yolo26n_imgsz960 | MOT17-04-DPM | >=160px | 34161 | 26871 | 78.7 |
| yolo26n_imgsz1280 | MOT17-04-DPM | 80-160px | 13396 | 6243 | 46.6 |
| yolo26n_imgsz1280 | MOT17-04-DPM | >=160px | 34161 | 27767 | 81.3 |
| fasterrcnn_mv3_320 | MOT17-04-DPM | 80-160px | 13396 | 1152 | 8.6 |
| fasterrcnn_mv3_320 | MOT17-04-DPM | >=160px | 34161 | 13110 | 38.4 |
| yolo26n_imgsz640 | MOT17-05-DPM | <40px | 99 | 68 | 68.7 |
| yolo26n_imgsz640 | MOT17-05-DPM | 40-80px | 971 | 624 | 64.3 |
| yolo26n_imgsz640 | MOT17-05-DPM | 80-160px | 1914 | 1263 | 66.0 |
| yolo26n_imgsz640 | MOT17-05-DPM | >=160px | 3933 | 3125 | 79.5 |
| yolo26n_imgsz960 | MOT17-05-DPM | <40px | 99 | 72 | 72.7 |
| yolo26n_imgsz960 | MOT17-05-DPM | 40-80px | 971 | 641 | 66.0 |
| yolo26n_imgsz960 | MOT17-05-DPM | 80-160px | 1914 | 1303 | 68.1 |
| yolo26n_imgsz960 | MOT17-05-DPM | >=160px | 3933 | 3116 | 79.2 |
| yolo26n_imgsz1280 | MOT17-05-DPM | <40px | 99 | 68 | 68.7 |
| yolo26n_imgsz1280 | MOT17-05-DPM | 40-80px | 971 | 632 | 65.1 |
| yolo26n_imgsz1280 | MOT17-05-DPM | 80-160px | 1914 | 1320 | 69.0 |
| yolo26n_imgsz1280 | MOT17-05-DPM | >=160px | 3933 | 2860 | 72.7 |
| fasterrcnn_mv3_320 | MOT17-05-DPM | <40px | 99 | 15 | 15.2 |
| fasterrcnn_mv3_320 | MOT17-05-DPM | 40-80px | 971 | 263 | 27.1 |
| fasterrcnn_mv3_320 | MOT17-05-DPM | 80-160px | 1914 | 1015 | 53.0 |
| fasterrcnn_mv3_320 | MOT17-05-DPM | >=160px | 3933 | 3074 | 78.2 |
| yolo26n_imgsz640 | MOT17-09-DPM | 80-160px | 204 | 142 | 69.6 |
| yolo26n_imgsz640 | MOT17-09-DPM | >=160px | 5121 | 3767 | 73.6 |
| yolo26n_imgsz960 | MOT17-09-DPM | 80-160px | 204 | 144 | 70.6 |
| yolo26n_imgsz960 | MOT17-09-DPM | >=160px | 5121 | 3829 | 74.8 |
| yolo26n_imgsz1280 | MOT17-09-DPM | 80-160px | 204 | 150 | 73.5 |
| yolo26n_imgsz1280 | MOT17-09-DPM | >=160px | 5121 | 3881 | 75.8 |
| fasterrcnn_mv3_320 | MOT17-09-DPM | 80-160px | 204 | 85 | 41.7 |
| fasterrcnn_mv3_320 | MOT17-09-DPM | >=160px | 5121 | 3264 | 63.7 |
| yolo26n_imgsz640 | MOT17-10-DPM | <40px | 68 | 2 | 2.9 |
| yolo26n_imgsz640 | MOT17-10-DPM | 40-80px | 4051 | 1740 | 43.0 |
| yolo26n_imgsz640 | MOT17-10-DPM | 80-160px | 5500 | 3697 | 67.2 |
| yolo26n_imgsz640 | MOT17-10-DPM | >=160px | 3220 | 2726 | 84.7 |
| yolo26n_imgsz960 | MOT17-10-DPM | <40px | 68 | 20 | 29.4 |
| yolo26n_imgsz960 | MOT17-10-DPM | 40-80px | 4051 | 2776 | 68.5 |
| yolo26n_imgsz960 | MOT17-10-DPM | 80-160px | 5500 | 3893 | 70.8 |
| yolo26n_imgsz960 | MOT17-10-DPM | >=160px | 3220 | 2776 | 86.2 |
| yolo26n_imgsz1280 | MOT17-10-DPM | <40px | 68 | 25 | 36.8 |
| yolo26n_imgsz1280 | MOT17-10-DPM | 40-80px | 4051 | 2973 | 73.4 |
| yolo26n_imgsz1280 | MOT17-10-DPM | 80-160px | 5500 | 3917 | 71.2 |
| yolo26n_imgsz1280 | MOT17-10-DPM | >=160px | 3220 | 2806 | 87.1 |
| fasterrcnn_mv3_320 | MOT17-10-DPM | <40px | 68 | 0 | 0.0 |
| fasterrcnn_mv3_320 | MOT17-10-DPM | 40-80px | 4051 | 155 | 3.8 |
| fasterrcnn_mv3_320 | MOT17-10-DPM | 80-160px | 5500 | 1875 | 34.1 |
| fasterrcnn_mv3_320 | MOT17-10-DPM | >=160px | 3220 | 2241 | 69.6 |
| yolo26n_imgsz640 | MOT17-11-DPM | 40-80px | 342 | 1 | 0.3 |
| yolo26n_imgsz640 | MOT17-11-DPM | 80-160px | 1650 | 466 | 28.2 |
| yolo26n_imgsz640 | MOT17-11-DPM | >=160px | 7444 | 6051 | 81.3 |
| yolo26n_imgsz960 | MOT17-11-DPM | 40-80px | 342 | 5 | 1.5 |
| yolo26n_imgsz960 | MOT17-11-DPM | 80-160px | 1650 | 672 | 40.7 |
| yolo26n_imgsz960 | MOT17-11-DPM | >=160px | 7444 | 6082 | 81.7 |
| yolo26n_imgsz1280 | MOT17-11-DPM | 40-80px | 342 | 16 | 4.7 |
| yolo26n_imgsz1280 | MOT17-11-DPM | 80-160px | 1650 | 717 | 43.5 |
| yolo26n_imgsz1280 | MOT17-11-DPM | >=160px | 7444 | 6100 | 81.9 |
| fasterrcnn_mv3_320 | MOT17-11-DPM | 40-80px | 342 | 0 | 0.0 |
| fasterrcnn_mv3_320 | MOT17-11-DPM | 80-160px | 1650 | 168 | 10.2 |
| fasterrcnn_mv3_320 | MOT17-11-DPM | >=160px | 7444 | 5497 | 73.8 |
| yolo26n_imgsz640 | MOT17-13-DPM | <40px | 324 | 22 | 6.8 |
| yolo26n_imgsz640 | MOT17-13-DPM | 40-80px | 5114 | 2579 | 50.4 |
| yolo26n_imgsz640 | MOT17-13-DPM | 80-160px | 4599 | 3274 | 71.2 |
| yolo26n_imgsz640 | MOT17-13-DPM | >=160px | 1605 | 1284 | 80.0 |
| yolo26n_imgsz960 | MOT17-13-DPM | <40px | 324 | 139 | 42.9 |
| yolo26n_imgsz960 | MOT17-13-DPM | 40-80px | 5114 | 3314 | 64.8 |
| yolo26n_imgsz960 | MOT17-13-DPM | 80-160px | 4599 | 3499 | 76.1 |
| yolo26n_imgsz960 | MOT17-13-DPM | >=160px | 1605 | 1308 | 81.5 |
| yolo26n_imgsz1280 | MOT17-13-DPM | <40px | 324 | 211 | 65.1 |
| yolo26n_imgsz1280 | MOT17-13-DPM | 40-80px | 5114 | 3670 | 71.8 |
| yolo26n_imgsz1280 | MOT17-13-DPM | 80-160px | 4599 | 3640 | 79.1 |
| yolo26n_imgsz1280 | MOT17-13-DPM | >=160px | 1605 | 1320 | 82.2 |
| fasterrcnn_mv3_320 | MOT17-13-DPM | <40px | 324 | 0 | 0.0 |
| fasterrcnn_mv3_320 | MOT17-13-DPM | 40-80px | 5114 | 538 | 10.5 |
| fasterrcnn_mv3_320 | MOT17-13-DPM | 80-160px | 4599 | 1517 | 33.0 |
| fasterrcnn_mv3_320 | MOT17-13-DPM | >=160px | 1605 | 990 | 61.7 |
