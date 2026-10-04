#!/usr/bin/env python3
"""
Local MOT Benchmark Evaluation Script
Computes MOTA, IDF1, HOTA, False Positives (FP), False Negatives (FN),
Identity Switches (IDSW), Precision, and Recall for MOT17 video sequences.
"""

import os
import argparse
import numpy as np
import pandas as pd
import motmetrics as mm
import trackeval


def calculate_iou_matrix(gt_boxes, pr_boxes):
    """
    Computes pairwise IoU between ground truth boxes and predicted boxes.
    Boxes are in [x, y, w, h] format.
    """
    if len(gt_boxes) == 0 or len(pr_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pr_boxes)), dtype=float)

    g_x1 = gt_boxes[:, 0][:, None]
    g_y1 = gt_boxes[:, 1][:, None]
    g_w = gt_boxes[:, 2][:, None]
    g_h = gt_boxes[:, 3][:, None]
    g_x2 = g_x1 + g_w
    g_y2 = g_y1 + g_h

    p_x1 = pr_boxes[:, 0][None, :]
    p_y1 = pr_boxes[:, 1][None, :]
    p_w = pr_boxes[:, 2][None, :]
    p_h = pr_boxes[:, 3][None, :]
    p_x2 = p_x1 + p_w
    p_y2 = p_y1 + p_h

    inter_x1 = np.maximum(g_x1, p_x1)
    inter_y1 = np.maximum(g_y1, p_y1)
    inter_x2 = np.minimum(g_x2, p_x2)
    inter_y2 = np.minimum(g_y2, p_y2)

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    g_area = g_w * g_h
    p_area = p_w * p_h
    union = g_area + p_area - inter_area

    iou = np.where(union > 0, inter_area / union, 0.0)
    return iou


def load_mot_gt(gt_path, min_visibility=0.0):
    """
    Loads MOT17 ground truth txt file.
    Filters for pedestrian class (ClassId == 1) and active flag (Consider == 1).
    """
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    gt = pd.read_csv(gt_path, header=None)
    gt.columns = [
        'FrameId', 'Id', 'X', 'Y', 'Width', 'Height',
        'Consider', 'ClassId', 'Visibility'
    ]

    # Official MOT17 filter: pedestrians only (ClassId == 1), consider == 1
    gt_filtered = gt[(gt['ClassId'] == 1) & (gt['Consider'] == 1)]
    if min_visibility > 0:
        gt_filtered = gt_filtered[gt_filtered['Visibility'] >= min_visibility]

    return gt_filtered


def load_mot_predictions(pred_path, min_conf=0.0):
    """
    Loads MOT tracker predictions from a standard formatted file:
    FrameId, Id, X, Y, Width, Height, Conf, ...
    """
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    # Can have variable columns; load first 7
    df = pd.read_csv(pred_path, header=None)
    col_names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf']
    for i in range(len(col_names)):
        df.rename(columns={i: col_names[i]}, inplace=True)

    if 'Conf' in df.columns and min_conf > 0.0:
        df = df[df['Conf'] >= min_conf]

    return df


def track_detections_iou(det_path, iou_thresh=0.25, min_conf=0.0):
    """
    Converts pure detection proposals (like det.txt where ID == -1)
    into a continuous track sequence using greedy IoU matching.
    """
    det = pd.read_csv(det_path, header=None)
    col_names = ['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf']
    for i in range(len(col_names)):
        det.rename(columns={i: col_names[i]}, inplace=True)

    if min_conf > 0:
        det = det[det['Conf'] >= min_conf]

    next_id = 1
    active_tracks = {}  # id -> [x, y, w, h]
    tracked_rows = []

    def iou_single(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2])
        y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = b1[2] * b1[3] + b2[2] * b2[3] - inter
        return inter / union if union > 0 else 0.0

    frames = sorted(det['FrameId'].unique())
    for fid in frames:
        frame_dets = det[det['FrameId'] == fid]
        current_active = {}
        used_ids = set()

        for _, row in frame_dets.iterrows():
            dbox = [row['X'], row['Y'], row['Width'], row['Height']]
            best_id = None
            best_iou = iou_thresh

            for tid, tbox in active_tracks.items():
                if tid in used_ids:
                    continue
                score = iou_single(dbox, tbox)
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_id is None:
                best_id = next_id
                next_id += 1

            used_ids.add(best_id)
            current_active[best_id] = dbox
            tracked_rows.append([
                fid, best_id, dbox[0], dbox[1], dbox[2], dbox[3], row['Conf']
            ])

        active_tracks = current_active

    return pd.DataFrame(tracked_rows, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf'])


def evaluate_tracking(gt_df, pred_df, max_iou=0.5):
    """
    Computes all standard tracking metrics:
    - MOTA, IDF1, FP, FN, IDSW, Precision, Recall (via motmetrics & CLEAR)
    - HOTA, DetA, AssA (via TrackEval)
    """
    # 1. Evaluate with motmetrics
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_df['FrameId'].unique()) | set(pred_df['FrameId'].unique()))

    for fid in all_frames:
        g = gt_df[gt_df['FrameId'] == fid]
        p = pred_df[pred_df['FrameId'] == fid]

        g_ids = g['Id'].values
        g_boxes = g[['X', 'Y', 'Width', 'Height']].values

        p_ids = p['Id'].values
        p_boxes = p[['X', 'Y', 'Width', 'Height']].values

        dist = mm.distances.iou_matrix(g_boxes, p_boxes, max_iou=max_iou)
        acc.update(g_ids, p_ids, dist)

    mh = mm.metrics.create()
    mm_summary = mh.compute(
        acc,
        metrics=[
            'mota', 'idf1', 'num_false_positives', 'num_misses',
            'num_switches', 'precision', 'recall'
        ]
    )

    # 2. Evaluate with TrackEval for HOTA
    unique_gt_ids = {raw_id: idx for idx, raw_id in enumerate(gt_df['Id'].unique())}
    unique_pr_ids = {raw_id: idx for idx, raw_id in enumerate(pred_df['Id'].unique())}

    gt_ids_list = []
    pr_ids_list = []
    sim_scores_list = []
    total_gt_dets = 0
    total_pr_dets = 0

    for fid in all_frames:
        g = gt_df[gt_df['FrameId'] == fid]
        p = pred_df[pred_df['FrameId'] == fid]

        g_ids = np.array([unique_gt_ids[x] for x in g['Id'].values], dtype=int)
        p_ids = np.array([unique_pr_ids[x] for x in p['Id'].values], dtype=int)

        g_boxes = g[['X', 'Y', 'Width', 'Height']].values.astype(float)
        p_boxes = p[['X', 'Y', 'Width', 'Height']].values.astype(float)

        sim = calculate_iou_matrix(g_boxes, p_boxes)

        gt_ids_list.append(g_ids)
        pr_ids_list.append(p_ids)
        sim_scores_list.append(sim)

        total_gt_dets += len(g_ids)
        total_pr_dets += len(p_ids)

    data = {
        'num_gt_dets': total_gt_dets,
        'num_tracker_dets': total_pr_dets,
        'num_gt_ids': len(unique_gt_ids),
        'num_tracker_ids': len(unique_pr_ids),
        'gt_ids': gt_ids_list,
        'tracker_ids': pr_ids_list,
        'similarity_scores': sim_scores_list,
        'num_timesteps': len(all_frames)
    }

    hota_eval = trackeval.metrics.HOTA()
    h_res = hota_eval.eval_sequence(data)

    results = {
        'HOTA': float(np.mean(h_res['HOTA']) * 100),
        'DetA': float(np.mean(h_res['DetA']) * 100),
        'AssA': float(np.mean(h_res['AssA']) * 100),
        'MOTA': float(mm_summary['mota'].iloc[0] * 100),
        'IDF1': float(mm_summary['idf1'].iloc[0] * 100),
        'FP': int(mm_summary['num_false_positives'].iloc[0]),
        'FN': int(mm_summary['num_misses'].iloc[0]),
        'IDSW': int(mm_summary['num_switches'].iloc[0]),
        'Precision': float(mm_summary['precision'].iloc[0] * 100),
        'Recall': float(mm_summary['recall'].iloc[0] * 100),
    }

    return results


def run_yolo_tracking_sequence(model_path, seq_dir, output_pred_path, imgsz=640, device='cpu'):
    """
    Runs YOLO + ByteTrack on sequence images, formats into MOTChallenge format,
    and saves to output_pred_path.
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    img_dir = os.path.join(seq_dir, "img1")
    image_names = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    rows = []
    print(f"Tracking with YOLO on {seq_dir} ({len(image_names)} frames)...")

    for idx, img_name in enumerate(image_names, start=1):
        img_path = os.path.join(img_dir, img_name)
        results = model.track(
            img_path,
            persist=True,
            verbose=False,
            imgsz=imgsz,
            device=device,
            classes=[0]  # Pedestrians (person class)
        )

        for res in results:
            if res.boxes is None or len(res.boxes) == 0:
                continue

            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            ids = (
                res.boxes.id.cpu().numpy()
                if res.boxes.id is not None
                else np.arange(1, len(boxes) + 1)
            )

            for box, conf, track_id in zip(boxes, confs, ids):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                rows.append([idx, int(track_id), float(x1), float(y1), float(w), float(h), float(conf)])

    df = pd.DataFrame(rows, columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Conf'])
    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)
    df.to_csv(output_pred_path, header=False, index=False)
    print(f"Saved tracking predictions to: {output_pred_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOTA, IDF1, HOTA, FP, FN on MOT17 sequences")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/Users/sreebhargavibalija/Desktop/8001/MOT17/train",
        help="Path to MOT17 train directory"
    )
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="/Users/sreebhargavibalija/Desktop/8001/yolo26n.pt",
        help="Path to YOLO weights file"
    )
    parser.add_argument(
        "--eval_baselines",
        action="store_true",
        default=True,
        help="Evaluate baseline detector proposals (DPM, FRCNN, SDP)"
    )
    parser.add_argument(
        "--eval_yolo",
        action="store_true",
        help="Run and evaluate YOLO ByteTrack on generated sequence(s)"
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="*",
        default=None,
        help="Specific sequences to evaluate (e.g. MOT17-02-DPM). If omitted, evaluates all available in dataset_dir."
    )
    args = parser.parse_args()

    pred_dir = os.path.join(args.dataset_dir, "..", "runs", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    all_sequences = sorted([
        d for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d)) and os.path.exists(os.path.join(args.dataset_dir, d, "gt", "gt.txt"))
    ])

    if args.sequences:
        all_sequences = [s for s in all_sequences if s in args.sequences]

    print(f"\n=======================================================")
    print(f" Found {len(all_sequences)} sequences with Ground Truth:")
    for s in all_sequences:
        print(f"  - {s}")
    print(f"=======================================================\n")

    summary_records = []

    for seq in all_sequences:
        seq_path = os.path.join(args.dataset_dir, seq)
        gt_path = os.path.join(seq_path, "gt", "gt.txt")
        gt_df = load_mot_gt(gt_path)

        # 1. Baseline detector evaluation
        det_path = os.path.join(seq_path, "det", "det.txt")
        if args.eval_baselines and os.path.exists(det_path):
            print(f"Evaluating Baseline Tracker on: {seq} (det.txt)...")
            base_pred_df = track_detections_iou(det_path, iou_thresh=0.25, min_conf=0.0)
            res = evaluate_tracking(gt_df, base_pred_df)
            res['Sequence'] = seq
            res['Tracker'] = f"Baseline Detector ({seq.split('-')[-1]}) + IoU Track"
            summary_records.append(res)

        # 2. YOLO ByteTrack evaluation
        if args.eval_yolo and os.path.exists(args.yolo_weights):
            yolo_pred_file = os.path.join(pred_dir, f"{seq}_yolo_bytetrack.txt")
            if not os.path.exists(yolo_pred_file):
                yolo_pred_df = run_yolo_tracking_sequence(args.yolo_weights, seq_path, yolo_pred_file)
            else:
                print(f"Loading existing YOLO predictions: {yolo_pred_file}")
                yolo_pred_df = load_mot_predictions(yolo_pred_file)

            res = evaluate_tracking(gt_df, yolo_pred_df)
            res['Sequence'] = seq
            res['Tracker'] = "YOLO (yolo26n) + ByteTrack"
            summary_records.append(res)

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        cols_order = [
            'Sequence', 'Tracker', 'HOTA', 'MOTA', 'IDF1',
            'FP', 'FN', 'IDSW', 'Precision', 'Recall', 'DetA', 'AssA'
        ]
        df_summary = df_summary[cols_order]

        print("\n\n=========================================================================================")
        print("                                FINAL BENCHMARK RESULTS")
        print("=========================================================================================")
        print(df_summary.to_string(index=False))

        out_csv = os.path.join(pred_dir, "benchmark_metrics_summary.csv")
        df_summary.to_csv(out_csv, index=False)
        print(f"\nSaved full benchmark results to: {out_csv}")


if __name__ == "__main__":
    main()
