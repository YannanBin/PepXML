#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/8/19 15:28
# @Author  : zdj
# @FileName: test.py
# @Software: PyCharm

import argparse
import logging
import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from Utils import extract_embeddings_in_batches, device
from ESM2ClassifierforCluster import Classifier, predict
from MergePredictions import merge_predictions
from Evaluation import evaluate_predictions
import warnings
warnings.filterwarnings("ignore")


def predict_for_testset(args):
    """
    遍历 test_data 下的 cluster_x_test.csv 文件，加载对应模型进行预测
    """
    os.makedirs(args.output_dir, exist_ok=True)
    test_files = [f for f in os.listdir(args.test_dir) if f.startswith("cluster_") and f.endswith("_test.csv")]
    if not test_files:
        logging.warning(f"{args.test_dir} 中没有找到独立测试集文件")
        return

    for test_file in test_files:
        cluster_id = test_file.split('_')[1]
        test_path = os.path.join(args.test_dir, test_file)
        logging.info(f"处理独立测试集 {test_file}")

        # 读取测试数据
        test_df = pd.read_csv(test_path)
        if "Sequences" not in test_df.columns:
            logging.warning(f"{test_file} 缺少 'Sequences' 列，跳过")
            continue
        test_sequences = [(i, seq) for i, seq in enumerate(test_df["Sequences"].values)]

        # ESM2 embedding
        test_embeddings = extract_embeddings_in_batches(test_sequences, batch_size=args.batch_size)
        X_test = torch.tensor(test_embeddings, dtype=torch.float32).to(device)

        # 读取标签列数（从 cluster_data 中的 matrix 文件获取）
        cluster_matrix_file = os.path.join(args.cluster_dir, f"cluster_{cluster_id}_matrix.csv")
        cluster_matrix_df = pd.read_csv(cluster_matrix_file)
        label_names = cluster_matrix_df.columns[1:]
        n_labels = len(label_names)
        input_dim = X_test.shape[1]

        # 加载训练好的模型 (默认最后一折 fold_{n_folds-1}_model.pth)
        model_path = os.path.join(args.model_dir, f"cluster_{cluster_id}", f"fold_{args.n_folds-1}_model.pth")
        if not os.path.exists(model_path):
            logging.warning(f"未找到模型 {model_path}，跳过")
            continue

        model = Classifier(input_dim=input_dim, n_labels=n_labels,
                           hidden_dim=args.hidden_dim, dropout=args.dropout)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # 预测
        test_dataset = TensorDataset(X_test, torch.zeros((len(X_test), n_labels)))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        probs = predict(model, test_loader, device)

        # 保存预测结果
        result_df = pd.DataFrame({
            "Sequences": test_df["Sequences"],
            **{label_names[i]: probs[:, i] for i in range(n_labels)}
        })
        result_path = os.path.join(args.output_dir, f"cluster_{cluster_id}_independent_predictions.csv")
        result_df.to_csv(result_path, index=False)
        logging.info(f"独立测试集预测结果已保存: {result_path}")


def main(args):
    start_time = time.time()
    logging.info("开始独立测试集预测...")

    # 1. 独立测试集预测
    predict_for_testset(args)

    # 2. 合并独立测试集预测结果
    logging.info("合并独立测试集预测结果...")
    merge_predictions(args)

    # 3. 评估独立测试集预测结果
    logging.info("评估独立测试集预测结果...")
    evaluate_predictions(args)

    end_time = time.time()
    logging.info(f"总运行时间: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent test evaluation for each cluster model")

    # 输入输出路径
    parser.add_argument('--test_dir', type=str, default="test/test_data",
                        help='独立测试集目录 (包含 cluster_x_test.csv)')
    parser.add_argument('--cluster_dir', type=str, default="generate_data/cluster_data",
                        help='簇数据目录 (用于获取标签信息)')
    parser.add_argument('--model_dir', type=str, default="model",
                        help='训练好的模型保存目录')
    parser.add_argument('--output_dir', type=str, default="test/test_pre_data",
                        help='独立测试集预测结果保存目录')

    # 模型参数
    parser.add_argument('--n_folds', type=int, default=5,
                        help='交叉验证折数 (用于选择最后一折模型)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='预测批次大小')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout 比例')

    # merge 参数
    parser.add_argument('--prediction_dir', type=str, default="test/test_pre_data",
                        help='独立测试集预测结果目录')
    parser.add_argument('--label_matrix_file', type=str, default="test/label_matrix_test.csv",
                        help='原始标签矩阵 (用于合并时对齐标签)')
    parser.add_argument('--output_path', type=str, default="test/merged_test_predictions.csv",
                        help='最终合并的预测结果保存路径')

    # evaluation 参数
    parser.add_argument('--pred_path', type=str, default="test/merged_test_predictions.csv",
                        help='预测结果路径 (用于评估)')
    parser.add_argument('--true_path', type=str, default="test/label_matrix_test.csv",
                        help='真实标签路径')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 3, 5],
                        help='评估指标中的 top-k 列表')

    parser.add_argument('--log_file', type=str, default="log_file/test.log",
                        help='日志文件路径')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logging.basicConfig(
        filename=args.log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    main(args)
