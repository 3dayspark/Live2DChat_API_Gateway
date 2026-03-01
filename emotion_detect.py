

import os
import logging
import numpy as np
from sklearn.metrics import f1_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 設定定数 ---
PRETRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pretrained_models', 'chinese-hubert-base') #PRETRAINED_MODEL_PATH = r'D:\python_project\sovits_api_deepseek\bert-base-chinese'
DATASET_FILE = 'data.csv'
OUTPUT_DIR = 'training_dir'

# ラベル定義（参考情報として保持、実際のデータセットは数値化済みであることを想定）
TARGET_MAP = {
    'happiness': 0,
    'sadness': 1,
    'anger': 2,
    'disgust': 3,
    'like': 4,
    'surprise': 5,
    'fear': 6
}

def compute_metrics(logits_and_labels):
    """
    評価用メトリクス（正解率とF1スコア）を計算する関数
    :param logits_and_labels: モデルの出力ロジットと正解ラベルのタプル
    :return: 計算されたメトリクスの辞書
    """
    logits, labels = logits_and_labels
    predictions = np.argmax(logits, axis=-1)
    
    # 正解率 (Accuracy) の計算
    acc = np.mean(predictions == labels)
    # F1スコア (Macro Average) の計算
    f1 = f1_score(labels, predictions, average='macro')
    
    return {'accuracy': acc, 'f1': f1}

def main():
    logger.info("Starting emotion detection model training...")

    # 1. データセットのロード
    # data.csv は既に前処理（ラベルの数値化等）が完了している前提
    if not os.path.exists(DATASET_FILE):
        logger.error(f"Dataset file not found: {DATASET_FILE}")
        return

    logger.info(f"Loading dataset from {DATASET_FILE}...")
    raw_datasets = load_dataset('csv', data_files=DATASET_FILE)

    # 2. データセットの分割
    # トレーニング用とテスト用に分割（テストデータ: 20%）
    split = raw_datasets['train'].train_test_split(test_size=0.2, seed=42)
    logger.info("Dataset split completed.")

    # 3. トークナイザーの準備
    logger.info(f"Loading tokenizer from {PRETRAINED_MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    def tokenize_fn(batch):
        """データセット内のテキストをトークン化する処理"""
        return tokenizer(batch['text'], truncation=True)

    # データセット全体にトークン化を適用
    tokenized_datasets = split.map(tokenize_fn, batched=True)
    logger.info("Tokenization completed.")

    # 4. モデルの初期化
    logger.info(f"Loading model from {PRETRAINED_MODEL_PATH}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL_PATH,
            num_labels=len(TARGET_MAP)  # クラス数: 7
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 5. トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,            # 出力ディレクトリ
        evaluation_strategy='epoch',      # エポックごとに評価
        save_strategy='epoch',            # エポックごとにモデル保存
        num_train_epochs=5,               # 学習エポック数
        per_device_train_batch_size=16,   # トレーニング時のバッチサイズ
        per_device_eval_batch_size=64,    # 評価時のバッチサイズ
        logging_dir=f'{OUTPUT_DIR}/logs', # ログ保存先
        logging_steps=10
    )

    # 6. Trainerの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 7. トレーニング開始
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed.")

    # モデルの最終保存（オプション）
    # trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))

if __name__ == "__main__":
    main()