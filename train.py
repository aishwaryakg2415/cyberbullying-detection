"""
=============================================================================
  Sarcasm-Aware Explainable Early Detection of Cyberbullying
  Module 5: Model Training Pipeline
  IEEE Research Paper Implementation
=============================================================================
"""

import os
import json
import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from datasets import DatasetDict
import evaluate

from config import (
    CYBERBULLYING_MODEL, TRAINING_CONFIG, MODELS_DIR, RESULTS_DIR,
    SEED, ID2LABEL, LABEL2ID
)
from dataset import CyberbullyingDataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fix seed
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────

accuracy_metric  = evaluate.load("accuracy")
f1_metric        = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric    = evaluate.load("recall")


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute all classification metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy"  : accuracy_metric.compute(
            predictions=predictions, references=labels)["accuracy"],
        "f1"        : f1_metric.compute(
            predictions=predictions, references=labels,
            average="binary")["f1"],
        "precision" : precision_metric.compute(
            predictions=predictions, references=labels,
            average="binary")["precision"],
        "recall"    : recall_metric.compute(
            predictions=predictions, references=labels,
            average="binary")["recall"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

class SACENetTrainer:
    """
    Fine-tuning pipeline for the SACENet cyberbullying classifier.

    Trains a Twitter-RoBERTa backbone on the multi-platform corpus
    with cosine LR schedule, early stopping, and automatic metric logging.
    """

    def __init__(
        self,
        base_model  : str = CYBERBULLYING_MODEL,
        output_dir  : str = MODELS_DIR,
        config      : Optional[Dict] = None,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.config     = config or TRAINING_CONFIG
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

        logger.info(f"Loading tokenizer & model: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels    = 2,
            id2label      = ID2LABEL,
            label2id      = LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    def train(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        """
        Fine-tune the model on the provided HuggingFace DatasetDict.

        Args:
            dataset_dict : HF DatasetDict with 'train', 'val', 'test' splits

        Returns:
            dict with train_result, eval_results, and test_results
        """
        training_args = TrainingArguments(
            output_dir                  = self.output_dir,
            num_train_epochs            = self.config["num_train_epochs"],
            per_device_train_batch_size = self.config["per_device_train_batch_size"],
            per_device_eval_batch_size  = self.config["per_device_eval_batch_size"],
            learning_rate               = self.config["learning_rate"],
            weight_decay                = self.config["weight_decay"],
            warmup_ratio                = self.config["warmup_ratio"],
            lr_scheduler_type           = self.config["lr_scheduler_type"],
            eval_strategy               = self.config["eval_strategy"],
            save_strategy               = self.config["save_strategy"],
            load_best_model_at_end      = self.config["load_best_model_at_end"],
            metric_for_best_model       = self.config["metric_for_best_model"],
            fp16                        = (
                self.config["fp16"] and torch.cuda.is_available()
            ),
            seed                        = SEED,
            logging_steps               = self.config["logging_steps"],
            report_to                   = self.config["report_to"],
            save_total_limit            = 2,
        )

        trainer = Trainer(
            model            = self.model,
            args             = training_args,
            train_dataset    = dataset_dict["train"],
            eval_dataset     = dataset_dict["val"],
            tokenizer        = self.tokenizer,
            data_collator    = DataCollatorWithPadding(self.tokenizer),
            compute_metrics  = compute_metrics,
            callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
        )

        logger.info("="*60)
        logger.info("  Starting SACENet Fine-Tuning")
        logger.info("="*60)
        train_result = trainer.train()

        # Evaluate on validation set
        logger.info("Evaluating on validation set …")
        val_results  = trainer.evaluate(eval_dataset=dataset_dict["val"])

        # Evaluate on test set
        logger.info("Evaluating on test set …")
        test_results = trainer.evaluate(eval_dataset=dataset_dict["test"])

        # Save best model
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        # Save results to JSON
        all_results = {
            "training_metrics" : train_result.metrics,
            "val_metrics"      : val_results,
            "test_metrics"     : test_results,
        }
        results_path = os.path.join(RESULTS_DIR, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        logger.info("="*60)
        logger.info(f"  Training complete! Model saved to: {self.output_dir}")
        logger.info(f"  Test F1: {test_results.get('eval_f1', 'N/A'):.4f}")
        logger.info(f"  Test Accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
        logger.info("="*60)

        return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build dataset
    logger.info("Building multi-platform dataset …")
    ds = CyberbullyingDataset(use_sample_corpus=True, balance=True, augment=True)

    logger.info("Tokenising dataset …")
    hf_dataset = ds.get_hf_dataset(tokenizer_name=CYBERBULLYING_MODEL)

    # Train
    trainer = SACENetTrainer(base_model=CYBERBULLYING_MODEL)
    results = trainer.train(hf_dataset)

    print("\n📊 Training Results:")
    print(json.dumps(results["test_metrics"], indent=2))
