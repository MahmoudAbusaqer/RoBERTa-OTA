import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, set_seed as transformers_set_seed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder
import time
import re
import json
from datetime import datetime
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    """Set all random seeds for COMPLETE reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # PYTHON RANDOM SEEDS
    random.seed(seed)
    np.random.seed(seed)

    # PYTORCH SEEDS
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PYTORCH DETERMINISTIC BEHAVIOR
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # TRANSFORMERS LIBRARY SEED
    transformers_set_seed(seed)

    print(f"COMPLETE REPRODUCIBILITY: All seeds set to {seed}")
    print(f"Python hash seed: {os.environ.get('PYTHONHASHSEED')}")
    print(f"CUDA deterministic: {torch.backends.cudnn.deterministic}")
    print(f"CUDA benchmark: {torch.backends.cudnn.benchmark}")

class Timer:
    """timer for detailed performance monitoring"""
    def __init__(self):
        self.epoch_times = []
        self.fold_times = []
        self.epoch_start = None
        self.fold_start = None

    def start_epoch(self):
        self.epoch_start = time.time()

    def end_epoch(self):
        if self.epoch_start:
            epoch_time = time.time() - self.epoch_start
            self.epoch_times.append(epoch_time)
            return epoch_time
        return 0

    def start_fold(self):
        self.fold_start = time.time()

    def end_fold(self):
        if self.fold_start:
            fold_time = time.time() - self.fold_start
            self.fold_times.append(fold_time)
            return fold_time
        return 0

    def get_stats(self):
        return {
            'avg_fold_time': np.mean(self.fold_times) if self.fold_times else 0,
            'std_fold_time': np.std(self.fold_times) if self.fold_times else 0,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'std_epoch_time': np.std(self.epoch_times) if self.epoch_times else 0,
            'total_epochs': len(self.epoch_times),
            'total_folds': len(self.fold_times)
        }

class HateSpeechDataset(Dataset):
    """Dataset class"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RobertaClassifier(nn.Module):
    """RoBERTa Classifier"""
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super(RobertaClassifier, self).__init__()
        self.model_name = 'roberta-base'

        # RoBERTa model
        self.transformer = RobertaModel.from_pretrained('roberta-base')

        # classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state.mean(dim=1)

        output = self.dropout(pooled_output)
        return self.classifier(output)

def calculate_comprehensive_metrics(true_labels, predictions, probabilities=None):
    """Calculate comprehensive classification metrics"""
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1_macro': f1_score(true_labels, predictions, average='macro'),
        'f1_micro': f1_score(true_labels, predictions, average='micro'),
        'f1_weighted': f1_score(true_labels, predictions, average='weighted'),
        'precision_macro': precision_score(true_labels, predictions, average='macro'),
        'precision_micro': precision_score(true_labels, predictions, average='micro'),
        'precision_weighted': precision_score(true_labels, predictions, average='weighted'),
        'recall_macro': recall_score(true_labels, predictions, average='macro'),
        'recall_micro': recall_score(true_labels, predictions, average='micro'),
        'recall_weighted': recall_score(true_labels, predictions, average='weighted'),
    }

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predictions, average=None)
    metrics['per_class'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist()
    }

    # AUC metrics
    if probabilities is not None:
        try:
            metrics['auc_macro'] = roc_auc_score(true_labels, probabilities,
                                               average='macro', multi_class='ovr')
            metrics['auc_weighted'] = roc_auc_score(true_labels, probabilities,
                                                  average='weighted', multi_class='ovr')
        except Exception as e:
            metrics['auc_macro'] = None
            metrics['auc_weighted'] = None

    return metrics

def get_gpu_memory():
    """Get GPU memory usage statistics"""
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}

def count_parameters(model):
    """Count and categorize model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Break down by component
    param_breakdown = {}
    if hasattr(model, 'transformer'):
        param_breakdown['roberta'] = sum(p.numel() for p in model.transformer.parameters())
    if hasattr(model, 'classifier'):
        param_breakdown['classifier'] = sum(p.numel() for p in model.classifier.parameters())

    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'breakdown': param_breakdown,
        'total_mb': total_params * 4 / 1024**2,
        'trainable_mb': trainable_params * 4 / 1024**2
    }

def train_model(model, train_loader, val_loader, device, num_epochs=10, patience=3, fold_num=1, timer=None):
    """Training function"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training history tracking
    fold_train_metrics = []
    fold_val_metrics = []

    print(f"    Training started...")

    for epoch in range(num_epochs):
        if timer:
            timer.start_epoch()

        # Training
        model.train()
        total_train_loss = 0
        all_train_predictions = []
        all_train_labels = []
        all_train_probabilities = []

        progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Get predictions and probabilities for comprehensive metrics
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_train_predictions.extend(predictions.detach().cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())
            all_train_probabilities.extend(probabilities.detach().cpu().numpy())

            progress_bar.set_postfix({'loss': loss.item()})

        epoch_time = timer.end_epoch() if timer else 0

        # Calculate comprehensive training metrics
        avg_train_loss = total_train_loss / len(train_loader)
        train_metrics = calculate_comprehensive_metrics(all_train_labels, all_train_predictions, all_train_probabilities)
        train_metrics['loss'] = avg_train_loss
        train_metrics['time'] = epoch_time
        fold_train_metrics.append(train_metrics)

        # Validation
        model.eval()
        total_val_loss = 0
        all_val_predictions = []
        all_val_labels = []
        all_val_probabilities = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation', leave=False)
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                # Get predictions and probabilities for comprehensive metrics
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                all_val_predictions.extend(predictions.detach().cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probabilities.extend(probabilities.detach().cpu().numpy())

                progress_bar.set_postfix({'val_loss': loss.item()})

        # Calculate comprehensive validation metrics
        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics = calculate_comprehensive_metrics(all_val_labels, all_val_predictions, all_val_probabilities)
        val_metrics['loss'] = avg_val_loss
        val_metrics['predictions'] = all_val_predictions
        val_metrics['true_labels'] = all_val_labels
        val_metrics['probabilities'] = all_val_probabilities
        fold_val_metrics.append(val_metrics)

        # GPU memory tracking
        gpu_memory = get_gpu_memory()

        # Print comprehensive epoch results
        print(f"    Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Train F1-W: {train_metrics['f1_weighted']:.4f} | "
              f"Val F1-W: {val_metrics['f1_weighted']:.4f} | "
              f"Val AUC-W: {val_metrics.get('auc_weighted', 0):.4f} | "
              f"GPU: {gpu_memory['allocated_gb']:.1f}GB | "
              f"Time: {epoch_time:.1f}s")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_val_metrics = val_metrics.copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch+1}")
                break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    total_training_time = sum([m['time'] for m in fold_train_metrics])
    print(f"     Training completed: {total_training_time:.1f}s total, "
          f"Avg: {total_training_time/len(fold_train_metrics):.1f}s/epoch")

    return model, total_training_time, best_val_metrics, fold_train_metrics, fold_val_metrics

def clean_text(text):
    """Enhanced text cleaning"""
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def run_roberta_evaluation_with_comprehensive_output(X, y, label_encoder, device):
    """RoBERTa evaluation with comprehensive analysis"""
    print(f"\n{'='*80}")
    print(f"ROBERTA EVALUATION")
    print(f"{'='*80}")

    model_start_time = time.time()
    timer = Timer()

    # RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_results = []

    # Store comprehensive results
    detailed_results = {
        'model_name': 'RoBERTa-base',
        'dataset_info': {
            'total_samples': len(X),
            'num_classes': len(label_encoder.classes_),
            'class_names': list(label_encoder.classes_),
            'class_distribution': {name: int(np.sum(y == i)) for i, name in enumerate(label_encoder.classes_)}
        },
        'training_config': {
            'max_length': 128,
            'batch_size': 16,
            'num_epochs': 10,
            'n_splits': 5,
            'learning_rate': 2e-5,
            'optimizer': 'AdamW',
            'weight_decay': 0.01,
            'architecture': 'RoBERTa'
        },
        'fold_results': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n FOLD {fold+1}/5")
        print(f" Train samples: {len(train_idx)}")
        print(f" Validation samples: {len(val_idx)}")

        timer.start_fold()
        fold_start_time = time.time()

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create datasets
        train_dataset = HateSpeechDataset(X_train.values, y_train, tokenizer)
        val_dataset = HateSpeechDataset(X_val.values, y_val, tokenizer)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                                num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                              num_workers=0, pin_memory=False)

        # Initialize RoBERTa model
        model = RobertaClassifier(num_classes=len(label_encoder.classes_)).to(device)

        # Model parameters info (only for first fold)
        if fold == 0:
            param_info = count_parameters(model)
            detailed_results['model_parameters'] = param_info
            print(f"\n RoBERTa Architecture:")
            print(f"   Total parameters: {param_info['total']:,}")
            print(f"   Trainable parameters: {param_info['trainable']:,}")
            print(f"   Model size: {param_info['total_mb']:.1f} MB")
            print(f"   Component breakdown:")
            for component, params in param_info['breakdown'].items():
                print(f"     {component}: {params:,} ({params/param_info['total']*100:.1f}%)")

        # Train model
        model, training_time, best_val_metrics, fold_train_metrics, fold_val_metrics = train_model(
            model, train_loader, val_loader, device, fold_num=fold+1, timer=timer
        )

        fold_time = timer.end_fold()
        total_fold_time = time.time() - fold_start_time
        gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'best_epoch': len(fold_val_metrics),
            'best_val_loss': best_val_metrics['loss'],
            'best_metrics': best_val_metrics,
            'training_time': training_time,
            'total_fold_time': total_fold_time,
            'epochs_completed': len(fold_train_metrics),
            'gpu_memory_peak': gpu_peak,
            'test_pred': best_val_metrics['predictions'],
            'test_true': best_val_metrics['true_labels'],
            'test_probabilities': best_val_metrics['probabilities'],
            'train_history': fold_train_metrics,
            'val_history': fold_val_metrics
        }

        all_fold_results.append(fold_result)
        detailed_results['fold_results'].append(fold_result)

        print(f"\n Fold {fold+1} Summary:")
        print(f"   Epochs Completed: {len(fold_train_metrics)}")
        print(f"   Best Validation Loss: {best_val_metrics['loss']:.4f}")
        print(f"   Best Validation F1-Weighted: {best_val_metrics['f1_weighted']:.4f}")
        print(f"   Best Validation Accuracy: {best_val_metrics['accuracy']:.4f}")
        print(f"   Best Validation AUC-Weighted: {best_val_metrics.get('auc_weighted', 0):.4f}")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Total Fold Time: {total_fold_time:.2f}s")
        print(f"   GPU Memory Peak: {gpu_peak:.1f}GB")

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    # Aggregate results
    metrics_to_aggregate = ['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted',
                           'precision_macro', 'precision_micro', 'precision_weighted',
                           'recall_macro', 'recall_micro', 'recall_weighted', 'auc_weighted']

    aggregated_metrics = {}
    for metric in metrics_to_aggregate:
        values = []
        for fold in all_fold_results:
            if metric in fold['best_metrics'] and fold['best_metrics'][metric] is not None:
                values.append(fold['best_metrics'][metric])

        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            ci_95 = [mean_val - 1.96 * std_val, mean_val + 1.96 * std_val]
            aggregated_metrics[metric] = {
                'mean': mean_val,
                'std': std_val,
                'ci_95': ci_95,
                'values': values
            }

    detailed_results['aggregated_metrics'] = aggregated_metrics
    timing_stats = timer.get_stats()
    detailed_results['timing_stats'] = timing_stats
    total_time = time.time() - model_start_time

    return {
        'model_name': 'RoBERTa-base',
        'status': 'SUCCESS',
        'total_time': total_time,
        'total_time_minutes': total_time/60,
        'detailed_results': detailed_results,
        'all_fold_results': all_fold_results,
        'aggregated_metrics': aggregated_metrics,
        'timing_stats': timing_stats
    }

def main():
    # Set all seeds
    set_seed(42)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA deterministic: {torch.backends.cudnn.deterministic}")
        print(f"CUDA benchmark: {torch.backends.cudnn.benchmark}")

    # Load and preprocess data
    print(f"\n Loading dataset...")
    try:
        df = pd.read_csv('cyberbullying_tweets.csv')
        print(f" Dataset loaded successfully: {len(df)} samples")

        df['tweet_text'] = df['tweet_text'].apply(clean_text)

        target_classes = ['age', 'ethnicity', 'gender', 'religion', 'other_cyberbullying']
        df_filtered = df[df['cyberbullying_type'].isin(target_classes)].copy()
        df_filtered = df_filtered.dropna(subset=['tweet_text', 'cyberbullying_type'])
        df_filtered = df_filtered[df_filtered['tweet_text'].str.len() > 0]

        print(f" After filtering and cleaning: {len(df_filtered)} samples")
        print("\n Class distribution:")
        class_counts = df_filtered['cyberbullying_type'].value_counts().sort_index()
        for class_name, count in class_counts.items():
            percentage = (count / len(df_filtered)) * 100
            print(f"   {class_name}: {count} samples ({percentage:.1f}%)")

        X = df_filtered['tweet_text']
        y = df_filtered['cyberbullying_type']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        print(f"\n Label mapping:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"   {class_name}: {i}")

    except FileNotFoundError:
        print("Dataset file 'cyberbullying_tweets.csv' not found!")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Run RoBERTa evaluation
    print(f"\n Starting RoBERTa evaluation on {len(df_filtered)} samples...")
    start_time = time.time()

    result = run_roberta_evaluation_with_comprehensive_output(X, y_encoded, label_encoder, device)

    if result['status'] == 'SUCCESS':
        # Print comprehensive results
        detailed_results = result['detailed_results']
        aggregated_metrics = result['aggregated_metrics']
        timing_stats = result['timing_stats']
        all_fold_results = result['all_fold_results']

        print(f'\n{"="*100}')
        print(f'ROBERTA COMPREHENSIVE RESULTS')
        print(f'{"="*100}')

        print(f'\n️  Timing Statistics:')
        print(f'Average Fold Time: {timing_stats["avg_fold_time"]:.2f} ± {timing_stats["std_fold_time"]:.2f} seconds')
        print(f'Average Epoch Time: {timing_stats["avg_epoch_time"]:.2f} ± {timing_stats["std_epoch_time"]:.2f} seconds')
        print(f'Total Epochs Trained: {timing_stats["total_epochs"]}')

        print(f'\nPerformance Metrics (Mean ± Std) [95% CI]:')
        for metric, stats in aggregated_metrics.items():
            if stats:
                print(f'   {metric:20s}: {stats["mean"]:.4f} ± {stats["std"]:.4f} '
                      f'[{stats["ci_95"][0]:.4f}, {stats["ci_95"][1]:.4f}]')

        print(f'\nPer-Fold Detailed Results:')
        print(f'{"Fold":<6} {"F1-Weighted":<12} {"F1-Macro":<10} {"F1-Micro":<10} {"Accuracy":<10} '
              f'{"AUC-Weighted":<12} {"Time(s)":<10} {"Epochs":<8} {"GPU(GB)":<8}')
        print('-' * 100)
        for fold in all_fold_results:
            metrics = fold['best_metrics']
            print(f'{fold["fold"]:<6} '
                  f'{metrics["f1_weighted"]:<12.4f} '
                  f'{metrics["f1_macro"]:<10.4f} '
                  f'{metrics["f1_micro"]:<10.4f} '
                  f'{metrics["accuracy"]:<10.4f} '
                  f'{metrics.get("auc_weighted", 0):<12.4f} '
                  f'{fold["training_time"]:<10.1f} '
                  f'{fold["epochs_completed"]:<8} '
                  f'{fold["gpu_memory_peak"]:<8.1f}')

        # Generate classification report
        all_true = []
        all_pred = []
        for fold in all_fold_results:
            all_true.extend(fold['test_true'])
            all_pred.extend(fold['test_pred'])

        print(f'\nAggregated Classification Report (All Folds Combined):')
        print(classification_report(all_true, all_pred, target_names=label_encoder.classes_, digits=4))

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f'roberta_{timestamp}.json'

        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        import copy
        json_results = copy.deepcopy(detailed_results)

        with open(results_filename, 'w') as f:
            json.dump(json_results, f, default=convert_for_json, indent=2)

        print(f'\nComprehensive results saved to: {results_filename}')

        print(f"\nROBERTA FINAL PERFORMANCE:")
        if 'f1_weighted' in aggregated_metrics:
            print(f"   F1-Score (Weighted): {aggregated_metrics['f1_weighted']['mean']:.4f} ± {aggregated_metrics['f1_weighted']['std']:.4f}")
        if 'accuracy' in aggregated_metrics:
            print(f"   Accuracy: {aggregated_metrics['accuracy']['mean']:.4f} ± {aggregated_metrics['accuracy']['std']:.4f}")
        if 'auc_weighted' in aggregated_metrics:
            print(f"   AUC (Weighted): {aggregated_metrics['auc_weighted']['mean']:.4f} ± {aggregated_metrics['auc_weighted']['std']:.4f}")

        # Architecture summary
        if 'model_parameters' in detailed_results:
            param_info = detailed_results['model_parameters']
            print(f"\n Architecture Details:")
            print(f"   Total Parameters: {param_info['total']:,} ({param_info['total_mb']:.1f} MB)")
            print(f"   Classifier Parameters: {param_info['breakdown']['classifier']:,}")

    else:
        print(f"RoBERTa evaluation failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()