import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from transformers import RobertaModel, RobertaTokenizer, set_seed as transformers_set_seed
from torch_geometric.nn import GCNConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np
import time
import psutil
from tqdm import tqdm
import warnings
import json
from datetime import datetime
import gc

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

class DetailedTimer:
    """timer for detailed performance monitoring"""
    def __init__(self):
        self.epoch_times = []
        self.fold_times = []
        self.total_start = None
        self.fold_start = None
        self.epoch_start = None

    def start_total(self):
        self.total_start = time.time()

    def start_fold(self):
        self.fold_start = time.time()

    def start_epoch(self):
        self.epoch_start = time.time()

    def end_epoch(self):
        if self.epoch_start:
            epoch_time = time.time() - self.epoch_start
            self.epoch_times.append(epoch_time)
            return epoch_time
        return 0

    def end_fold(self):
        if self.fold_start:
            fold_time = time.time() - self.fold_start
            self.fold_times.append(fold_time)
            return fold_time
        return 0

    def get_total_time(self):
        return time.time() - self.total_start if self.total_start else 0

    def get_statistics(self):
        return {
            'total_time': self.get_total_time(),
            'fold_times': self.fold_times,
            'avg_fold_time': np.mean(self.fold_times) if self.fold_times else 0,
            'std_fold_time': np.std(self.fold_times) if self.fold_times else 0,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'std_epoch_time': np.std(self.epoch_times) if self.epoch_times else 0,
            'total_epochs': len(self.epoch_times),
            'total_folds': len(self.fold_times)
        }

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
    if hasattr(model, 'gnn'):
        param_breakdown['gnn'] = sum(p.numel() for p in model.gnn.parameters())
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

class EarlyStopping:
    """Early stopping with validation loss monitoring"""
    def __init__(self, patience=3, min_delta=0, verbose=True, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        if self.mode == 'min':
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class CustomDataset(Dataset):
    """Dataset class"""
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

class EnhancedGNNModel(nn.Module):
    """Enhanced GNN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnhancedGNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.layer_norm1(x)
        x = self.dropout(x)

        x = self.relu(self.conv2(x, edge_index))
        x = self.layer_norm2(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        return x

class RobertaOTA(nn.Module):
    """
    ROBERTA-OTA ARCHITECTURE

    RoBERTa-Optimized: RoBERTa + Enhanced GNN + Deep Classifier
    """
    def __init__(self, gnn_input_dim, gnn_hidden_dim, gnn_output_dim, num_classes=5):
        super(RobertaOTA, self).__init__()
        self.transformer_name = 'roberta-base'

        # ROBERTA
        self.transformer = RobertaModel.from_pretrained('roberta-base')

        # ENHANCED GNN MODEL
        self.gnn = EnhancedGNNModel(gnn_input_dim, gnn_hidden_dim, gnn_output_dim)

        # DEEP CLASSIFIER
        combined_dim = self.transformer.config.hidden_size + gnn_output_dim  # 768 + 32 = 800
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(0.3),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(combined_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(combined_dim // 2, combined_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(combined_dim // 4),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 4, num_classes)
        )

    def forward(self, input_ids, attention_mask, x, edge_index):
        transformer_out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(transformer_out, 'pooler_output') and transformer_out.pooler_output is not None:
            attention_out = transformer_out.pooler_output
        else:
            attention_out = transformer_out.last_hidden_state.mean(dim=1)  # Fallback to mean pooling

        gnn_features = self.gnn(x, edge_index)
        gnn_features = torch.mean(gnn_features, dim=0).unsqueeze(0).repeat(attention_out.size(0), 1)

        combined_features = torch.cat([attention_out, gnn_features], dim=1)
        output = self.classifier(combined_features)
        return output

def create_advanced_5class_cyberbullying_ontology_graph():
    """Ontology graph"""
    nodes = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.7, 0.3],  # age - demographic + moderate complexity + low diversity
        [1.0, 1.0, 0.0, 0.0, 0.6, 0.7],  # ethnicity - demographic + cultural + moderate complexity + high diversity
        [1.0, 0.0, 1.0, 0.0, 0.6, 0.5],  # gender - demographic + gender + moderate complexity + moderate diversity
        [1.0, 1.0, 0.0, 1.0, 0.9, 0.8],  # religion - demographic + cultural + religious + HIGH complexity + high diversity
        [0.5, 0.5, 0.5, 0.5, 0.4, 0.9],  # other - mixed characteristics + low complexity + highest diversity
    ], dtype=torch.float)

    edge_index = torch.tensor([
        [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4,
         0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 0, 2, 0, 3, 0, 4, 0, 2, 1, 3, 1, 4, 1, 3, 2, 4, 2, 4, 3,
         2, 3, 3, 0, 4, 1, 4, 2, 0, 1],
    ], dtype=torch.long)

    return nodes, edge_index

def calculate_comprehensive_metrics(true_labels, predictions, probabilities=None):
    """Metrics calculation"""
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

def train_epoch(model, data_loader, loss_fn, optimizer, device, nodes, edge_index, epoch, timer):
    """Training function"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    timer.start_epoch()
    progress_bar = tqdm(data_loader, desc=f'Training Epoch {epoch}', leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask, nodes.to(device), edge_index.to(device))
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().detach().numpy())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_time = timer.end_epoch()
    metrics = calculate_comprehensive_metrics(all_labels, all_predictions, np.array(all_probabilities))
    metrics['loss'] = total_loss / len(data_loader)
    metrics['epoch_time'] = epoch_time

    return metrics

def evaluate_with_probabilities(model, data_loader, loss_fn, device, nodes, edge_index):
    """Evaluation function"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, nodes.to(device), edge_index.to(device))
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().detach().numpy())

    metrics = calculate_comprehensive_metrics(all_labels, all_predictions, np.array(all_probabilities))
    metrics['loss'] = total_loss / len(data_loader)

    return metrics, all_labels, all_predictions, np.array(all_probabilities)

def filter_5_classes(df):
    """Filter dataset to only include the 5 cyberbullying classes"""
    text_column = 'tweet_text' if 'tweet_text' in df.columns else 'text'
    label_column = 'cyberbullying_type' if 'cyberbullying_type' in df.columns else 'class'

    target_classes = ['age', 'ethnicity', 'gender', 'religion', 'other_cyberbullying']
    df_filtered = df[df[label_column].isin(target_classes)].copy()

    print(f"Original dataset size: {len(df)}")
    print(f"Filtered dataset size (5 classes): {len(df_filtered)}")
    print(f"\nClass distribution after filtering:")
    print(df_filtered[label_column].value_counts())

    return df_filtered

def run_roberta_ota_evaluation(df, num_epochs=20, n_splits=5):
    """Run comprehensive RoBERTa-OTA evaluation"""
    # Clean environment and set seed
    torch.cuda.empty_cache()
    gc.collect()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*100}")
    print(f"EVALUATING: RoBERTa-OTA Specific Architecture")
    print(f"Using device: {device}")
    print(f"RoBERTa-OTA 5-Class Cyberbullying Detection")
    print(f"{'='*100}")

    model_start_time = time.time()

    try:
        # Initialize timer
        timer = DetailedTimer()
        timer.start_total()

        # Prepare data
        text_column = 'tweet_text' if 'tweet_text' in df.columns else 'text'
        label_column = 'cyberbullying_type' if 'cyberbullying_type' in df.columns else 'class'

        texts = df[text_column].values
        le = LabelEncoder()
        labels = le.fit_transform(df[label_column])
        class_names = le.classes_

        # Initialize components
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        nodes, edge_index = create_advanced_5class_cyberbullying_ontology_graph()
        max_len = 128

        dataset = CustomDataset(texts, labels, tokenizer, max_len)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        all_fold_results = []

        print(f"\nStarting {n_splits}-Fold Cross-Validation with RoBERTa-OTA...")

        # Store comprehensive results
        detailed_results = {
            'model_name': 'RoBERTa-OTA Specific',
            'dataset_info': {
                'total_samples': len(texts),
                'num_classes': len(class_names),
                'class_names': list(class_names),
                'class_distribution': {name: int(np.sum(labels == i)) for i, name in enumerate(class_names)}
            },
            'training_config': {
                'max_length': 128,
                'batch_size': 16,
                'num_epochs': num_epochs,
                'n_splits': n_splits,
                'learning_rate': 1e-5,
                'optimizer': 'AdamW',
                'weight_decay': 1e-5,
                'architecture': 'RoBERTa-OTA: RoBERTa + Enhanced GNN + Deep Classifier'
            },
            'fold_results': []
        }

        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
            print(f'\n FOLD {fold}/{n_splits}')
            timer.start_fold()

            # Create data loaders
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler,
                                    num_workers=0, pin_memory=False)
            val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler,
                                  num_workers=0, pin_memory=False)

            print(f"Train samples: {len(train_idx)}")
            print(f"Validation samples: {len(val_idx)}")

            # Initialize model with RoBERTa-OTA architecture
            model = RobertaOTA(
                gnn_input_dim=6,
                gnn_hidden_dim=64,
                gnn_output_dim=32,
                num_classes=len(class_names)
            ).to(device)

            # Model parameters info (only for first fold)
            if fold == 1:
                param_info = count_parameters(model)
                detailed_results['model_parameters'] = param_info
                print(f"\n RoBERTa-OTA Architecture:")
                print(f"   Total parameters: {param_info['total']:,}")
                print(f"   Trainable parameters: {param_info['trainable']:,}")
                print(f"   Model size: {param_info['total_mb']:.1f} MB")
                print(f"   Component breakdown:")
                for component, params in param_info['breakdown'].items():
                    print(f"     {component}: {params:,} ({params/param_info['total']*100:.1f}%)")

            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5, eps=1e-8)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            early_stopping = EarlyStopping(patience=3, verbose=False, mode='min')

            best_val_loss = float('inf')
            best_epoch = 0
            best_metrics = None

            print(f"\n Training RoBERTa-OTA started...")

            # Training loop for this fold
            for epoch in range(num_epochs):
                # Training
                train_metrics = train_epoch(model, train_loader, loss_fn, optimizer,
                                          device, nodes, edge_index, epoch + 1, timer)

                # Validation
                val_metrics, val_true, val_pred, val_probs = evaluate_with_probabilities(
                    model, val_loader, loss_fn, device, nodes, edge_index)

                # Learning rate scheduling
                scheduler.step(val_metrics['loss'])
                current_lr = optimizer.param_groups[0]['lr']

                # Check if this is the best model so far
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_epoch = epoch + 1
                    best_metrics = val_metrics.copy()

                # Get GPU memory info
                gpu_info = get_gpu_memory()

                print(f'Epoch {epoch + 1:2d}/{num_epochs} | '
                      f'Train Loss: {train_metrics["loss"]:.4f} | Val Loss: {val_metrics["loss"]:.4f} | '
                      f'Train Acc: {train_metrics["accuracy"]:.4f} | Val Acc: {val_metrics["accuracy"]:.4f} | '
                      f'Val F1-W: {val_metrics["f1_weighted"]:.4f} | '
                      f'LR: {current_lr:.2e} | '
                      f'GPU: {gpu_info["allocated_gb"]:.1f}GB | '
                      f'Time: {train_metrics["epoch_time"]:.1f}s')

                early_stopping(val_metrics['loss'])
                if early_stopping.early_stop:
                    print(f' Early stopping triggered at epoch {epoch + 1}')
                    break

            fold_time = timer.end_fold()
            gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

            # Store results for this fold
            fold_result = {
                'fold': fold,
                'training_time': fold_time,
                'epochs_completed': epoch + 1,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'best_metrics': best_metrics,
                'gpu_memory_peak': gpu_peak,
                'test_pred': val_pred,
                'test_true': val_true,
                'test_probabilities': val_probs.tolist()
            }

            all_fold_results.append(fold_result)
            detailed_results['fold_results'].append(fold_result)

            print(f'\nFold {fold} Summary:')
            print(f'   Best Epoch: {best_epoch}')
            print(f'   Best Validation Loss: {best_val_loss:.4f}')
            print(f'   Best Validation F1-Weighted: {best_metrics["f1_weighted"]:.4f}')
            print(f'   Best Validation Accuracy: {best_metrics["accuracy"]:.4f}')
            print(f'   Training Time: {fold_time:.2f}s')
            print(f'   GPU Memory Peak: {gpu_peak:.1f}GB')

            # Clear GPU memory
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            gc.collect()

        # Calculate aggregated statistics
        timing_stats = timer.get_statistics()

        # Aggregate metrics across all folds
        metrics_to_aggregate = ['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted',
                               'precision_macro', 'precision_micro', 'precision_weighted',
                               'recall_macro', 'recall_micro', 'recall_weighted',
                               'auc_macro', 'auc_weighted']

        aggregated_metrics = {}
        for metric in metrics_to_aggregate:
            values = []
            for fold in all_fold_results:
                val = fold['best_metrics'].get(metric)
                if val is not None:
                    values.append(val)

            if values:
                aggregated_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values,
                }

        detailed_results['aggregated_metrics'] = aggregated_metrics
        detailed_results['timing_stats'] = timing_stats

        total_time = time.time() - model_start_time

        print(f'\nRoBERTa-OTA Specific FINAL RESULTS:')
        print(f'   F1-Weighted: {aggregated_metrics["f1_weighted"]["mean"]:.4f} ± {aggregated_metrics["f1_weighted"]["std"]:.4f}')
        print(f'   Accuracy: {aggregated_metrics["accuracy"]["mean"]:.4f} ± {aggregated_metrics["accuracy"]["std"]:.4f}')
        print(f'   F1-Macro: {aggregated_metrics["f1_macro"]["mean"]:.4f} ± {aggregated_metrics["f1_macro"]["std"]:.4f}')
        print(f'   Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)')

        return {
            'model_name': 'roberta-ota',
            'status': 'SUCCESS',
            'total_time': total_time,
            'total_time_minutes': total_time/60,
            'detailed_results': detailed_results,
            'all_fold_results': all_fold_results,
            'aggregated_metrics': aggregated_metrics,
            'timing_stats': timing_stats,
            'f1_weighted_mean': aggregated_metrics["f1_weighted"]["mean"],
            'f1_weighted_std': aggregated_metrics["f1_weighted"]["std"],
            'accuracy_mean': aggregated_metrics["accuracy"]["mean"],
            'accuracy_std': aggregated_metrics["accuracy"]["std"],
            'f1_macro_mean': aggregated_metrics["f1_macro"]["mean"],
            'f1_macro_std': aggregated_metrics["f1_macro"]["std"],
            'precision_weighted_mean': aggregated_metrics["precision_weighted"]["mean"],
            'precision_weighted_std': aggregated_metrics["precision_weighted"]["std"],
            'recall_weighted_mean': aggregated_metrics["recall_weighted"]["mean"],
            'recall_weighted_std': aggregated_metrics["recall_weighted"]["std"],
            'avg_fold_time': timing_stats["avg_fold_time"],
            'std_fold_time': timing_stats["std_fold_time"],
        }

    except Exception as e:
        print(f"ERROR with RoBERTa-OTA: {str(e)}")
        return {
            'model_name': 'roberta-ota',
            'status': 'FAILED',
            'error': str(e),
            'total_time': time.time() - model_start_time
        }

def main():
    set_seed(42)

    # Clean environment
    torch.cuda.empty_cache()
    gc.collect()

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Print CUDA deterministic settings
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load dataset
    print(f"\nLoading dataset...")
    try:
        df = pd.read_csv('cyberbullying_tweets.csv')
        print(f"Dataset loaded successfully: {len(df)} samples")

        # Filter to 5 classes and clean
        df_filtered = filter_5_classes(df)

    except FileNotFoundError:
        print("Dataset file 'cyberbullying_tweets.csv' not found!")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Run comprehensive RoBERTa-OTA evaluation
    print(f"\nStarting RoBERTa-OTA evaluation on {len(df_filtered)} samples...")
    start_time = time.time()

    result = run_roberta_ota_evaluation(df_filtered, num_epochs=20, n_splits=5)

    if result['status'] == 'SUCCESS':
        print(f"RoBERTa-OTA completed successfully")

        # Print comprehensive results
        detailed_results = result['detailed_results']
        aggregated_metrics = result['aggregated_metrics']
        timing_stats = result['timing_stats']
        all_fold_results = result['all_fold_results']

        print(f'\n{"="*100}')
        print(f'ROBERTA-OTA COMPREHENSIVE RESULTS')
        print(f'{"="*100}')

        print(f'\n Timing Statistics:')
        print(f'Average Fold Time: {timing_stats["avg_fold_time"]:.2f} ± {timing_stats["std_fold_time"]:.2f} seconds')
        print(f'Average Epoch Time: {timing_stats["avg_epoch_time"]:.2f} ± {timing_stats["std_epoch_time"]:.2f} seconds')
        print(f'Total Epochs Trained: {timing_stats["total_epochs"]}')

        print(f'\nPerformance Metrics (Mean ± Std):')
        for metric, stats in aggregated_metrics.items():
            if stats:
                print(f'   {metric:20s}: {stats["mean"]:.4f} ± {stats["std"]:.4f}')

        print(f'\nPer-Fold Detailed Results:')
        print(f'{"Fold":<6} {"F1-Weighted":<12} {"F1-Macro":<10} {"Accuracy":<10} '
              f'{"Time(s)":<10} {"Epochs":<8} {"GPU(GB)":<8}')
        print('-' * 80)
        for fold in all_fold_results:
            metrics = fold['best_metrics']
            print(f'{fold["fold"]:<6} '
                  f'{metrics["f1_weighted"]:<12.4f} '
                  f'{metrics["f1_macro"]:<10.4f} '
                  f'{metrics["accuracy"]:<10.4f} '
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
        label_names = detailed_results['dataset_info']['class_names']
        print(classification_report(all_true, all_pred, target_names=label_names, digits=4))

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f'roberta-ota_{timestamp}.json'

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

    else:
        print(f"RoBERTa-OTA failed")

    total_time = time.time() - start_time

    # Create results DataFrame and save
    results_df = pd.DataFrame([result])
    timestamp = int(time.time())
    results_file = f'roberta-ota_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)

    # Print final summary
    print(f"\n{'='*120}")
    print(f"ROBERTA-OTA SPECIFIC EVALUATION COMPLETE")
    print(f"{'='*120}")
    print(f"Total evaluation time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {results_file}")

    if result['status'] == 'SUCCESS':
        print(f"\nROBERTA-OTA FINAL PERFORMANCE:")
        print(f"   F1-Weighted: {result['f1_weighted_mean']:.4f} ± {result['f1_weighted_std']:.4f}")
        print(f"   Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
        print(f"   F1-Macro: {result['f1_macro_mean']:.4f} ± {result['f1_macro_std']:.4f}")
        print(f"   Total Time: {result['total_time']:.1f}s ({result['total_time_minutes']:.1f} minutes)")

        # Architecture summary
        if 'model_parameters' in detailed_results:
            param_info = detailed_results['model_parameters']
            print(f"\n RoBERTa-OTA Architecture Details:")
            print(f"   Total Parameters: {param_info['total']:,} ({param_info['total_mb']:.1f} MB)")
            print(f"   Component breakdown:")
            for component, params in param_info['breakdown'].items():
                print(f"     {component}: {params:,} ({params/param_info['total']*100:.1f}%)")


if __name__ == "__main__":
    main()