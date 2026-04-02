"""
Experiment: Measure how motion artifacts affect AD classification accuracy.

This script:
1. Trains a Vision Transformer on clean images (M0) to classify AD vs healthy
2. Evaluates the trained model on each motion severity level (M0-M4)
3. Reports 5 metrics: Accuracy, Precision, Recall, F1 Score, AUROC
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from DeepLearning.ViTModel import VisionTransformer
from Utils.DataUtils.ADDataLoader import (
    load_cdr_labels,
    get_subject_splits,
    create_dataloaders_for_motion_level,
    create_test_loader_for_motion_level,
)


def train_model(model, train_loader, val_loader, epochs=10, device='cuda'):
    """Train the ViT model for AD classification."""
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_val_acc


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model and return 5 metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - AUROC
    """
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    if len(all_labels) == 0:
        return {
            'accuracy': 0, 'precision': 0, 'recall': 0,
            'f1_score': 0, 'auroc': 0, 'total_samples': 0
        }

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    try:
        if len(np.unique(all_labels)) > 1:
            auroc = roc_auc_score(all_labels, all_probabilities)
        else:
            auroc = 0.0
    except ValueError:
        auroc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auroc': auroc,
        'total_samples': len(all_labels)
    }


def run_experiment(epochs=10, batch_size=16):
    """Run the full experiment measuring AD accuracy across motion levels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cdr_labels = load_cdr_labels()
    train_subjects, val_subjects, test_subjects = get_subject_splits(cdr_labels)

    print(f"\nSubject splits:")
    print(f"  Train: {len(train_subjects)} subjects")
    print(f"  Val: {len(val_subjects)} subjects")
    print(f"  Test: {len(test_subjects)} subjects")

    print("\n" + "="*70)
    print("PHASE 1: Training on clean images (M0 - no motion)")
    print("="*70)

    train_loader, val_loader, _ = create_dataloaders_for_motion_level(0, batch_size)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = VisionTransformer(num_classes=2).to(device)
    model, best_val_acc = train_model(model, train_loader, val_loader, epochs, device)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    torch.save(model.state_dict(), "ad_classifier_trained_on_clean.pth")
    print("Model saved to ad_classifier_trained_on_clean.pth")

    print("\n" + "="*70)
    print("PHASE 2: Evaluating across motion severity levels")
    print("="*70)

    results = {}
    motion_levels = [0, 1, 2, 3, 4]
    motion_labels = ["M0 (No Motion)", "M1 (Small)", "M2 (Mild)", "M3 (Moderate)", "M4 (Severe)"]

    for level in motion_levels:
        test_loader = create_test_loader_for_motion_level(level, test_subjects, cdr_labels, batch_size)
        metrics = evaluate_model(model, test_loader, device)
        results[level] = metrics

        print(f"\n{motion_labels[level]}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  AUROC:     {metrics['auroc']:.4f}")
        print(f"  Samples:   {metrics['total_samples']}")

    save_results_to_csv(results, motion_labels)
    save_results_to_txt(results, motion_labels)
    print_summary_table(results, motion_labels)
    plot_results(results, motion_labels)

    return results


def save_results_to_csv(results, motion_labels):
    """Save results to CSV file."""
    data = []
    for level, label in enumerate(motion_labels):
        metrics = results[level]
        data.append({
            'Motion Level': label,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'AUROC': metrics['auroc'],
            'Samples': metrics['total_samples']
        })

    df = pd.DataFrame(data)
    df.to_csv('ad_motion_metrics_results.csv', index=False)
    print("\nResults saved to ad_motion_metrics_results.csv")


def save_results_to_txt(results, motion_labels):
    """Save results to a readable text file."""
    with open('ad_motion_metrics_results.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("AD CLASSIFICATION PERFORMANCE ACROSS MOTION SEVERITY LEVELS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("METRICS TABLE\n")
        f.write("-" * 70 + "\n")
        header = f"{'Motion Level':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'AUROC':>10}\n"
        f.write(header)
        f.write("-" * 70 + "\n")
        
        for level, label in enumerate(motion_labels):
            m = results[level]
            row = f"{label:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['auroc']:>10.4f}\n"
            f.write(row)
        
        f.write("-" * 70 + "\n\n")
        
        # Degradation analysis
        baseline = results[0]
        f.write("DEGRADATION FROM BASELINE (M0 - No Motion)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Motion Level':<20} {'Acc Δ':>10} {'Prec Δ':>10} {'Recall Δ':>10} {'F1 Δ':>10} {'AUROC Δ':>10}\n")
        f.write("-" * 70 + "\n")
        
        for level in range(5):
            m = results[level]
            label = motion_labels[level]
            acc_d = m['accuracy'] - baseline['accuracy']
            prec_d = m['precision'] - baseline['precision']
            rec_d = m['recall'] - baseline['recall']
            f1_d = m['f1_score'] - baseline['f1_score']
            auc_d = m['auroc'] - baseline['auroc']
            row = f"{label:<20} {acc_d:>+10.4f} {prec_d:>+10.4f} {rec_d:>+10.4f} {f1_d:>+10.4f} {auc_d:>+10.4f}\n"
            f.write(row)
        
        f.write("-" * 70 + "\n\n")
        
        # Individual level details
        f.write("DETAILED RESULTS PER MOTION LEVEL\n")
        f.write("=" * 70 + "\n")
        for level, label in enumerate(motion_labels):
            m = results[level]
            f.write(f"\n{label}\n")
            f.write(f"  Accuracy:    {m['accuracy']:.4f}\n")
            f.write(f"  Precision:   {m['precision']:.4f}\n")
            f.write(f"  Recall:      {m['recall']:.4f}\n")
            f.write(f"  F1 Score:    {m['f1_score']:.4f}\n")
            f.write(f"  AUROC:       {m['auroc']:.4f}\n")
            f.write(f"  Samples:     {m['total_samples']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print("Results saved to ad_motion_metrics_results.txt")


def print_summary_table(results, motion_labels):
    """Print a formatted summary table."""
    print("\n" + "="*70)
    print("SUMMARY: All Metrics Across Motion Severity Levels")
    print("="*70)

    header = f"{'Motion Level':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'AUROC':>10}"
    print(header)
    print("-" * 70)

    for level, label in enumerate(motion_labels):
        m = results[level]
        row = f"{label:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['auroc']:>10.4f}"
        print(row)

    print("-" * 70)

    baseline = results[0]
    print(f"\nDegradation from M0 (No Motion) baseline:")
    for level in range(1, 5):
        m = results[level]
        acc_deg = baseline['accuracy'] - m['accuracy']
        f1_deg = baseline['f1_score'] - m['f1_score']
        auroc_deg = baseline['auroc'] - m['auroc']
        print(f"  {motion_labels[level]}: Acc {acc_deg:+.4f}, F1 {f1_deg:+.4f}, AUROC {auroc_deg:+.4f}")


def plot_results(results, motion_labels):
    """Plot all 5 metrics across motion levels."""
    levels = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auroc']
    display_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC']
    colors = ['steelblue', 'forestgreen', 'coral', 'mediumpurple', 'gold']

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    for idx, (metric, display_name, color) in enumerate(zip(metrics_names, display_names, colors)):
        values = [results[l][metric] for l in levels]
        axes[idx].bar(range(len(motion_labels)), values, color=color)
        axes[idx].set_xticks(range(len(motion_labels)))
        axes[idx].set_xticklabels(['M0', 'M1', 'M2', 'M3', 'M4'])
        axes[idx].set_ylabel(display_name)
        axes[idx].set_title(f'{display_name} vs Motion')
        axes[idx].set_ylim([0, 1])
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('ad_motion_all_metrics.png', dpi=150)
    plt.show()
    print("Metrics plot saved to ad_motion_all_metrics.png")

    fig2, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(motion_labels))
    width = 0.15

    for idx, (metric, display_name, color) in enumerate(zip(metrics_names, display_names, colors)):
        values = [results[l][metric] for l in levels]
        ax.bar(x + idx * width, values, width, label=display_name, color=color)

    ax.set_xlabel('Motion Severity Level')
    ax.set_ylabel('Score')
    ax.set_title('AD Classification Performance vs Motion Artifact Severity')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(motion_labels, rotation=15, ha='right')
    ax.legend(loc='lower left')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig('ad_motion_combined_metrics.png', dpi=150)
    plt.show()
    print("Combined plot saved to ad_motion_combined_metrics.png")


if __name__ == "__main__":
    results = run_experiment(epochs=30, batch_size=16)
