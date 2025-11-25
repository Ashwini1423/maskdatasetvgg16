"""
Visualization and Plotting Module for Thermal Hotspot Detection
Generates comprehensive analysis plots and results visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path


class ThermalVisualization:
    """Visualization utilities for thermal hotspot detection system."""

    def __init__(self, style='seaborn-v0_8-darkgrid'):
        """
        Initialize visualization engine.

        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        sns.set_palette("husl")
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    def plot_feature_extraction_pipeline(self, original_image, preprocessed_image,
                                        lbp_map, loop_map, ldp_map, save_path=None):
        """
        Plot the feature extraction pipeline visualization.

        Args:
            original_image: Original thermal image
            preprocessed_image: After CLAHE preprocessing
            lbp_map: LBP feature map
            loop_map: LOOP feature map
            ldp_map: LDP feature map
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image, cmap='hot')
        ax1.set_title('Original Thermal Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Preprocessing stages
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(preprocessed_image, cmap='hot')
        ax2.set_title('CLAHE Enhanced Image', fontsize=12, fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        enhanced_filtered = cv2.GaussianBlur((preprocessed_image).astype(np.uint8), (5, 5), 0)
        ax3.imshow(enhanced_filtered, cmap='hot')
        ax3.set_title('Filtered Image', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # Feature maps
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(lbp_map, cmap='viridis')
        ax4.set_title('LBP Feature Map', fontsize=12, fontweight='bold')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.imshow(loop_map, cmap='plasma')
        ax5.set_title('LOOP Feature Map', fontsize=12, fontweight='bold')
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.imshow(ldp_map, cmap='magma')
        ax6.set_title('LDP Feature Map', fontsize=12, fontweight='bold')
        ax6.axis('off')

        # Histograms
        ax7 = fig.add_subplot(gs[2, 0])
        hist_original = cv2.calcHist([original_image.astype(np.uint8)], [0], None, [256], [0, 256])
        ax7.plot(hist_original, color=self.colors[0], linewidth=2)
        ax7.set_title('Original Image Histogram', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Pixel Value')
        ax7.set_ylabel('Frequency')
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(gs[2, 1])
        hist_enhanced = cv2.calcHist([preprocessed_image.astype(np.uint8)], [0], None, [256], [0, 256])
        ax8.plot(hist_enhanced, color=self.colors[1], linewidth=2)
        ax8.set_title('CLAHE Enhanced Histogram', fontsize=11, fontweight='bold')
        ax8.set_xlabel('Pixel Value')
        ax8.set_ylabel('Frequency')
        ax8.grid(True, alpha=0.3)

        ax9 = fig.add_subplot(gs[2, 2])
        ax9.text(0.5, 0.5, 'Feature Extraction\nPipeline Complete',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax9.axis('off')

        fig.suptitle('Thermal Image Feature Extraction Pipeline', fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[+] Feature extraction pipeline plot saved to {save_path}")

        plt.show()

    def plot_feature_histograms(self, lbp_hist, loop_hist, ldp_hist, save_path=None):
        """
        Plot feature histograms for all extractors.

        Args:
            lbp_hist: LBP histogram
            loop_hist: LOOP histogram
            ldp_hist: LDP histogram
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # LBP Histogram
        axes[0].bar(range(len(lbp_hist)), lbp_hist, color=self.colors[0], alpha=0.7, edgecolor='black')
        axes[0].set_title('LBP Histogram', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Bin')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3, axis='y')

        # LOOP Histogram
        axes[1].bar(range(len(loop_hist)), loop_hist, color=self.colors[1], alpha=0.7, edgecolor='black')
        axes[1].set_title('LOOP Histogram', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Bin')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3, axis='y')

        # LDP Histogram
        axes[2].bar(range(len(ldp_hist)), ldp_hist, color=self.colors[2], alpha=0.7, edgecolor='black')
        axes[2].set_title('LDP Histogram', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Bin')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3, axis='y')

        fig.suptitle('Feature Histograms - Texture Feature Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[+] Feature histograms plot saved to {save_path}")

        plt.show()

    def plot_egfs_results(self, all_scores, selected_features, n_display=30, save_path=None):
        """
        Plot EGFS feature selection results.

        Args:
            all_scores: All feature scores
            selected_features: Indices of selected features
            n_display: Number of top features to display
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Top features by score
        ax1 = fig.add_subplot(gs[0, :])
        top_indices = np.argsort(all_scores)[::-1][:n_display]
        top_scores = all_scores[top_indices]

        colors_bar = [self.colors[0] if idx in selected_features else '#CCCCCC'
                     for idx in top_indices]
        bars = ax1.barh(range(len(top_scores)), top_scores, color=colors_bar, edgecolor='black')
        ax1.set_yticks(range(len(top_scores)))
        ax1.set_yticklabels([f'Feature {i}' for i in top_indices], fontsize=9)
        ax1.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax1.set_title(f'Top {n_display} Features by EGFS Score (Red = Selected)',
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()

        # Feature selection distribution
        ax2 = fig.add_subplot(gs[1, 0])
        selected_count = len(selected_features)
        total_count = len(all_scores)
        sizes = [selected_count, total_count - selected_count]
        labels = [f'Selected\n({selected_count})', f'Not Selected\n({total_count - selected_count})']
        colors_pie = [self.colors[0], '#CCCCCC']
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax2.set_title('Feature Selection Distribution', fontsize=11, fontweight='bold')

        # Feature score distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(all_scores, bins=30, color=self.colors[1], alpha=0.7, edgecolor='black')
        ax3.axvline(np.median(all_scores), color='red', linestyle='--', linewidth=2, label='Median')
        ax3.axvline(np.mean(all_scores), color='green', linestyle='--', linewidth=2, label='Mean')
        ax3.set_xlabel('Feature Score', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax3.set_title('Feature Score Distribution', fontsize=11, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Enhanced Graph-Based Feature Selection (EGFS) Results',
                    fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[+] EGFS results plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['No Hotspot', 'Hotspot'],
                   yticklabels=['No Hotspot', 'Hotspot'],
                   annot_kws={'size': 14, 'fontweight': 'bold'},
                   ax=ax, linewidths=2, linecolor='black')

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix - Hotspot Detection', fontsize=13, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[+] Confusion matrix plot saved to {save_path}")

        plt.show()

    def plot_classification_metrics(self, metrics, save_path=None):
        """
        Plot classification metrics.

        Args:
            metrics: Dictionary with evaluation metrics
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Metrics bar chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['precision'],
                        metrics['recall'], metrics['f1_score']]

        bars = axes[0].bar(metric_names, metric_values, color=self.colors[:4],
                          alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Classification Metrics', fontsize=12, fontweight='bold')
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # Confusion matrix display
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']

        metrics_text = f"""
        True Negatives (TN):  {tn}
        False Positives (FP): {fp}
        False Negatives (FN): {fn}
        True Positives (TP):  {tp}

        Sensitivity (Recall): {metrics['recall']:.4f}
        Specificity:          {tn/(tn+fp):.4f}
        """

        axes[1].text(0.5, 0.5, metrics_text, ha='center', va='center',
                    fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1].axis('off')
        axes[1].set_title('Detailed Metrics', fontsize=12, fontweight='bold')

        fig.suptitle('SVM Classifier Performance Evaluation', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[+] Classification metrics plot saved to {save_path}")

        plt.show()

    def plot_training_summary(self, training_history, save_path=None):
        """
        Plot training summary information.

        Args:
            training_history: Training history dictionary
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Processing times
        ax1 = fig.add_subplot(gs[0, 0])
        times = [training_history['extraction_time'],
                training_history['selection_time'],
                training_history['train_time']]
        stages = ['Feature\nExtraction', 'EGFS\nSelection', 'SVM\nTraining']
        colors_bar = self.colors[:3]

        bars = ax1.bar(stages, times, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
        ax1.set_title('Processing Time per Stage', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}s', ha='center', va='bottom', fontweight='bold')

        # Feature reduction
        ax2 = fig.add_subplot(gs[0, 1])
        features_before = training_history['features_before']
        features_after = training_history['features_after']
        reduction = (1 - features_after/features_before) * 100

        x_pos = [0, 1]
        heights = [features_before, features_after]
        colors_feat = [self.colors[3], self.colors[0]]
        bars = ax2.bar(['Before EGFS', 'After EGFS'], heights, color=colors_feat,
                      alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Number of Features', fontsize=11, fontweight='bold')
        ax2.set_title(f'Feature Reduction ({reduction:.1f}%)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, height in zip(bars, heights):
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        # Summary text
        ax3 = fig.add_subplot(gs[1, :])
        summary_text = f"""
        THERMAL HOTSPOT DETECTION - TRAINING SUMMARY

        Total Processing Time:     {sum(times):.2f} seconds

        Feature Extraction Stage:  {training_history['extraction_time']:.2f}s ({training_history['extraction_time']/sum(times)*100:.1f}%)
        EGFS Selection Stage:      {training_history['selection_time']:.2f}s ({training_history['selection_time']/sum(times)*100:.1f}%)
        SVM Training Stage:        {training_history['train_time']:.2f}s ({training_history['train_time']/sum(times)*100:.1f}%)

        Feature Reduction:        {features_before} → {features_after} ({reduction:.1f}% reduction)
        """

        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax3.axis('off')

        fig.suptitle('Training Summary Report', fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[+] Training summary plot saved to {save_path}")

        plt.show()

    def plot_real_time_detection(self, image, prediction, confidence, save_path=None):
        """
        Plot real-time detection result.

        Args:
            image: Input image
            prediction: Prediction (0 or 1)
            confidence: Confidence score
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.imshow(image, cmap='hot')

        # Add detection result overlay
        result_text = "🔥 HOTSPOT DETECTED" if prediction == 1 else "✓ NO HOTSPOT"
        result_color = 'red' if prediction == 1 else 'green'

        ax.text(0.5, 0.05, result_text, transform=ax.transAxes,
               fontsize=20, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.8,
                        edgecolor='white', linewidth=2),
               color='white')

        ax.text(0.5, 0.15, f'Confidence: {confidence:.2%}', transform=ax.transAxes,
               fontsize=14, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               color='white')

        ax.set_title('Real-Time Hotspot Detection', fontsize=14, fontweight='bold')
        ax.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[+] Real-time detection plot saved to {save_path}")

        plt.show()
