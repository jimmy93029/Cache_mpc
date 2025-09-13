import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TrajectoryOverlapAnalyzer:
    def __init__(self, test_dir, horizon, action_dim):
        self.test_dir = Path(test_dir)
        self.horizon = horizon
        self.action_dim = action_dim
        self.data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all planned action CSV files."""
        print("Loading planned action data...")
        
        for episode_dir in self.test_dir.glob("episode_*"):
            episode_num = episode_dir.name.split('_')[1]
            csv_files = list(episode_dir.glob("planned-actions-time*.csv"))
            
            episode_data = {}
            for csv_file in csv_files:
                time_step = int(csv_file.stem.split('time')[1])
                df = pd.read_csv(csv_file)
                episode_data[time_step] = df
            
            self.data[int(episode_num)] = episode_data
        
        print(f"Loaded data for {len(self.data)} episodes")
    
    def extract_action_sequences(self, episode, time_step):
        """Extract action sequences from a specific time step."""
        if episode not in self.data or time_step not in self.data[episode]:
            return None
        
        df = self.data[episode][time_step]
        sequences = []
        
        for _, row in df.iterrows():
            sequence = []
            for h in range(self.horizon):
                actions_at_h = []
                for action_dim in range(self.action_dim):
                    col_name = f't{time_step + h + 1}_action{action_dim}'
                    if col_name in df.columns:
                        actions_at_h.append(row[col_name])
                if actions_at_h:
                    sequence.extend(actions_at_h)
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def compute_trajectory_similarity(self, seq1, seq2, method='cosine'):
        """Compute similarity between two trajectory sequences."""
        if method == 'cosine':
            return cosine_similarity([seq1], [seq2])[0, 0]
        elif method == 'euclidean':
            return 1 / (1 + euclidean_distances([seq1], [seq2])[0, 0])
        else:
            # Normalized correlation
            return np.corrcoef(seq1, seq2)[0, 1]
    
    def analyze_temporal_overlap(self, episode, similarity_threshold=0.95):
        """Analyze overlap between consecutive time steps."""
        if episode not in self.data:
            return {}
        
        time_steps = sorted(self.data[episode].keys())
        overlap_stats = {}
        
        for i in range(len(time_steps) - 1):
            t1, t2 = time_steps[i], time_steps[i + 1]
            
            seq1 = self.extract_action_sequences(episode, t1)
            seq2 = self.extract_action_sequences(episode, t2)
            
            if seq1 is None or seq2 is None:
                continue
            
            # Compute pairwise similarities
            similarities = []
            for s1 in seq1:
                for s2 in seq2:
                    sim = self.compute_trajectory_similarity(s1, s2)
                    similarities.append(sim)
            
            overlap_count = sum(1 for sim in similarities if sim > similarity_threshold)
            overlap_percentage = (overlap_count / len(similarities)) * 100
            
            overlap_stats[f"{t1}->{t2}"] = {
                'overlap_count': overlap_count,
                'total_comparisons': len(similarities),
                'overlap_percentage': overlap_percentage,
                'max_similarity': max(similarities),
                'avg_similarity': np.mean(similarities)
            }
        
        return overlap_stats
    
    def plot_similarity_heatmap(self, episode, time_step, save_path=None):
        """Plot similarity heatmap between elite trajectories."""
        sequences = self.extract_action_sequences(episode, time_step)
        if sequences is None:
            return
        
        # Compute similarity matrix
        n_trajs = len(sequences)
        similarity_matrix = np.zeros((n_trajs, n_trajs))
        
        for i in range(n_trajs):
            for j in range(n_trajs):
                similarity_matrix[i, j] = self.compute_trajectory_similarity(
                    sequences[i], sequences[j]
                )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, annot=True, cmap='viridis', 
                   square=True, fmt='.2f', cbar_kws={'label': 'Cosine Similarity'})
        plt.title(f'Trajectory Similarity Matrix\nEpisode {episode}, Time Step {time_step}')
        plt.xlabel('Trajectory Index')
        plt.ylabel('Trajectory Index')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_overlap_trends(self, episode, save_path=None):
        """Plot how overlap changes over time."""
        overlap_stats = self.analyze_temporal_overlap(episode)
        
        if not overlap_stats:
            return
        
        transitions = list(overlap_stats.keys())
        overlap_percentages = [overlap_stats[t]['overlap_percentage'] for t in transitions]
        avg_similarities = [overlap_stats[t]['avg_similarity'] for t in transitions]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Overlap percentage plot
        ax1.plot(range(len(transitions)), overlap_percentages, 'o-', linewidth=2, markersize=6)
        ax1.set_ylabel('Overlap Percentage (%)')
        ax1.set_title(f'Trajectory Overlap Over Time - Episode {episode}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(len(transitions)))
        ax1.set_xticklabels(transitions, rotation=45)
        
        # Average similarity plot
        ax2.plot(range(len(transitions)), avg_similarities, 's-', linewidth=2, markersize=6, color='orange')
        ax2.set_ylabel('Average Similarity')
        ax2.set_xlabel('Time Transition')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(len(transitions)))
        ax2.set_xticklabels(transitions, rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_clusters(self, episode, time_steps=None, save_path=None):
        """Plot PCA visualization of trajectory clusters."""
        if time_steps is None:
            time_steps = sorted(list(self.data[episode].keys()))[:5]  # First 5 time steps
        
        all_sequences = []
        time_labels = []
        
        for t in time_steps:
            sequences = self.extract_action_sequences(episode, t)
            if sequences is not None:
                all_sequences.extend(sequences)
                time_labels.extend([f't={t}'] * len(sequences))
        
        if len(all_sequences) < 2:
            return
        
        # Apply PCA
        pca = PCA(n_components=2)
        sequences_2d = pca.fit_transform(all_sequences)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(time_steps)))
        
        for i, t in enumerate(time_steps):
            mask = np.array(time_labels) == f't={t}'
            plt.scatter(sequences_2d[mask, 0], sequences_2d[mask, 1], 
                       c=[colors[i]], label=f'Time {t}', alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'PCA Visualization of Trajectory Clusters - Episode {episode}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_value_score_analysis(self, episode, time_step, save_path=None):
        """Analyze relationship between trajectory values and scores."""
        if episode not in self.data or time_step not in self.data[episode]:
            return
        
        df = self.data[episode][time_step]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Value vs Score scatter plot
        ax1.scatter(df['value'], df['score'], alpha=0.7, s=50)
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Score')
        ax1.set_title('Value vs Score Relationship')
        ax1.grid(True, alpha=0.3)
        
        # Value distribution
        ax2.hist(df['value'], bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Value Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Score distribution
        ax3.hist(df['score'], bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Rank correlation
        rank_corr = df['value'].corr(df['score'], method='spearman')
        ax4.text(0.5, 0.5, f'Spearman Correlation:\n{rank_corr:.3f}', 
                transform=ax4.transAxes, ha='center', va='center', 
                fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_title('Value-Score Correlation')
        ax4.axis('off')
        
        plt.suptitle(f'Value-Score Analysis - Episode {episode}, Time {time_step}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, episode=1, output_dir=None):
        """Generate a comprehensive overlap analysis report."""
        if output_dir is None:
            output_dir = self.test_dir / 'analysis_plots'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Generating comprehensive overlap analysis for episode {episode}...")
        
        # Get available time steps for this episode
        if episode not in self.data:
            print(f"No data found for episode {episode}")
            return
        
        time_steps = sorted(self.data[episode].keys())
        print(f"Available time steps: {time_steps}")
        
        # 1. Similarity heatmap for first few time steps
        for t in time_steps[:3]:
            self.plot_similarity_heatmap(
                episode, t, 
                save_path=output_dir / f'similarity_heatmap_ep{episode}_t{t}.png'
            )
        
        # 2. Temporal overlap trends
        self.plot_temporal_overlap_trends(
            episode, 
            save_path=output_dir / f'temporal_overlap_ep{episode}.png'
        )
        
        # 3. PCA clusters
        self.plot_pca_clusters(
            episode, time_steps[:5],
            save_path=output_dir / f'pca_clusters_ep{episode}.png'
        )
        
        # 4. Value-score analysis
        for t in time_steps[:2]:
            self.plot_value_score_analysis(
                episode, t,
                save_path=output_dir / f'value_score_analysis_ep{episode}_t{t}.png'
            )
        
        print(f"Analysis complete. Plots saved to {output_dir}")


def main():
    """Example usage of the overlap analyzer."""
    # You need to specify the path to your test results
    test_dir = Path("test/your_task/your_exp/your_seed/1")  # Adjust this path
    
    # These should match your environment configuration
    horizon = 10  # Adjust based on your cfg.horizon
    action_dim = 4  # Adjust based on your environment
    
    # Create analyzer
    analyzer = TrajectoryOverlapAnalyzer(test_dir, horizon, action_dim)
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report(episode=1)


if __name__ == '__main__':
    main()