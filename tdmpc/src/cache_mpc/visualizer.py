import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from dtaidistance import dtw
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

class CacheMPCVisualizer:
    """Cache MPC 专用可视化工具类"""
    
    def __init__(self, horizon, action_dim):
        self.horizon = horizon
        self.action_dim = action_dim
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def load_planned_actions_data(self, test_dir, episode_num=1, max_time_steps=10):
        """从CSV文件加载轨迹数据"""
        import os
        import glob
        
        episode_dir = f"{test_dir}/episode_{episode_num}"
        csv_files = glob.glob(f"{episode_dir}/planned-actions-time*.csv")
        
        trajectories_data = {}
        for csv_file in sorted(csv_files)[:max_time_steps]:
            # 提取时间步
            filename = os.path.basename(csv_file)
            time_step = int(filename.split('time')[1].split('.')[0])
            
            # 读取CSV数据
            df = pd.read_csv(csv_file)
            
            # 提取轨迹序列
            trajectories = []
            for _, row in df.iterrows():
                trajectory = []
                for h in range(self.horizon):
                    for a in range(self.action_dim):
                        col_name = f't{time_step + h + 1}_action{a}'
                        if col_name in df.columns:
                            trajectory.append(row[col_name])
                trajectories.append(np.array(trajectory))
            
            trajectories_data[time_step] = {
                'trajectories': np.array(trajectories),
                'values': df['value'].values,
                'scores': df['score'].values
            }
        
        return trajectories_data
    
    def trajectory_bundle_visualization(self, trajectories_data, save_path=None, 
                                      show_top_k=5, action_dim_to_plot=0):
        """
        1. 轨迹束可视化 (Trajectory Bundle)
        展示多条轨迹的temporal evolution和重叠模式
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        time_steps = sorted(trajectories_data.keys())
        
        # 1.1 单个时间步的轨迹束 (左上)
        ax1 = axes[0, 0]
        
        if time_steps:
            first_time = time_steps[0]
            data = trajectories_data[first_time]
            trajs = data['trajectories']
            values = data['values']
            
            # 选择top-k高价值轨迹
            top_indices = np.argsort(values)[-show_top_k:]
            
            horizon_steps = np.arange(self.horizon)
            
            for i, idx in enumerate(top_indices):
                # 提取第一个动作维度的轨迹
                traj_1d = trajs[idx].reshape(self.horizon, self.action_dim)[:, action_dim_to_plot]
                
                alpha = 0.8 if i == len(top_indices)-1 else 0.6  # 最高价值轨迹更突出
                linewidth = 3 if i == len(top_indices)-1 else 2
                
                ax1.plot(horizon_steps, traj_1d, 
                        color=self.colors[i % len(self.colors)], 
                        alpha=alpha, linewidth=linewidth,
                        label=f'轨迹 {idx+1} (v={values[idx]:.3f})')
            
            ax1.set_xlabel('规划步骤')
            ax1.set_ylabel(f'动作维度 {action_dim_to_plot}')
            ax1.set_title(f'时间步 {first_time} - Top {show_top_k} 轨迹束')
            ax1.legend(loc='best', fontsize=9)
            ax1.grid(True, alpha=0.3)
        
        # 1.2 多时间步轨迹演化 (右上)
        ax2 = axes[0, 1]
        
        # 选择每个时间步的最佳轨迹进行对比
        best_trajectories = []
        time_labels = []
        
        for t in time_steps[:5]:  # 只显示前5个时间步
            data = trajectories_data[t]
            best_idx = np.argmax(data['values'])
            best_traj = data['trajectories'][best_idx].reshape(self.horizon, self.action_dim)
            
            ax2.plot(np.arange(self.horizon), best_traj[:, action_dim_to_plot],
                    color=self.colors[t % len(self.colors)], 
                    linewidth=2, alpha=0.8, label=f't={t}')
        
        ax2.set_xlabel('规划步骤')
        ax2.set_ylabel(f'动作维度 {action_dim_to_plot}')
        ax2.set_title('最优轨迹的时序演化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 1.3 轨迹重叠区域可视化 (左下)
        ax3 = axes[1, 0]
        
        if len(time_steps) >= 2:
            t1, t2 = time_steps[0], time_steps[1]
            data1, data2 = trajectories_data[t1], trajectories_data[t2]
            
            # 选择两个时间步的最佳轨迹
            best_idx1 = np.argmax(data1['values'])
            best_idx2 = np.argmax(data2['values'])
            
            traj1 = data1['trajectories'][best_idx1].reshape(self.horizon, self.action_dim)
            traj2 = data2['trajectories'][best_idx2].reshape(self.horizon, self.action_dim)
            
            horizon_steps = np.arange(self.horizon)
            
            # 绘制两条轨迹
            line1 = ax3.plot(horizon_steps, traj1[:, action_dim_to_plot], 
                           'b-', linewidth=3, label=f'时间步 {t1}')[0]
            line2 = ax3.plot(horizon_steps, traj2[:, action_dim_to_plot], 
                           'r-', linewidth=3, label=f'时间步 {t2}')[0]
            
            # 填充重叠区域
            ax3.fill_between(horizon_steps, 
                           traj1[:, action_dim_to_plot], 
                           traj2[:, action_dim_to_plot],
                           alpha=0.3, color='yellow', label='差异区域')
            
            # 计算MAE差异
            mae_diff = mean_absolute_error(traj1[:, action_dim_to_plot], 
                                         traj2[:, action_dim_to_plot])
            ax3.text(0.02, 0.98, f'MAE差异: {mae_diff:.4f}', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax3.set_xlabel('规划步骤')
            ax3.set_ylabel(f'动作维度 {action_dim_to_plot}')
            ax3.set_title('相邻时间步轨迹重叠分析')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 1.4 轨迹束统计信息 (右下)
        ax4 = axes[1, 1]
        
        # 计算每个时间步的轨迹多样性指标
        diversity_metrics = []
        time_points = []
        
        for t in time_steps:
            data = trajectories_data[t]
            trajs = data['trajectories']
            
            # 计算轨迹间的平均MAE距离
            n_trajs = len(trajs)
            total_distance = 0
            count = 0
            
            for i in range(n_trajs):
                for j in range(i+1, n_trajs):
                    distance = mean_absolute_error(trajs[i], trajs[j])
                    total_distance += distance
                    count += 1
            
            avg_diversity = total_distance / count if count > 0 else 0
            diversity_metrics.append(avg_diversity)
            time_points.append(t)
        
        ax4.plot(time_points, diversity_metrics, 'go-', linewidth=2, markersize=8)
        ax4.set_xlabel('时间步')
        ax4.set_ylabel('平均轨迹多样性 (MAE)')
        ax4.set_title('轨迹束多样性随时间变化')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('轨迹束可视化 - Cache MPC 重叠分析', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def similarity_network_visualization(self, trajectories_data, 
                                       similarity_threshold=0.8, save_path=None):
        """
        2. 相似性网络图 (Similarity Network)
        将轨迹作为节点，相似性作为边，直观显示重用关系
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        time_steps = sorted(trajectories_data.keys())
        
        # 2.1 单时间步内的轨迹相似性网络 (左上)
        ax1 = axes[0, 0]
        
        if time_steps:
            first_time = time_steps[0]
            data = trajectories_data[first_time]
            trajs = data['trajectories'][:10]  # 只取前10个轨迹避免过于复杂
            values = data['values'][:10]
            
            # 计算相似性矩阵 (使用负MAE作为相似性)
            n_trajs = len(trajs)
            similarity_matrix = np.zeros((n_trajs, n_trajs))
            
            for i in range(n_trajs):
                for j in range(n_trajs):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        mae_dist = mean_absolute_error(trajs[i], trajs[j])
                        # 转换为相似性 (距离越小，相似性越高)
                        similarity_matrix[i, j] = 1 / (1 + mae_dist)
            
            # 创建网络图
            G = nx.Graph()
            
            # 添加节点 (轨迹)
            for i in range(n_trajs):
                G.add_node(i, value=values[i])
            
            # 添加边 (相似性超过阈值的轨迹对)
            edges_added = []
            for i in range(n_trajs):
                for j in range(i+1, n_trajs):
                    if similarity_matrix[i, j] > similarity_threshold:
                        G.add_edge(i, j, weight=similarity_matrix[i, j])
                        edges_added.append((i, j, similarity_matrix[i, j]))
            
            if len(G.nodes()) > 0:
                # 使用spring layout
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
                
                # 绘制边
                edges = G.edges()
                edge_weights = [G[u][v]['weight'] for u, v in edges]
                
                nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.6, 
                                     width=[w*3 for w in edge_weights],
                                     edge_color='gray')
                
                # 绘制节点 - 大小根据价值，颜色根据价值
                node_sizes = [300 + v*200 for v in values]  # 基础大小300，根据价值调整
                node_colors = values
                
                nodes = nx.draw_networkx_nodes(G, pos, ax=ax1,
                                             node_size=node_sizes,
                                             node_color=node_colors,
                                             cmap='viridis', alpha=0.8)
                
                # 添加节点标签
                nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_weight='bold')
                
                # 添加边权重标签 (只显示权重高的边)
                edge_labels = {(i, j): f'{w:.3f}' 
                             for i, j, w in edges_added if w > similarity_threshold + 0.05}
                nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                           font_size=7, ax=ax1)
                
                # 添加颜色条
                if nodes:
                    cbar = plt.colorbar(nodes, ax=ax1)
                    cbar.set_label('轨迹价值', rotation=270, labelpad=15)
            
            ax1.set_title(f'时间步 {first_time} 轨迹相似性网络\n(阈值 > {similarity_threshold})')
            ax1.axis('off')
        
        # 2.2 跨时间步的轨迹重用网络 (右上)
        ax2 = axes[0, 1]
        
        if len(time_steps) >= 2:
            # 创建跨时间步的网络
            G_temporal = nx.Graph()
            
            # 收集所有轨迹数据
            all_trajectories = []
            all_values = []
            all_time_labels = []
            trajectory_ids = []
            
            for t_idx, t in enumerate(time_steps[:3]):  # 只用前3个时间步
                data = trajectories_data[t]
                top_indices = np.argsort(data['values'])[-5:]  # 每个时间步取top5
                
                for local_idx, global_idx in enumerate(top_indices):
                    node_id = f"t{t}_traj{local_idx}"
                    trajectory_ids.append(node_id)
                    all_trajectories.append(data['trajectories'][global_idx])
                    all_values.append(data['values'][global_idx])
                    all_time_labels.append(t)
                    G_temporal.add_node(node_id, time=t, value=data['values'][global_idx])
            
            # 计算跨时间的相似性并添加边
            for i in range(len(all_trajectories)):
                for j in range(i+1, len(all_trajectories)):
                    # 只连接不同时间步的轨迹
                    if all_time_labels[i] != all_time_labels[j]:
                        mae_dist = mean_absolute_error(all_trajectories[i], all_trajectories[j])
                        similarity = 1 / (1 + mae_dist)
                        
                        if similarity > similarity_threshold:
                            G_temporal.add_edge(trajectory_ids[i], trajectory_ids[j], 
                                              weight=similarity)
            
            if len(G_temporal.nodes()) > 0:
                # 使用分层布局
                pos_temporal = {}
                time_positions = {t: idx for idx, t in enumerate(sorted(set(all_time_labels)))}
                
                # 为每个时间步的节点分配位置
                for node in G_temporal.nodes():
                    time_step = G_temporal.nodes[node]['time']
                    x = time_positions[time_step] * 3  # 时间步间距
                    
                    # 同一时间步内的节点垂直分布
                    same_time_nodes = [n for n in G_temporal.nodes() 
                                     if G_temporal.nodes[n]['time'] == time_step]
                    y_offset = same_time_nodes.index(node) - len(same_time_nodes)/2
                    y = y_offset * 0.8
                    
                    pos_temporal[node] = (x, y)
                
                # 绘制边
                edge_weights_temporal = [G_temporal[u][v]['weight'] for u, v in G_temporal.edges()]
                nx.draw_networkx_edges(G_temporal, pos_temporal, ax=ax2, alpha=0.6,
                                     width=[w*4 for w in edge_weights_temporal],
                                     edge_color='red')
                
                # 绘制节点
                node_colors_temporal = [G_temporal.nodes[node]['value'] for node in G_temporal.nodes()]
                node_times = [G_temporal.nodes[node]['time'] for node in G_temporal.nodes()]
                
                # 根据时间步使用不同形状
                for time_step in set(node_times):
                    time_nodes = [node for node in G_temporal.nodes() 
                                if G_temporal.nodes[node]['time'] == time_step]
                    time_pos = {node: pos_temporal[node] for node in time_nodes}
                    time_colors = [G_temporal.nodes[node]['value'] for node in time_nodes]
                    
                    markers = ['o', 's', '^']  # 圆形、方形、三角形
                    marker = markers[time_step % len(markers)]
                    
                    nx.draw_networkx_nodes(G_temporal, time_pos, nodelist=time_nodes,
                                         node_color=time_colors, node_shape=marker,
                                         node_size=200, cmap='plasma', alpha=0.8, ax=ax2)
                
                # 添加标签
                nx.draw_networkx_labels(G_temporal, pos_temporal, ax=ax2, font_size=6)
            
            ax2.set_title('跨时间步轨迹重用网络')
            ax2.axis('off')
        
        # 2.3 相似性分布直方图 (左下)
        ax3 = axes[1, 0]
        
        all_similarities = []
        
        for t in time_steps[:3]:  # 分析前3个时间步
            data = trajectories_data[t]
            trajs = data['trajectories'][:10]  # 前10个轨迹
            
            for i in range(len(trajs)):
                for j in range(i+1, len(trajs)):
                    mae_dist = mean_absolute_error(trajs[i], trajs[j])
                    similarity = 1 / (1 + mae_dist)
                    all_similarities.append(similarity)
        
        ax3.hist(all_similarities, bins=30, alpha=0.7, color='skyblue', 
                edgecolor='black', density=True)
        ax3.axvline(similarity_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'阈值 = {similarity_threshold}')
        ax3.axvline(np.mean(all_similarities), color='green', linestyle='--', linewidth=2,
                   label=f'平均值 = {np.mean(all_similarities):.3f}')
        
        ax3.set_xlabel('相似性得分')
        ax3.set_ylabel('密度')
        ax3.set_title('轨迹相似性分布')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 2.4 网络统计信息 (右下)
        ax4 = axes[1, 1]
        
        # 分析不同阈值下的网络连通性
        thresholds = np.arange(0.5, 1.0, 0.05)
        connectivity_stats = []
        
        for threshold in thresholds:
            if time_steps:
                first_time = time_steps[0]
                data = trajectories_data[first_time]
                trajs = data['trajectories'][:10]
                
                # 创建临时网络
                temp_G = nx.Graph()
                temp_G.add_nodes_from(range(len(trajs)))
                
                for i in range(len(trajs)):
                    for j in range(i+1, len(trajs)):
                        mae_dist = mean_absolute_error(trajs[i], trajs[j])
                        similarity = 1 / (1 + mae_dist)
                        if similarity > threshold:
                            temp_G.add_edge(i, j)
                
                # 计算连通性统计
                num_edges = temp_G.number_of_edges()
                num_components = nx.number_connected_components(temp_G)
                max_component_size = max([len(c) for c in nx.connected_components(temp_G)]) if num_edges > 0 else 1
                
                connectivity_stats.append({
                    'threshold': threshold,
                    'edges': num_edges,
                    'components': num_components,
                    'max_component': max_component_size
                })
        
        if connectivity_stats:
            stats_df = pd.DataFrame(connectivity_stats)
            
            ax4_twin = ax4.twinx()
            
            line1 = ax4.plot(stats_df['threshold'], stats_df['edges'], 'b-o', 
                           label='边数量', linewidth=2, markersize=6)
            line2 = ax4_twin.plot(stats_df['threshold'], stats_df['max_component'], 'r-s',
                                label='最大连通分量', linewidth=2, markersize=6)
            
            ax4.set_xlabel('相似性阈值')
            ax4.set_ylabel('边数量', color='blue')
            ax4_twin.set_ylabel('最大连通分量大小', color='red')
            ax4.set_title('网络连通性 vs 相似性阈值')
            
            # 合并图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('轨迹相似性网络分析 - Cache MPC', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def temporal_overlap_heatmap(self, trajectories_data, save_path=None, window_size=3):
        """
        4. 时序重叠热图
        显示不同时间窗口的重叠情况，帮助确定最佳缓存时机
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        time_steps = sorted(trajectories_data.keys())
        
        # 4.1 时间步间的全局相似性热图 (左上)
        ax1 = axes[0, 0]
        
        if len(time_steps) >= 2:
            n_times = len(time_steps)
            global_similarity_matrix = np.zeros((n_times, n_times))
            
            for i, t1 in enumerate(time_steps):
                for j, t2 in enumerate(time_steps):
                    if i == j:
                        global_similarity_matrix[i, j] = 1.0
                    else:
                        data1 = trajectories_data[t1]
                        data2 = trajectories_data[t2]
                        
                        # 选择最优轨迹进行比较
                        best_idx1 = np.argmax(data1['values'])
                        best_idx2 = np.argmax(data2['values'])
                        
                        best_traj1 = data1['trajectories'][best_idx1]
                        best_traj2 = data2['trajectories'][best_idx2]
                        
                        mae_dist = mean_absolute_error(best_traj1, best_traj2)
                        similarity = 1 / (1 + mae_dist)
                        global_similarity_matrix[i, j] = similarity
            
            im1 = ax1.imshow(global_similarity_matrix, cmap='Blues', aspect='auto', 
                           vmin=0, vmax=1)
            
            # 添加数值标注
            for i in range(n_times):
                for j in range(n_times):
                    text_color = 'white' if global_similarity_matrix[i, j] > 0.5 else 'black'
                    ax1.text(j, i, f'{global_similarity_matrix[i, j]:.3f}',
                           ha="center", va="center", color=text_color, fontsize=9)
            
            ax1.set_xticks(range(n_times))
            ax1.set_yticks(range(n_times))
            ax1.set_xticklabels([f't={t}' for t in time_steps])
            ax1.set_yticklabels([f't={t}' for t in time_steps])
            ax1.set_xlabel('时间步 j')
            ax1.set_ylabel('时间步 i')
            ax1.set_title('时间步间最优轨迹相似性')
            
            plt.colorbar(im1, ax=ax1, label='相似性得分')
        
        # 4.2 滑动窗口重叠分析 (右上)
        ax2 = axes[0, 1]
        
        if len(time_steps) >= window_size:
            window_overlaps = []
            window_centers = []
            
            for i in range(len(time_steps) - window_size + 1):
                window_times = time_steps[i:i + window_size]
                window_center = np.mean(window_times)
                
                # 计算窗口内所有轨迹对的平均相似性
                window_similarities = []
                
                for t1 in window_times:
                    for t2 in window_times:
                        if t1 != t2:
                            data1 = trajectories_data[t1]
                            data2 = trajectories_data[t2]
                            
                            # 计算top轨迹间的相似性
                            top_indices1 = np.argsort(data1['values'])[-3:]  # top 3
                            top_indices2 = np.argsort(data2['values'])[-3:]  # top 3
                            
                            for idx1 in top_indices1:
                                for idx2 in top_indices2:
                                    mae_dist = mean_absolute_error(
                                        data1['trajectories'][idx1],
                                        data2['trajectories'][idx2]
                                    )
                                    similarity = 1 / (1 + mae_dist)
                                    window_similarities.append(similarity)
                
                avg_similarity = np.mean(window_similarities) if window_similarities else 0
                window_overlaps.append(avg_similarity)
                window_centers.append(window_center)
            
            ax2.plot(window_centers, window_overlaps, 'ro-', linewidth=3, markersize=8)
            ax2.fill_between(window_centers, window_overlaps, alpha=0.3, color='red')
            
            ax2.set_xlabel('窗口中心时间')
            ax2.set_ylabel('平均相似性')
            ax2.set_title(f'滑动窗口重叠分析 (窗口大小={window_size})')
            ax2.grid(True, alpha=0.3)
        
        # 4.3 缓存命中率预测热图 (左下)
        ax3 = axes[1, 0]
        
        # 模拟缓存命中率矩阵
        cache_hit_matrix = np.zeros((len(time_steps), len(time_steps)))
        
        for i, t_cache in enumerate(time_steps):  # 缓存时间
            for j, t_query in enumerate(time_steps):  # 查询时间
                if i <= j:  # 只能使用过去的缓存
                    # 基于相似性计算命中率
                    if i < len(time_steps) and j < len(time_steps):
                        time_gap = t_query - t_cache
                        
                        # 时间间隔越大，命中率越低
                        decay_factor = np.exp(-time_gap * 0.1)
                        
                        if t_cache in trajectories_data and t_query in trajectories_data:
                            data_cache = trajectories_data[t_cache]
                            data_query = trajectories_data[t_query]
                            
                            best_idx_cache = np.argmax(data_cache['values'])
                            best_idx_query = np.argmax(data_query['values'])
                            
                            mae_dist = mean_absolute_error(
                                data_cache['trajectories'][best_idx_cache],
                                data_query['trajectories'][best_idx_query]
                            )
                            similarity = 1 / (1 + mae_dist)
                            
                            cache_hit_rate = similarity * decay_factor
                            cache_hit_matrix[i, j] = cache_hit_rate
        
        im3 = ax3.imshow(cache_hit_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax3.set_xticks(range(len(time_steps)))
        ax3.set_yticks(range(len(time_steps)))
        ax3.set_xticklabels([f't={t}' for t in time_steps])
        ax3.set_yticklabels([f't={t}' for t in time_steps])
        ax3.set_xlabel('查询时间')
        ax3.set_ylabel('缓存时间')
        ax3.set_title('缓存命中率预测矩阵')
        
        plt.colorbar(im3, ax=ax3, label='预测命中率')
        
        # 4.4 最佳缓存时机分析 (右下)
        ax4 = axes[1, 1]
        
        # 计算每个时间步作为缓存起点的总体效率
        cache_efficiency = []
        
        for i, t_start in enumerate(time_steps[:-1]):  # 不包括最后一个时间步
            efficiency_scores = []
            
            for j, t_end in enumerate(time_steps[i+1:], start=i+1):
                if t_start in trajectories_data and t_end in trajectories_data:
                    data_start = trajectories_data[t_start]
                    data_end = trajectories_data[t_end]
                    
                    # 计算缓存效率 = 相似性 * 价值权重 / 时间成本
                    best_idx_start = np.argmax(data_start['values'])
                    best_idx_end = np.argmax(data_end['values'])
                    
                    mae_dist = mean_absolute_error(
                        data_start['trajectories'][best_idx_start],
                        data_end['trajectories'][best_idx_end]
                    )
                    similarity = 1 / (1 + mae_dist)
                    
                    # 价值权重
                    value_weight = (data_start['values'][best_idx_start] + 
                                  data_end['values'][best_idx_end]) / 2
                    
                    # 时间成本 (时间间隔越大成本越高)
                    time_cost = 1 + (t_end - t_start) * 0.1
                    
                    efficiency = similarity * value_weight / time_cost
                    efficiency_scores.append(efficiency)
            
            avg_efficiency = np.mean(efficiency_scores) if efficiency_scores else 0
            cache_efficiency.append(avg_efficiency)
        
        if cache_efficiency:
            bars = ax4.bar(range(len(cache_efficiency)), cache_efficiency, 
                          color='lightcoral', alpha=0.7, edgecolor='black')
            
            # 标记最佳缓存时机
            best_cache_idx = np.argmax(cache_efficiency)
            bars[best_cache_idx].set_color('darkred')
            bars[best_cache_idx].set_alpha(1.0)
            
            ax4.set_xticks(range(len(cache_efficiency)))
            ax4.set_xticklabels([f't={time_steps[i]}' for i in range(len(cache_efficiency))])
            ax4.set_xlabel('缓存起始时间')
            ax4.set_ylabel('缓存效率得分')
            ax4.set_title('最佳缓存时机分析')
            
            # 添加最佳时机标注
            ax4.text(best_cache_idx, cache_efficiency[best_cache_idx] + 0.01,
                    '最佳时机', ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.suptitle('时序重叠热图分析 - Cache MPC 最优时机', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# 使用示例和完整分析流程
def run_complete_cache_mpc_analysis(test_dir, horizon=10, action_dim=4, episode_num=1):
    """运行完整的Cache MPC可视化分析流程"""
    
    print("🚀 开始Cache MPC轨迹重叠分析...")
    
    # 初始化可视化工具
    visualizer = CacheMPCVisualizer(horizon, action_dim)
    
    # 加载数据
    print("📊 加载轨迹数据...")
    try:
        trajectories_data = visualizer.load_planned_actions_data(test_dir, episode_num)
        print(f"✓ 成功加载 {len(trajectories_data)} 个时间步的数据")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None
    
    # 创建输出目录
    import os
    output_dir = f"{test_dir}/visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 轨迹束可视化
    print("🎯 生成轨迹束可视化...")
    bundle_fig = visualizer.trajectory_bundle_visualization(
        trajectories_data, 
        save_path=f"{output_dir}/trajectory_bundle_analysis.png"
    )
    
    # 2. 相似性网络图
    print("🕸️ 生成相似性网络图...")
    network_fig = visualizer.similarity_network_visualization(
        trajectories_data,
        similarity_threshold=0.8,
        save_path=f"{output_dir}/similarity_network_analysis.png"
    )
    
    # 3. 时序重叠热图
    print("🔥 生成时序重叠热图...")
    heatmap_fig = visualizer.temporal_overlap_heatmap(
        trajectories_data,
        save_path=f"{output_dir}/temporal_overlap_heatmap.png"
    )
    
    print(f"✨ 分析完成！结果保存在: {output_dir}")
    print("\n📋 分析报告:")
    print("1. trajectory_bundle_analysis.png - 轨迹束可视化")
    print("2. similarity_network_analysis.png - 相似性网络图") 
    print("3. temporal_overlap_heatmap.png - 时序重叠热图")
    
    return {
        'bundle_fig': bundle_fig,
        'network_fig': network_fig, 
        'heatmap_fig': heatmap_fig,
        'data': trajectories_data
    }

# 使用指南
if __name__ == "__main__":
    # 示例使用
    print("Cache MPC 可视化工具包")
    print("=" * 50)
    print("使用方法:")
    print("1. 确保已运行test.py收集轨迹数据")
    print("2. 调用 run_complete_cache_mpc_analysis(test_dir, horizon, action_dim)")
    print("3. 查看生成的可视化结果")
    print("\n参数说明:")
    print("- test_dir: 测试数据目录路径")
    print("- horizon: 规划地平线长度") 
    print("- action_dim: 动作维度数量")
    print("- episode_num: 要分析的episode编号")
    
    # 示例调用（取消注释以使用）
    # test_directory = "test/your_task/your_exp/your_seed/1"
    # results = run_complete_cache_mpc_analysis(test_directory, horizon=10, action_dim=4)
    
    plt.show()