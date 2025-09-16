import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import glob
import re


class CartpoleTrajectoryBundleVisualizer:
    """专门为Cartpole CSV数据设计的轨迹束可视化器"""
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        

    def load_data(self, episode_dir):
        """加载cartpole CSV数据"""

        episode_path = Path(episode_dir)
        csv_files = list(episode_path.glob("planned-actions-time*.csv"))
        
        # Extract time steps from filenames
        time_steps = sorted([int(f.stem.split('time')[-1]) for f in csv_files])
        
        trajectories_data = {}
        
        # Load data for existing time steps
        for time_step in time_steps:
            csv_file = episode_path / f"planned-actions-time{time_step}.csv"
            if not csv_file.exists():
                pass

            df = pd.read_csv(csv_file)

            # 解析列名获取horizon和action_dim
            action_cols = [col for col in df.columns if col.startswith('t') and 'action' in col]
            
            if not action_cols:
                continue
                
            # 提取horizon和action维度信息
            time_steps_in_traj = set()
            action_dims = set()
            
            for col in action_cols:
                # 解析格式如 "t1_action0"
                parts = col.split('_')
                if len(parts) == 2:
                    t_part = parts[0][1:]  # 去掉't'
                    action_part = parts[1][6:]  # 去掉'action'
                    time_steps_in_traj.add(int(t_part))
                    action_dims.add(int(action_part))
            
            horizon = len(time_steps_in_traj)
            action_dim = len(action_dims)
            
            # 重构轨迹数据
            trajectories = []
            for _, row in df.iterrows():
                trajectory = []
                for t in sorted(time_steps_in_traj):  # 遍历时间步
                    time_step_actions = []  # 存储当前时间步的所有行动维度数据
                    for a in sorted(action_dims):  # 遍历每个行动维度
                        col_name = f't{t}_action{a}'
                        if col_name in df.columns:  # 如果该列存在
                            time_step_actions.append(row[col_name])  # 将该维度的值加入当前时间步
                    trajectory.append(time_step_actions)  # 将当前时间步的所有行动维度加入轨迹
                trajectories.append(np.array(trajectory))  # 将整个轨迹加入结果列表
            
            trajectories_data[time_step] = {
                'trajectories': np.array(trajectories, dtype=object),
                'values': df['value'].values if 'value' in df.columns else np.ones(len(df)),
                'scores': df['score'].values if 'score' in df.columns else np.ones(len(df)),
                'horizon': horizon,
                'action_dim': action_dim
            }
        
        print(f"加载了 {len(trajectories_data)} 个时间步的数据")
        if trajectories_data:
            first_data = list(trajectories_data.values())[0]
            print(f"规划地平线: {first_data['horizon']}")
            print(f"动作维度: {first_data['action_dim']}")
            print(f"每个时间步的轨迹数: {len(first_data['trajectories'])}")
        
        return trajectories_data


    def create_three_bundle_plots(self, trajectories_data, t1, t2, action_dim_to_plot=0, save_path=None):
        """创建三个核心轨迹束图 - 修正时间偏移，使用全部轨迹"""
        
        if len(trajectories_data) < 2:
            print("需要至少2个时间步的数据来创建对比图")
            return None

        if action_dim_to_plot >= trajectories_data[0]['action_dim']:
            print("欲呈現動作維度 > 實際維度")
            return None

        if t2 not in trajectories_data:
            print("時間點超出邊界")
            return None
        
        # 创建图形
        # ===== 图1: 轨迹束对比 (正确的时间计算) =====
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax1 = axes[0]

        # 选择两个时间步进行对比
        data1 = trajectories_data[t1]
        data2 = trajectories_data[t2]

        # 使用全部轨迹
        trajs1 = data1['trajectories']  # 全部轨迹 (64个)
        trajs2 = data2['trajectories'] 

        print(f"绘制轨迹数量: Bundle1={len(trajs1)}, Bundle2={len(trajs2)}")

        # 绘制第一束轨迹 (蓝色) - t1时间步的所有轨迹
        for traj_idx, traj in enumerate(trajs1):
            # traj格式: traj[t_idx][action_dim]
            # 实际时间 = t_idx + t1
            
            action_values = []
            time_positions = []
            
            for t_idx in range(len(traj)):  # 遍历轨迹中的每个时间步位置
                actual_time = t_idx + t1  # 计算实际时间
                
                if isinstance(traj[t_idx], (list, np.ndarray)):
                    
                    action_value = traj[t_idx][action_dim_to_plot]
                    action_values.append(action_value)
                    time_positions.append(actual_time)
            
            # 绘制这条轨迹
            if len(action_values) > 0:
                ax1.plot(time_positions, action_values, color='blue', alpha=0.3, linewidth=1)

        # 绘制第二束轨迹 (红色) - t2时间步的所有轨迹
        if t2 in trajectories_data:
            for traj_idx, traj in enumerate(trajs2):
                # 实际时间 = t_idx + t2
                action_values = []
                time_positions = []
                
                for t_idx in range(len(traj)):  # 遍历轨迹中的每个时间步位置
                    actual_time = t_idx + t2  # 计算实际时间
                    
                    if isinstance(traj[t_idx], (list, np.ndarray)):
                        action_value = traj[t_idx][action_dim_to_plot]
                        action_values.append(action_value)
                        time_positions.append(actual_time)
                
                # 绘制这条轨迹
                if len(action_values) > 0:
                    ax1.plot(time_positions, action_values, color='red', alpha=0.3, linewidth=1)

        # 设置图形属性
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.set_title(f'3. Trajectory Bundle Comparison\nComparing two bundles (t={t1} vs t={t2})')
        ax1.grid(True, alpha=0.3)

        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, alpha=0.7, label=f't={t1} Bundle ({len(trajs1)} trajectories)'),
            Line2D([0], [0], color='red', lw=2, alpha=0.7, label=f't={t2} Bundle ({len(trajs2)} trajectories)')
        ]
        ax1.legend(handles=legend_elements)

        # 计算并显示时间范围信息
        horizon = data1['horizon']
        t1_time_range = [t1, t1 + horizon - 1]
        t2_time_range = [t2, t2 + horizon - 1]

        ax1.text(0.02, 0.98, 
                f'Time ranges:\nBlue: [{t1_time_range[0]}, {t1_time_range[1]}]\nRed: [{t2_time_range[0]}, {t2_time_range[1]}]', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=9)
        
        # ===== 图2: 轨迹束演化 (修正时间计算) =====
        ax2 = axes[1]

        # 绘制第一个时间步的轨迹束 (蓝色)
        for i, traj in enumerate(trajs1):
            action_sequence = []
            time_positions = []
            
            for t_idx in range(len(traj)):
                actual_time = t_idx + t1  # 正确的时间计算
                
                if (isinstance(traj[t_idx], (list, np.ndarray)) and 
                    action_dim_to_plot < len(traj[t_idx])):
                    action_sequence.append(traj[t_idx][action_dim_to_plot])
                    time_positions.append(actual_time)
            
            if len(action_sequence) > 0:
                ax2.plot(time_positions, action_sequence, color='blue', alpha=0.3, linewidth=1)

        # 绘制第二个时间步的轨迹束 (红色)
        for i, traj in enumerate(trajs2):
            action_sequence = []
            time_positions = []
            
            for t_idx in range(len(traj)):
                actual_time = t_idx + t2  # 正确的时间计算
                
                if (isinstance(traj[t_idx], (list, np.ndarray)) and 
                    action_dim_to_plot < len(traj[t_idx])):
                    action_sequence.append(traj[t_idx][action_dim_to_plot])
                    time_positions.append(actual_time)
            
            if len(action_sequence) > 0:
                ax2.plot(time_positions, action_sequence, color='red', alpha=0.3, linewidth=1)

        # 添加图例
        from matplotlib.lines import Line2D
        horizon = data1['horizon']
        t1_time_range = [t1, t1 + horizon - 1]
        t2_time_range = [t2, t2 + horizon - 1]

        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label=f't={t1} Bundle (time: {t1_time_range})'),
            Line2D([0], [0], color='red', lw=2, label=f't={t2} Bundle (time: {t2_time_range})')
        ]
        ax2.legend(handles=legend_elements)

        ax2.set_xlabel('Planning Steps (Actual Time)')
        ax2.set_ylabel('Action Value')
        ax2.set_title('5. Evolution of Trajectory Bundles\nTrajectory bundles change over time')
        ax2.grid(True, alpha=0.3)

        # ===== 图3: 轨迹束重叠分析 (修正时间计算) =====
        ax3 = axes[2]

        # 计算时间范围和重叠
        horizon = data1['horizon']
        t1_times = list(range(t1, t1 + horizon))  # [t1, t1+1, t1+2, ...]
        t2_times = list(range(t2, t2 + horizon))  # [t2, t2+1, t2+2, ...]

        # 找到重叠的时间区间
        overlap_times = sorted(set(t1_times).intersection(set(t2_times)))

        if not overlap_times:
            ax3.text(0.5, 0.5, 'No overlapping time indices\nbetween the two bundles', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('6. Trajectory Bundle Overlap Analysis\nNo overlap found')
            return fig

        print(f"重叠时间索引: {overlap_times}")

        # 提取重叠区间的数据
        bundle1_sequences = []
        bundle2_sequences = []

        for traj in trajs1:
            action_sequence = []
            for overlap_time in overlap_times:
                # 找到这个重叠时间在traj中的位置
                t_idx_in_traj1 = overlap_time - t1  # 在traj1中的索引
                
                if (0 <= t_idx_in_traj1 < len(traj) and
                    isinstance(traj[t_idx_in_traj1], (list, np.ndarray)) and
                    action_dim_to_plot < len(traj[t_idx_in_traj1])):
                    action_sequence.append(traj[t_idx_in_traj1][action_dim_to_plot])
                else:
                    action_sequence.append(0)
            
            if action_sequence:  # 只添加非空序列
                bundle1_sequences.append(action_sequence)

        for traj in trajs2:
            action_sequence = []
            for overlap_time in overlap_times:
                # 找到这个重叠时间在traj中的位置
                t_idx_in_traj2 = overlap_time - t2  # 在traj2中的索引
                
                if (0 <= t_idx_in_traj2 < len(traj) and
                    isinstance(traj[t_idx_in_traj2], (list, np.ndarray)) and
                    action_dim_to_plot < len(traj[t_idx_in_traj2])):
                    action_sequence.append(traj[t_idx_in_traj2][action_dim_to_plot])
                else:
                    action_sequence.append(0)
            
            if action_sequence:  # 只添加非空序列
                bundle2_sequences.append(action_sequence)

        if not bundle1_sequences or not bundle2_sequences:
            ax3.text(0.5, 0.5, 'Insufficient data in overlap region', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('6. Trajectory Bundle Overlap Analysis\nInsufficient data')
            return fig

        bundle1_sequences = np.array(bundle1_sequences)
        bundle2_sequences = np.array(bundle2_sequences)

        # 计算包络范围
        bundle1_upper = np.max(bundle1_sequences, axis=0)
        bundle1_lower = np.min(bundle1_sequences, axis=0)
        bundle2_upper = np.max(bundle2_sequences, axis=0)
        bundle2_lower = np.min(bundle2_sequences, axis=0)

        # 计算重叠区域
        overlap_upper = np.minimum(bundle1_upper, bundle2_upper)
        overlap_lower = np.maximum(bundle1_lower, bundle2_lower)

        # 绘制束范围 (只在重叠区间)
        overlap_steps = np.array(overlap_times)

        ax3.fill_between(overlap_steps, bundle1_lower, bundle1_upper, 
                        alpha=0.4, color='blue', label=f't={t1} Bundle Range')
        ax3.fill_between(overlap_steps, bundle2_lower, bundle2_upper, 
                        alpha=0.4, color='red', label=f't={t2} Bundle Range')

        # 绘制重叠区域
        valid_overlap = overlap_upper >= overlap_lower
        if np.any(valid_overlap):
            overlap_lower_masked = np.where(valid_overlap, overlap_lower, np.nan)
            overlap_upper_masked = np.where(valid_overlap, overlap_upper, np.nan)
            ax3.fill_between(overlap_steps, overlap_lower_masked, overlap_upper_masked,
                            alpha=0.8, color='green', label='Overlap Area')
            
            # 计算重叠百分比
            overlap_volume = np.nansum(overlap_upper_masked - overlap_lower_masked)
            total_volume = np.sum(bundle1_upper - bundle1_lower) + np.sum(bundle2_upper - bundle2_lower)
            overlap_percentage = (overlap_volume / total_volume) * 100 if total_volume > 0 else 0
            
            ax3.text(0.02, 0.98, f'Overlap: {overlap_percentage:.1f}%', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

        ax3.set_xlabel('Planning Steps (Overlap Region)')
        ax3.set_ylabel('Action Value')
        ax3.set_title(f'6. Trajectory Bundle Overlap Analysis\nGreen = Reusable cache region\nOverlap times: {overlap_times}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存到: {save_path}")
        
        return fig


    def analyze_overlap_statistics(self, trajectories_data, t1, t2, action_dim_to_plot=0):
        """分析重叠统计信息 - 修正时间计算"""
        
        if t2 not in trajectories_data:
            print("時間點超出邊界")
            return None
            
        print("\n" + "="*50)
        print("轨迹束重叠分析统计 (修正时间计算)")
        print("="*50)
    
        data1, data2 = trajectories_data[t1], trajectories_data[t2]
        
        # 计算正确的时间范围
        horizon = data1['horizon']
        t1_times = list(range(t1, t1 + horizon))  # [t1, t1+1, t1+2, ...]
        t2_times = list(range(t2, t2 + horizon))  # [t2, t2+1, t2+2, ...]
        
        # 找到重叠区间
        overlap_times = sorted(set(t1_times).intersection(set(t2_times)))
        
        print(f"时间步 {t1} -> {t2}:")
        print(f"  时间范围1: {t1_times}")
        print(f"  时间范围2: {t2_times}")
        print(f"  重叠时间: {overlap_times}")
        print(f"  重叠长度: {len(overlap_times)}")
        
        if not overlap_times:
            print(f"  没有重叠区间")
            print()
            return
        
        # 选择最优轨迹进行对比
        best_idx1 = np.argmax(data1['values'])
        best_idx2 = np.argmax(data2['values'])
        
        traj1 = data1['trajectories'][best_idx1]
        traj2 = data2['trajectories'][best_idx2]
        
        # 提取重叠区间的动作序列
        seq1 = []
        seq2 = []
        
        # 对每个重叠时间，找到在各自轨迹中的位置
        for overlap_time in overlap_times:
            # 计算在各自轨迹中的索引位置
            t_idx_in_traj1 = overlap_time - t1  # 在traj1中的位置
            t_idx_in_traj2 = overlap_time - t2  # 在traj2中的位置
            
            # 检查索引有效性并提取数据
            if (0 <= t_idx_in_traj1 < len(traj1) and
                0 <= t_idx_in_traj2 < len(traj2) and
                isinstance(traj1[t_idx_in_traj1], (list, np.ndarray)) and
                isinstance(traj2[t_idx_in_traj2], (list, np.ndarray)) and
                action_dim_to_plot < len(traj1[t_idx_in_traj1]) and 
                action_dim_to_plot < len(traj2[t_idx_in_traj2])):
                
                seq1.append(traj1[t_idx_in_traj1][action_dim_to_plot])
                seq2.append(traj2[t_idx_in_traj2][action_dim_to_plot])
        
        if len(seq1) == 0 or len(seq2) == 0:
            print(f"  重叠区间数据不足")
            print()
            return 
        
        seq1 = np.array(seq1)
        seq2 = np.array(seq2)
        
        # 计算相似性指标
        mae_distance = np.mean(np.abs(seq1 - seq2))
        mse_distance = np.mean((seq1 - seq2) ** 2)
        
        # 避免除零错误
        norm1 = np.linalg.norm(seq1)
        norm2 = np.linalg.norm(seq2)
        if norm1 > 0 and norm2 > 0:
            cosine_sim = np.dot(seq1, seq2) / (norm1 * norm2)
        else:
            cosine_sim = 0.0
        
        print(f"  重叠区间相似性分析:")
        print(f"    序列1: {seq1}")
        print(f"    序列2: {seq2}")
        print(f"    MAE距离: {mae_distance:.4f}")
        print(f"    MSE距离: {mse_distance:.4f}")
        print(f"    余弦相似性: {cosine_sim:.4f}")
        print(f"    相似性评级: {'高' if cosine_sim > 0.8 else '中' if cosine_sim > 0.6 else '低'}")
        
        # 缓存潜力评估
        cache_potential = "高" if cosine_sim > 0.8 and mae_distance < 0.1 else "中" if cosine_sim > 0.6 else "低"
        print(f"    缓存潜力: {cache_potential}")
        
        # 添加额外的统计信息
        print(f"    数据点数量: {len(seq1)}")
        print(f"    最大差异: {np.max(np.abs(seq1 - seq2)):.4f}")
        print(f"    标准差比率: {np.std(seq1)/np.std(seq2) if np.std(seq2) > 0 else 'N/A'}")
        print()

def run_cartpole_visualization(episode_dir, action_dim_to_plot=0):
    """运行cartpole轨迹束可视化"""
    
    print("开始Cartpole轨迹束可视化...")
    
    # 创建可视化器
    visualizer = CartpoleTrajectoryBundleVisualizer()
    
    # 加载数据
    trajectories_data = visualizer.load_data(episode_dir)
    
    if not trajectories_data:
        print("未找到有效的轨迹数据")
        return

    output_dir = Path(episode_dir) / 'png'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建三个核心图表
    time_steps = sorted(trajectories_data.keys())
    for t1 in time_steps:
        t2 = t1 + 1
        output_path = output_dir / f"trajectory_bundle_analysis_{t1}.png"
        fig = visualizer.create_three_bundle_plots(
            trajectories_data, 
            t1, 
            t2,
            action_dim_to_plot=action_dim_to_plot,
            save_path=output_path
        )
        # 分析重叠统计
        visualizer.analyze_overlap_statistics(trajectories_data, t1, t2, action_dim_to_plot)
    
    
    print(f"\n可视化完成!")
    print(f"输出文件: {output_path}")
    print(f"\n下载命令:")
    print(f"scp username@server:{output_path.absolute()} ./")
    
    return fig, trajectories_data


# 使用示例
if __name__ == "__main__":
    # 设置你的数据路径
    episode_directory = "/media/HDD7/jimmywu/cache-mpc/tdmpc/tests/cartpole-swingup/cartpole-swingup-horizon5/1/episode_1"
    
    if not os.path.exists(episode_directory):
        print(f"Error: The directory '{episode_directory}' does not exist.")
    else:
        print(f"Directory '{episode_directory}' exists. Proceeding with visualization.")
        # run your visualization code
        fig, data = run_cartpole_visualization(episode_directory, action_dim_to_plot=0)
    
    print("\n使用说明:")
    print("1. 修改 episode_directory 为你的实际路径")
    print("2. 运行脚本生成三个轨迹束图")
    print("3. 查看重叠分析统计信息")
    print("4. 下载生成的PNG文件")