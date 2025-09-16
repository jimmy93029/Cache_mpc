import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path


class Interactive3DTrajectoryVisualizer:
    """
    交互式3D軌跡可視化器 - 支援旋轉、縮放、點選等交互功能
    """
    
    def __init__(self, trajectories_data):
        self.trajectories_data = trajectories_data
        self.pca = None
        self.scaler = None
        self.pca_trajectories_3d = {}
        self.trajectory_metadata = {}
        
    def fit_pca_3d(self, n_components=3):
        """訓練3D PCA模型"""
        print("準備3D PCA訓練數據...")
        
        all_action_vectors = []
        metadata = []
        
        for time_step, data in self.trajectories_data.items():
            trajectories = data['trajectories']
            
            for traj_idx, trajectory in enumerate(trajectories):
                for t_idx in range(len(trajectory)):
                    action_vector = trajectory[t_idx]
                    if len(action_vector) == data['action_dim']:
                        all_action_vectors.append(action_vector)
                        metadata.append({
                            'time_step': time_step,
                            'traj_idx': traj_idx,
                            't_idx': t_idx,
                            'actual_time': time_step + t_idx
                        })
        
        all_action_vectors = np.array(all_action_vectors)
        self.metadata = metadata
        
        print(f"訓練數據: {len(all_action_vectors)} 個向量，{all_action_vectors.shape[1]} 維")
        
        # 標準化和PCA
        self.scaler = StandardScaler()
        scaled_vectors = self.scaler.fit_transform(all_action_vectors)
        
        self.pca = PCA(n_components=n_components)
        self.pca.fit(scaled_vectors)
        
        explained_variance = self.pca.explained_variance_ratio_
        print(f"解釋變異: PC1={explained_variance[0]:.1%}, PC2={explained_variance[1]:.1%}, PC3={explained_variance[2]:.1%}")
        print(f"累計解釋: {sum(explained_variance):.1%}")
        
        return explained_variance
    
    def transform_trajectories_3d(self):
        """將軌跡轉換到3D PCA空間"""
        if self.pca is None:
            raise ValueError("請先執行 fit_pca_3d()")
        
        print("轉換軌跡到3D空間...")
        
        for time_step, data in self.trajectories_data.items():
            trajectories = data['trajectories']
            
            transformed_trajectories = []
            trajectory_info = []
            
            for traj_idx, trajectory in enumerate(trajectories):
                pca_trajectory = []
                times = []
                
                for t_idx in range(len(trajectory)):
                    action_vector = trajectory[t_idx]
                    if len(action_vector) == data['action_dim']:
                        scaled_vector = self.scaler.transform([action_vector])
                        pca_coords = self.pca.transform(scaled_vector)[0]
                        pca_trajectory.append(pca_coords)
                        times.append(time_step + t_idx)
                
                if pca_trajectory:
                    transformed_trajectories.append({
                        'coords': np.array(pca_trajectory),
                        'times': times,
                        'traj_idx': traj_idx,
                        'time_step': time_step,
                        'value': data['values'][traj_idx] if data['values'] is not None else 0,
                        'score': data['scores'][traj_idx] if data['scores'] is not None else 0
                    })
            
            self.pca_trajectories_3d[time_step] = transformed_trajectories
            
        print(f"轉換完成: {len(self.pca_trajectories_3d)} 個時間步")
 
    def create_time_evolution_3d(self, timesteps_to_plot=None):
        """
        創建時間演化3D圖 - 時間作為Z軸
        """
        if not self.pca_trajectories_3d:
            raise ValueError("請先執行 transform_trajectories_3d()")
        
        if timesteps_to_plot is None:
            timesteps_to_plot = sorted(self.pca_trajectories_3d.keys())
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for i, time_step in enumerate(timesteps_to_plot):
            trajectories = self.pca_trajectories_3d[time_step]
            color = colors[i % len(colors)]
            
            for traj_idx, traj in enumerate(trajectories[:30]):  # 限制數量
                coords = traj['coords']
                times = np.array(traj['times'])
                
                if len(coords) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=times,  # 時間作為Z軸
                        mode='lines+markers',
                        line=dict(color=color, width=2),
                        marker=dict(size=3, color=times, colorscale='Plasma'),
                        name=f'Bundle t={time_step}',
                        legendgroup=f'bundle_{time_step}',
                        showlegend=(traj_idx == 0),
                        hovertemplate=
                        '<b>Bundle t=%{customdata[0]}</b><br>' +
                        'PC1: %{x:.3f}<br>' +
                        'PC2: %{y:.3f}<br>' +
                        'Time: %{z}<br>' +
                        'Value: %{customdata[1]:.3f}' +
                        '<extra></extra>',
                        customdata=np.column_stack([
                            [time_step] * len(coords),
                            [traj['value']] * len(coords)
                        ])
                    ))
        
        fig.update_layout(
            title='3D Time Evolution View<br><sub>X,Y: Action Components • Z: Time</sub>',
            scene=dict(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                zaxis_title='Planning Time',
                camera=dict(eye=dict(x=2, y=2, z=1))
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_similarity_network_3d(self, time_step, similarity_threshold=0.8):
        """
        創建3D相似性網絡圖
        """
        if time_step not in self.pca_trajectories_3d:
            raise ValueError(f"時間步 {time_step} 不存在")
        
        trajectories = self.pca_trajectories_3d[time_step]
        
        # 計算軌跡間相似性
        similarity_matrix = self._compute_trajectory_similarity_3d(trajectories)
        
        fig = go.Figure()
        
        # 添加軌跡節點
        coords_array = np.array([traj['coords'].mean(axis=0) for traj in trajectories])
        
        fig.add_trace(go.Scatter3d(
            x=coords_array[:, 0],
            y=coords_array[:, 1],
            z=coords_array[:, 2],
            mode='markers',
            marker=dict(size=8, color='lightblue', opacity=0.8),
            name='Trajectories',
            hovertemplate='Trajectory %{customdata}<extra></extra>',
            customdata=list(range(len(trajectories)))
        ))
        
        # 添加相似性連線
        for i in range(len(trajectories)):
            for j in range(i+1, len(trajectories)):
                if similarity_matrix[i, j] > similarity_threshold:
                    fig.add_trace(go.Scatter3d(
                        x=[coords_array[i, 0], coords_array[j, 0]],
                        y=[coords_array[i, 1], coords_array[j, 1]],
                        z=[coords_array[i, 2], coords_array[j, 2]],
                        mode='lines',
                        line=dict(
                            color='red',
                            width=similarity_matrix[i, j] * 10
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        fig.update_layout(
            title=f'3D Similarity Network (t={time_step})<br>' + 
                  f'<sub>Threshold: {similarity_threshold}</sub>',
            scene=dict(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                zaxis_title='PCA Component 3'
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    
    def save_interactive_html(self, fig, filename):
        """保存交互式HTML文件"""
        fig.write_html(filename, include_plotlyjs='cdn')
        # print(f"交互式3D圖已保存到: {filename}")
        # print("用瀏覽器打開即可交互操作！")


def load_data(episode_dir):
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


def main():
    """
    使用示例 - 展示如何創建交互式3D軌跡圖
    """
    episode_dir = "/media/HDD7/jimmywu/cache-mpc/tdmpc/tests/humanoid-run/humanoid-run-horizon5/1/episode_5"
    trajectories_data = load_data(episode_dir)

    # 創建可視化器
    visualizer = Interactive3DTrajectoryVisualizer(trajectories_data)
    
    # 訓練3D PCA
    explained_var = visualizer.fit_pca_3d(n_components=3)
    
    # 轉換軌跡
    visualizer.transform_trajectories_3d()
    
    output_dir = Path(episode_dir) / 'html'
    output_dir.mkdir(parents=True, exist_ok=True)
    time_steps = len(trajectories_data)
    
    # Option 1: Create comparison plots for consecutive timesteps
    for t in range(time_steps - 1):
        fig = visualizer.create_time_evolution_3d(timesteps_to_plot=[t, t+1])
        output_path = output_dir / f"3d_trajectories_t{t}_vs_t{t+1}.html"
        visualizer.save_interactive_html(fig, output_path)
        

if __name__ == "__main__":
    main()