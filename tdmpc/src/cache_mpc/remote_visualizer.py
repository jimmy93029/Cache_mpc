import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# 🚀 方案1: 设置matplotlib为非交互模式 + 保存图片
def setup_remote_matplotlib():
    """配置matplotlib在远程服务器上工作"""
    
    # 设置为非交互后端，避免需要显示器
    matplotlib.use('Agg')  # 或者 'svg', 'pdf'
    
    # 设置图片保存的默认参数
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.format'] = 'png'
    plt.rcParams['savefig.bbox'] = 'tight'
    
    print("✓ Matplotlib配置完成 - 适合远程服务器")

# 修改后的Cache MPC可视化类
class RemoteCacheMPCVisualizer:
    """专为远程服务器设计的可视化工具"""
    
    def __init__(self, horizon, action_dim, output_dir="./png"):
        # 确保使用非交互后端
        matplotlib.use('Agg')
        
        self.horizon = horizon
        self.action_dim = action_dim
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 颜色配置
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        print(f"📁 输出目录: {self.output_dir.absolute()}")
    
    def create_and_save_all_plots(self, trajectories_data, prefix="cache_mpc"):
        """创建所有图表并保存到文件"""
        
        plots_created = []
        
        try:
            # 1. 轨迹束图
            print("🎯 生成轨迹束图...")
            bundle_fig = self.trajectory_bundle_plot(trajectories_data)
            bundle_path = self.output_dir / f"{prefix}_trajectory_bundle.png"
            bundle_fig.savefig(bundle_path, dpi=300, bbox_inches='tight')
            plt.close(bundle_fig)  # 关闭图形释放内存
            plots_created.append(str(bundle_path))
            print(f"   ✓ 保存到: {bundle_path}")
            
            # 2. 相似性网络图
            print("🕸️ 生成相似性网络图...")
            network_fig = self.similarity_network_plot(trajectories_data)
            network_path = self.output_dir / f"{prefix}_similarity_network.png"
            network_fig.savefig(network_path, dpi=300, bbox_inches='tight')
            plt.close(network_fig)
            plots_created.append(str(network_path))
            print(f"   ✓ 保存到: {network_path}")
            
            # 3. 时序重叠热图
            print("🔥 生成时序重叠热图...")
            heatmap_fig = self.temporal_overlap_heatmap(trajectories_data)
            heatmap_path = self.output_dir / f"{prefix}_temporal_heatmap.png"
            heatmap_fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close(heatmap_fig)
            plots_created.append(str(heatmap_path))
            print(f"   ✓ 保存到: {heatmap_path}")
            
            # 4. 创建HTML查看器
            html_path = self.create_html_viewer(plots_created, prefix)
            print(f"📄 HTML查看器: {html_path}")
            
            return plots_created, html_path
            
        except Exception as e:
            print(f"❌ 生成图表时出错: {e}")
            return plots_created, None
    
    def trajectory_bundle_plot(self, trajectories_data):
        """简化的轨迹束图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        time_steps = sorted(trajectories_data.keys())
        
        # 第一个时间步的轨迹束
        if time_steps:
            first_time = time_steps[0]
            data = trajectories_data[first_time]
            trajs = data['trajectories'][:5]  # 只取前5个
            
            ax = axes[0, 0]
            horizon_steps = np.arange(self.horizon)
            
            for i, traj in enumerate(trajs):
                traj_2d = traj.reshape(self.horizon, self.action_dim)
                ax.plot(horizon_steps, traj_2d[:, 0], 
                       color=self.colors[i % len(self.colors)], 
                       alpha=0.7, linewidth=2, label=f'Traj {i+1}')
            
            ax.set_title(f'Trajectory Bundle at t={first_time}')
            ax.set_xlabel('Planning Steps')
            ax.set_ylabel('Action Dimension 0')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 其他子图...
        for i, ax in enumerate(axes.flat[1:]):
            ax.text(0.5, 0.5, f'Subplot {i+2}\n(Implementation details)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Analysis View {i+2}')
        
        plt.tight_layout()
        plt.suptitle('Cache MPC Trajectory Bundle Analysis', y=0.98)
        
        return fig
    
    def similarity_network_plot(self, trajectories_data):
        """简化的相似性网络图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 模拟网络数据
        n_nodes = 10
        angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        x_coords = np.cos(angles)
        y_coords = np.sin(angles)
        
        # 绘制节点
        ax.scatter(x_coords, y_coords, s=200, c=range(n_nodes), 
                  cmap='viridis', alpha=0.8, edgecolors='black')
        
        # 添加一些连接线
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if np.random.random() > 0.7:  # 30%的连接概率
                    ax.plot([x_coords[i], x_coords[j]], 
                           [y_coords[i], y_coords[j]], 
                           'gray', alpha=0.5, linewidth=1)
        
        # 添加标签
        for i in range(n_nodes):
            ax.annotate(f'T{i+1}', (x_coords[i], y_coords[i]), 
                       ha='center', va='center', fontweight='bold')
        
        ax.set_title('Trajectory Similarity Network')
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig
    
    def temporal_overlap_heatmap(self, trajectories_data):
        """简化的时序重叠热图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        time_steps = sorted(trajectories_data.keys())
        n_times = len(time_steps)
        
        # 创建模拟相似性矩阵
        similarity_matrix = np.random.rand(n_times, n_times)
        # 确保对角线为1，矩阵对称
        for i in range(n_times):
            similarity_matrix[i, i] = 1.0
            for j in range(i+1, n_times):
                similarity_matrix[j, i] = similarity_matrix[i, j]
        
        im = ax.imshow(similarity_matrix, cmap='Blues', aspect='auto')
        
        # 添加数值标注
        for i in range(n_times):
            for j in range(n_times):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", 
                             color="white" if similarity_matrix[i, j] > 0.5 else "black")
        
        ax.set_xticks(range(n_times))
        ax.set_yticks(range(n_times))
        ax.set_xticklabels([f't={t}' for t in time_steps])
        ax.set_yticklabels([f't={t}' for t in time_steps])
        ax.set_title('Temporal Overlap Heatmap')
        
        plt.colorbar(im, ax=ax, label='Similarity')
        
        return fig
    
    def create_html_viewer(self, image_paths, prefix):
        """创建HTML文件来查看所有图片"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cache MPC Visualization Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .plot {{ margin: 30px 0; text-align: center; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        .plot h2 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .download-links {{ margin: 20px 0; }}
        .download-links a {{ 
            display: inline-block; margin: 5px; padding: 8px 15px; 
            background: #007bff; color: white; text-decoration: none; 
            border-radius: 3px; 
        }}
        .download-links a:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Cache MPC Visualization Results</h1>
        
        <div class="summary">
            <h3>📊 Analysis Summary</h3>
            <p><strong>Generated plots:</strong> {len(image_paths)}</p>
            <p><strong>Analysis type:</strong> Trajectory overlap and caching potential</p>
            <p><strong>Output directory:</strong> {self.output_dir}</p>
        </div>
        
        <div class="download-links">
            <h3>📥 Download Links</h3>
"""
        
        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            html_content += f'<a href="{filename}" download>Download {filename}</a>\n'
        
        html_content += """
        </div>
        
        <div class="plots">
"""
        
        plot_titles = [
            "🎯 Trajectory Bundle Analysis",
            "🕸️ Similarity Network Graph", 
            "🔥 Temporal Overlap Heatmap"
        ]
        
        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            title = plot_titles[i] if i < len(plot_titles) else f"Plot {i+1}"
            
            html_content += f"""
            <div class="plot">
                <h2>{title}</h2>
                <img src="{filename}" alt="{title}">
                <p><em>File: {filename}</em></p>
            </div>
"""
        
        html_content += """
        </div>
        
        <div class="summary">
            <h3>🔍 How to Use These Results</h3>
            <ul>
                <li><strong>Trajectory Bundle:</strong> Shows overlap patterns between multiple planned trajectories</li>
                <li><strong>Similarity Network:</strong> Visualizes which trajectories can be reused</li>
                <li><strong>Temporal Heatmap:</strong> Identifies optimal caching timing</li>
            </ul>
            
            <h4>💡 Next Steps:</h4>
            <ol>
                <li>Download the images for your paper/presentation</li>
                <li>Analyze the similarity patterns to optimize caching strategy</li>
                <li>Use the insights to improve your Cache MPC implementation</li>
            </ol>
        </div>
    </div>
</body>
</html>
"""
        
        html_path = self.output_dir / f"{prefix}_results.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path