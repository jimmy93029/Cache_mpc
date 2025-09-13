import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set font for displaying Chinese characters (if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def explain_trajectory_bundle():
    """Detailed explanation of the trajectory bundle concept"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. What is a single trajectory? (Top-left)
    ax1 = axes[0, 0]
    
    time_steps = np.arange(0, 10, 0.1)
    single_trajectory = np.sin(time_steps) + 0.5 * np.cos(2 * time_steps)
    
    ax1.plot(time_steps, single_trajectory, 'b-', linewidth=3, label='Single Trajectory')
    ax1.scatter(time_steps[::10], single_trajectory[::10], color='red', s=50, zorder=5)
    
    # Annotating time steps
    for i in range(0, len(time_steps), 20):
        ax1.annotate(f't={time_steps[i]:.1f}', 
                    (time_steps[i], single_trajectory[i]), 
                    xytext=(5, 10), textcoords='offset points',
                    fontsize=8, ha='left')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Action Value')
    ax1.set_title('1. Example of a Single Trajectory\nA trajectory = A series of actions over time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Multiple independent trajectories (Top-center)
    ax2 = axes[0, 1]
    
    # Generate multiple similar but different trajectories
    n_trajectories = 5
    trajectories = []
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i in range(n_trajectories):
        # Base trajectory + random variation
        base = np.sin(time_steps + i * 0.2) + 0.5 * np.cos(2 * time_steps + i * 0.1)
        noise = np.random.normal(0, 0.1, len(time_steps))
        trajectory = base + noise
        trajectories.append(trajectory)
        
        ax2.plot(time_steps, trajectory, color=colors[i], alpha=0.7, 
                linewidth=2, label=f'Trajectory {i+1}')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Action Value')
    ax2.set_title('2. Multiple Independent Trajectories\nEach line represents a possible action plan')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Trajectory Bundle - Focus on overlapping areas (Top-right)
    ax3 = axes[0, 2]
    
    # Plot trajectory bundle, focusing on overlaps
    for i, trajectory in enumerate(trajectories):
        ax3.plot(time_steps, trajectory, color=colors[i], alpha=0.6, 
                linewidth=2, label=f'Trajectory {i+1}')
    
    # Calculate and show overlap area (via envelope lines)
    trajectories_array = np.array(trajectories)
    upper_envelope = np.max(trajectories_array, axis=0)
    lower_envelope = np.min(trajectories_array, axis=0)
    mean_trajectory = np.mean(trajectories_array, axis=0)
    
    # Fill the overlapping area
    ax3.fill_between(time_steps, lower_envelope, upper_envelope, 
                    alpha=0.2, color='yellow', label='Trajectory Bundle Range')
    ax3.plot(time_steps, mean_trajectory, 'k--', linewidth=3, 
            label='Mean Trajectory')
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Action Value')
    ax3.set_title('3. Visualizing Trajectory Bundle\nYellow area = "Bundle" range of all trajectories')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Cache MPC Trajectory Bundle Concept (Bottom-left)
    ax4 = axes[1, 0]
    
    # Simulate elite trajectories at different time steps in MPC
    planning_horizon = 10
    time_points = np.arange(planning_horizon)
    
    # Elite trajectories at time step t=0
    elite_trajs_t0 = []
    for i in range(5):  # 5 elite trajectories
        traj = np.sin(time_points + i * 0.1) + np.random.normal(0, 0.05, planning_horizon)
        elite_trajs_t0.append(traj)
        ax4.plot(time_points, traj, 'b-', alpha=0.6, linewidth=2)
    
    ax4.fill_between(time_points, 
                    np.min(elite_trajs_t0, axis=0), 
                    np.max(elite_trajs_t0, axis=0),
                    alpha=0.3, color='blue', label='Trajectory Bundle at t=0')
    
    ax4.set_xlabel('Planning Steps')
    ax4.set_ylabel('Action Value')
    ax4.set_title('4. Cache MPC Trajectory Bundle\nMultiple elite trajectories form a "bundle" at the same time step')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Comparing trajectory bundles at different time steps (Bottom-center)
    ax5 = axes[1, 1]
    
    # Trajectory bundle at time step t=0
    for i, traj in enumerate(elite_trajs_t0):
        if i == 0:  # Add label only for the first trajectory
            ax5.plot(time_points, traj, 'b-', alpha=0.6, linewidth=2, label='t=0 Bundle')
        else:
            ax5.plot(time_points, traj, 'b-', alpha=0.6, linewidth=2)
    
    # Elite trajectories at time step t=1 (slightly offset to show evolution)
    elite_trajs_t1 = []
    for i in range(5):
        traj = np.sin(time_points + i * 0.1 + 0.2) + np.random.normal(0, 0.05, planning_horizon)
        elite_trajs_t1.append(traj)
        if i == 0:
            ax5.plot(time_points, traj, 'r-', alpha=0.6, linewidth=2, label='t=1 Bundle')
        else:
            ax5.plot(time_points, traj, 'r-', alpha=0.6, linewidth=2)
    
    ax5.set_xlabel('Planning Steps')
    ax5.set_ylabel('Action Value')
    ax5.set_title('5. Evolution of Trajectory Bundles\nTrajectory bundles change over time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Trajectory Bundle Overlap Analysis (Bottom-right)
    ax6 = axes[1, 2]
    
    # Show overlap of trajectory bundles at two time steps
    ax6.fill_between(time_points, 
                    np.min(elite_trajs_t0, axis=0), 
                    np.max(elite_trajs_t0, axis=0),
                    alpha=0.4, color='blue', label='t=0 Bundle Range')
    
    ax6.fill_between(time_points, 
                    np.min(elite_trajs_t1, axis=0), 
                    np.max(elite_trajs_t1, axis=0),
                    alpha=0.4, color='red', label='t=1 Bundle Range')
    
    # Calculate overlap area
    overlap_upper = np.minimum(np.max(elite_trajs_t0, axis=0), np.max(elite_trajs_t1, axis=0))
    overlap_lower = np.maximum(np.min(elite_trajs_t0, axis=0), np.min(elite_trajs_t1, axis=0))
    
    # Fill only where overlap occurs
    overlap_mask = overlap_upper > overlap_lower
    if np.any(overlap_mask):
        ax6.fill_between(time_points, 
                        np.where(overlap_mask, overlap_lower, np.nan), 
                        np.where(overlap_mask, overlap_upper, np.nan),
                        alpha=0.8, color='green', label='Overlap Area')
    
    ax6.set_xlabel('Planning Steps')
    ax6.set_ylabel('Action Value')
    ax6.set_title('6. Trajectory Bundle Overlap Analysis\nGreen = Reusable cache region')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Save the figure without displaying it
    if not os.path.exists('png'):
        os.makedirs('png')
    
    fig.savefig('png/trajectory_bundle.png')
    
    plt.close(fig)  # Close the figure to avoid displaying it
    
    return fig

def create_practical_example():
    """Create practical application examples"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Simulate real MPC data
    np.random.seed(42)
    
    # 1. Robot Path Planning (Top-left)
    ax1 = axes[0, 0]
    
    time_horizon = 100
    robot_paths = []
    for i in range(6):  # 6 possible robot paths
        path = np.cumsum(np.random.randn(time_horizon))  # Random walk path
        robot_paths.append(path)
        ax1.plot(path, label=f'Path {i+1}')
    
    ax1.set_title('1. Robot Path Planning\nMultiple paths for planning')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drone Control Example (Top-right)
    ax2 = axes[0, 1]
    
    # Simulate drone flight paths with obstacle avoidance
    drone_paths = []
    for i in range(6):
        path = np.cumsum(np.random.randn(time_horizon))  # Random flight path
        drone_paths.append(path)
        ax2.plot(path, label=f'Flight Path {i+1}')
    
    ax2.set_title('2. Drone Control\nPath planning with avoidance')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Height')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trajectory Bundle Similarity Analysis (Bottom-left)
    ax3 = axes[1, 0]
    
    # Generate two different trajectory bundles and compare
    bundle_1 = np.random.randn(20, 6)  # 6 possible paths
    bundle_2 = np.random.randn(20, 6)
    
    # Plot bundle 1
    ax3.plot(bundle_1, alpha=0.7, color='blue')
    
    # Plot bundle 2
    ax3.plot(bundle_2, alpha=0.7, color='red')
    
    ax3.set_title('3. Trajectory Bundle Comparison\nComparing two bundles')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cache MPC Value (Bottom-right)
    ax4 = axes[1, 1]
    
    # Bar chart for comparison between different caching strategies
    cache_strategies = ['Full Recalculation', 'Partial Cache Reuse', 'Full Cache Reuse']
    computation_times = [12.5, 8.1, 3.7]  # Time to compute in seconds
    
    ax4.bar(cache_strategies, computation_times, color=['red', 'blue', 'green'])
    ax4.set_title('4. MPC Caching Strategies\nComparison of computation times')
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Computation Time (s)')
    
    # Save the figure without displaying it
    if not os.path.exists('png'):
        os.makedirs('png')
    
    fig.savefig('png/practical_example.png')
    
    plt.close(fig)  # Close the figure to avoid displaying it
    
    return fig

# Example Usage: Saving figures
explain_trajectory_bundle()
create_practical_example()
