import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import pts2ply

def visualize_point_cloud_matplotlib(points, colors=None, title="Point Cloud Visualization"):
    """
    Visualize point cloud using matplotlib
    
    Args:
        points (np.ndarray): Nx3 array of points
        colors (np.ndarray, optional): Nx3 array of RGB colors (0-255)
        title (str): Title for the plot
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # If colors are provided, normalize them to [0,1] range
    if colors is not None:
        colors = colors.astype(float) / 255.0
    
    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=colors if colors is not None else None,
              s=1, # point size
              alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([points[:,0].max()-points[:,0].min(),
                         points[:,1].max()-points[:,1].min(),
                         points[:,2].max()-points[:,2].min()]).max() / 2.0
    
    mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
    mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
    mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud_o3d(points, colors=None, window_name="Point Cloud Viewer", initial_zoom=0.3):
    """
    Visualize point cloud data using Open3D with unlimited zoom capability
    """
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # Normalize colors to [0,1] if they're in [0,255]
        if colors.max() > 1:
            colors = colors.astype(float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    def custom_draw_geometry_with_key_callback(pcd):
        # Custom visualization function with enhanced controls
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=window_name, width=1280, height=720)
        
        # Add point cloud
        vis.add_geometry(pcd)
        
        # Get render option and set initial parameters
        render_option = vis.get_render_option()
        render_option.point_size = 5.0
        render_option.background_color = np.array([0, 0, 0])
        
        # Enable point size adjustment
        def increase_points(vis):
            render_option = vis.get_render_option()
            render_option.point_size = min(render_option.point_size * 1.2, 20.0)
            print(f"Point size increased to {render_option.point_size:.1f}")
            return False

        def decrease_points(vis):
            render_option = vis.get_render_option()
            render_option.point_size = max(render_option.point_size / 1.2, 1.0)
            print(f"Point size decreased to {render_option.point_size:.1f}")
            return False
        
        # Register key callbacks for point size
        vis.register_key_callback(ord('.'), increase_points)  # . key to increase point size
        vis.register_key_callback(ord(','), decrease_points)  # , key to decrease point size
        
        # Get view control
        view_control = vis.get_view_control()
        
        # Calculate initial view based on geometry
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_center = bbox.get_center()
        bbox_extent = bbox.get_extent()

        # Set initial view parameters
        front = [0, 0, -1]  # Looking towards negative z
        up = [0, -1, 0]     # Y axis points up
        
        view_control.set_front(front)
        view_control.set_up(up)
        view_control.set_lookat(bbox_center)
        view_control.set_zoom(0.7 / initial_zoom)
        
        # Enable unlimited zoom
        view_control.change_field_of_view(step=-90)  # Reduces FOV to allow closer zoom
        
        print("\nEnhanced Visualization Controls:")
        print("  Left click + drag: Rotate")
        print("  Shift + left click + drag: Pan")
        print("  Mouse wheel: Zoom in/out (unlimited)")
        print("  , key: Decrease point size")
        print("  . key: Increase point size")
        print("  R key: Reset view")
        print("  Q key or ESC: Exit viewer")
        
        vis.run()
        vis.destroy_window()

    custom_draw_geometry_with_key_callback(pcd)

def modify_sfm_class(SFM):
    """
    Modify the SFM class to add enhanced visualization capabilities
    """
    original_to_ply = SFM.ToPly
    
    def new_to_ply(self, filename):
        """
        Extended ToPly method that includes visualization with enhanced settings
        """
        # Get colors using the original _GetColors method
        colors = self._GetColors()
        
        # Remove any invalid points (those with NaN or infinite values)
        valid_mask = np.all(np.isfinite(self.point_cloud), axis=1)
        clean_points = self.point_cloud[valid_mask]
        clean_colors = colors[valid_mask]
        
        # Filter out outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(clean_points)
        pcd.colors = o3d.utility.Vector3dVector(clean_colors.astype(float) / 255.0)
        
        # Statistical outlier removal with more lenient parameters
        try:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
        except Exception as e:
            print("Warning: Could not perform statistical outlier removal:", str(e))
        
        # Convert back to numpy arrays
        filtered_points = np.asarray(pcd.points)
        filtered_colors = np.asarray(pcd.colors) * 255
        
        print(f"\nDisplaying point cloud with {len(filtered_points)} points...")
        print("Use mouse wheel to zoom in/out (unlimited zoom)")
        print("Use ,/. keys to adjust point size")
        
        # Show Open3D visualization with enhanced controls
        visualize_point_cloud_o3d(filtered_points, filtered_colors)
        
        # Save the filtered point cloud
        pts2ply(filtered_points, filtered_colors, filename)
        print(f"\nPoint cloud saved to: {filename}")
    
    # Replace the ToPly method
    SFM.ToPly = new_to_ply
    return SFM