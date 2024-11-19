import numpy as np
from plyfile import PlyData
import open3d as o3d
import matplotlib.pyplot as plt

def read_ply_basic(file_path):
    """
    使用plyfile库读取PLY文件的基本信息
    返回点云数据和属性信息
    """
    ply_data = PlyData.read(file_path)
    vertex = ply_data['vertex']
    
    # 获取点坐标
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # 获取可能存在的其他属性（如颜色、法向量等）
    properties = {}
    for prop in vertex.properties:
        if prop.name not in ['x', 'y', 'z']:
            properties[prop.name] = vertex[prop.name]
            
    return points, properties

def visualize_ply_matplotlib(points, properties=None):
    """
    使用Matplotlib进行简单的3D可视化
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 如果存在颜色属性，使用颜色进行绘制
    if properties and all(x in properties for x in ['red', 'green', 'blue']):
        colors = np.vstack([properties['red'], properties['green'], properties['blue']]).T / 255.0
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def visualize_ply_open3d(file_path):
    """
    使用Open3D进行交互式可视化
    提供更好的3D交互体验
    """
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])

def analyze_ply(file_path):
    """
    分析PLY文件的基本信息
    """
    ply_data = PlyData.read(file_path)
    
    # 基本信息
    print(f"文件格式: {'binary' if ply_data.text is False else 'ascii'}")
    print(f"总点数: {len(ply_data['vertex'])}")
    
    # 查看可用属性
    print("\n可用属性:")
    for prop in ply_data['vertex'].properties:
        print(f"- {prop.name}: {prop.dtype}")
    
    # 计算点云的基本统计信息
    vertex = ply_data['vertex']
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    print("\n空间范围:")
    print(f"X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

def main():
    """
    使用示例
    """
    file_path = "./results/Herz-Jesus-P25/point-clouds/cloud_3_view.ply"  # 替换为你的PLY文件路径
    
    # 1. 分析文件基本信息
    print("===== PLY文件基本信息 =====")
    analyze_ply(file_path)
    
    # 2. 读取数据
    points, properties = read_ply_basic(file_path)
    
    # 3. 使用Matplotlib可视化
    print("\n使用Matplotlib显示点云...")
    visualize_ply_matplotlib(points, properties)
    
    # 4. 使用Open3D交互式可视化
    print("\n使用Open3D显示点云...")
    visualize_ply_open3d(file_path)

if __name__ == "__main__":
    main()