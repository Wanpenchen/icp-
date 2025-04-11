import os
import pandas as pd
import numpy as np
import open3d as o3d
#  有bug，还没定位到具体位置
def read_csv_with_encoding(csv_path, sep='\t'):
    """
    读取CSV文件，自动检测编码
    :param csv_path: CSV文件路径
    :param sep: 分隔符，默认为制表符
    :return: 读取后的DataFrame
    """
    encodings = ['utf-8-sig', 'gbk', 'latin1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, sep=sep, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError("无法解析CSV文件编码")
    return df

def clean_data(df, x_col, y_col, z_col):
    """
    清理数据，包括删除空值、替换非法字符和转换为数值类型
    :param df: 输入的DataFrame
    :param x_col: X坐标列名
    :param y_col: Y坐标列名
    :param z_col: Z坐标列名
    :return: 清理后的DataFrame
    """
    # 删除空值
    df = df.dropna(subset=[x_col, y_col, z_col])
    # 替换非法字符（允许科学计数法中的e/E）
    pattern = r'[^0-9\.\-eE]'
    df[[x_col, y_col, z_col]] = df[[x_col, y_col, z_col]].replace(pattern, '', regex=True)
    # 转换为数值并删除无效行
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df[z_col] = pd.to_numeric(df[z_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col, z_col])
    return df

def save_pointcloud(pcd, save_ply, csv_path, verbose):
    """
    保存点云数据到PLY文件
    :param pcd: 点云对象
    :param save_ply: 保存路径
    :param csv_path: 输入的CSV文件路径
    :param verbose: 是否打印详细信息
    """
    if save_ply:
        if os.path.isdir(save_ply):
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            save_ply = os.path.join(save_ply, f"{base_name}.ply")
        else:
            parent_dir = os.path.dirname(save_ply)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        try:
            o3d.io.write_point_cloud(save_ply, pcd)
            if verbose:
                print(f"保存文件至: {save_ply}")
        except Exception as e:
            print(f"❌ 保存文件失败: {str(e)}")

def print_pointcloud_info(coordinates, verbose):
    """
    打印点云信息，包括点数和坐标范围
    :param coordinates: 点云坐标
    :param verbose: 是否打印详细信息
    """
    if verbose:
        print(f"✅ 成功转换点云: {len(coordinates)}点")
        print(f"坐标范围 X: [{coordinates[:, 0].min():.2f}, {coordinates[:, 0].max():.2f}]")
        print(f"坐标范围 Y: [{coordinates[:, 1].min():.2f}, {coordinates[:, 1].max():.2f}]")
        print(f"坐标范围 Z: [{coordinates[:, 2].min():.2f}, {coordinates[:, 2].max():.2f}]")

def csv_to_pointcloud(csv_path,
                      x_col='x', y_col='y', z_col='z',
                      sep='\t',
                      verbose=True,
                      save_ply=None):
    """
    将CSV文件转换为Open3D点云对象，并支持同名.ply保存
    """
    try:
        # 检查输入的CSV文件是否存在
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV文件 {csv_path} 不存在")

        # 1. 读取CSV（自动检测编码）
        df = read_csv_with_encoding(csv_path, sep)

        # 2. 清理数据
        df = clean_data(df, x_col, y_col, z_col)

        # 3. 转换为点云（使用float64类型）
        coordinates = df[[x_col, y_col, z_col]].values.astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coordinates)  # 注意这里应该使用Vector3dVector

        # 4. 保存文件
        save_pointcloud(pcd, save_ply, csv_path, verbose)

        # 5. 打印信息
        print_pointcloud_info(coordinates, verbose)

        return pcd

    except FileNotFoundError as e:
        print(f"❌ 转换失败: {str(e)}")
        raise
    except ValueError as e:
        print(f"❌ 转换失败: {str(e)}")
        raise
    except Exception as e:
        print(f"❌ 转换失败: 未知错误 {str(e)}")
        raise

# 调试调用
if __name__ == "__main__":
    pcd = csv_to_pointcloud("data/test/transfrom/Hokuyo_0.csv", save_ply="data/test/")