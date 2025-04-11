"""
 导入Open3D库，它是一个用于处理3D数据的开源库，
 负责点云数据加载、处理、算法执行和可视化
 ICP（Iterative Closest Point）算法实现的核心依赖库
 1. 点云加载与保存: source = o3d.io.read_point_cloud(source_path)  # 从文件加载点云
 2. 数据预处理:
   - source_down = source.voxel_down_sample(voxel_size)  # 体素降采样
    - source_down.estimate_normals()  # 估计点云法线（用于Point-to-Plane ICP）
 3. ICP算法实现: reg_result = o3d.pipelines.registration.registration_icp(...)  # 执行ICP配准
 4. 可视化: o3d.visualization.draw_geometries([source, target])  # 显示配准结果
"""
import open3d as o3d

# 导入NumPy库，用于处理数学运算和矩阵操作，Open3D底层依赖该库
# 例如，生成4x4单位矩阵作为初始变换矩阵: trans_init = np.eye(4)
import numpy as np

# 导入time库，用于代码执行时间的测量和性能监控，帮助优化算法效率
import time

# 从typing模块导入Optional和Tuple
# Optional 用于表示一个变量可以是指定类型的值，也可以是 None
# Tuple 用于表示元组类型，并且可以指定元组中每个元素的类型
from typing import Optional, Tuple

# 导入traceback模块，用于打印完整的错误堆栈信息，方便调试
import traceback


def compute_fpfh(pcd, voxel_size):
    # 计算FPFH特征值
    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )


def icp_registration(
        source_path: str,
        target_path: str,
        voxel_size: float = 0.05,
        max_iter: int = 100,
        radius_normal_multiplier: float = 2.0,
        threshold_multiplier: float = 1.5,
        max_nn: int = 30,
        enable_visualization: bool = True,
        use_robust_kernel: bool = False
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    大部分数据自由，可供用户自主设置
    Args:
        source_path (str): 源点云路径(.pcd/.ply)
        target_path (str): 目标点云路径(.pcd/.ply)
        voxel_size (float): 体素降采样尺寸（默认0.05）
        max_iter (int): ICP最大迭代次数（默认100）
        radius_normal_multiplier (float): 法线估计半径倍数（默认2.0）
        threshold_multiplier (float): ICP距离阈值倍数（默认1.5）
        max_nn (int): 法线估计最大近邻数（默认30）
        enable_visualization (bool): 是否启用可视化（默认True）
        use_robust_kernel (bool): 是否使用鲁棒核函数（减少异常值影响）

    Returns:
        transformation (np.ndarray): 4x4变换矩阵（失败返回None）
        fitness (float): 配准内点比例（失败返回None）
    """
    try:
        '''======================== 1.数据加载 =============================='''
        # 从指定路径读取待配准点云数据
        source = o3d.io.read_point_cloud(source_path)
        # 从指定路径读取目标源点云数据
        target = o3d.io.read_point_cloud(target_path)

        # 检查两个点云文件输入是否为空
        # 后端调用时，实际不运行，在html文件中实现（前端实现）
        if source.is_empty() or target.is_empty():
            # 如果为空，抛出值错误并提示检查文件路径和格式
            raise ValueError("点云数据为空，请检查文件路径和格式")

        '''===================== 2.预处理（带参数灵活性） ====================='''
        # 降采样
        # 减少点云数据量，提高处理效率(voxel_size = 0.05)
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)

        # 法线估计
        # 计算法线估计的半径，通过体素尺寸乘以法线估计半径倍数得到
        radius_normal = voxel_size * radius_normal_multiplier
        # 配置KD树搜索参数，用于法线估计(radius=0.1, max_nn=30)
        search_param = o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=max_nn
        )
        # 对降采样后的点云进行法线估计，用于Point-to-Plane ICP
        source_down.estimate_normals(search_param)
        target_down.estimate_normals(search_param)

        '''========================= 3.粗配准 ============================='''
        # 计算ICP配准的距离阈值，通过体素尺寸乘以阈值倍数得到
        threshold = voxel_size * threshold_multiplier
        # 生成一个4x4的单位矩阵作为初始变换矩阵，也可替换为粗配准结果
        trans_init = np.eye(4)

        # 鲁棒核函数配置
        # 初始化Point-to-Plane的变换估计方法
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        # 如果启用了鲁棒核函数
        if use_robust_kernel:
            # 定义Tukey损失函数，用于减少异常值的影响
            loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
            # 使用带有鲁棒核函数的Point-to-Plane变换估计方法
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)

        ''' ===================== 4.执行ICP + 耗时测量 ====================='''
        # 记录开始时间
        start_time = time.time()
        # 执行ICP配准算法
        reg_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, trans_init,
            estimation_method=estimation_method,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                # 设置最大迭代次数
                max_iteration=max_iter,
                # 设置相对均方根误差的收敛阈值
                relative_rmse=1e-6,
                # 设置相对内点比例的收敛阈值
                relative_fitness=1e-6
            )
        )
        # 计算ICP配准的耗时
        elapsed_time = time.time() - start_time

        '''======================= 5.后处理与评估 ========================='''
        # 如果配准的内点比例小于0.1，打印警告信息
        if reg_result.fitness < 0.1:
            print(f"[警告] 配准质量差 (fitness={reg_result.fitness:.2%})")

        print("\n===== ICP配准结果 =====")
        # 处理不同版本Open3D属性名变化
        try:
            # 尝试打印迭代次数（适用于某些Open3D版本）
            print(f"迭代次数: {reg_result.num_iterations}")
        except AttributeError:
            try:
                # 尝试打印另一种可能的迭代次数属性名（适用于某些Open3D版本）
                print(f"迭代次数: {reg_result.iteration_num}")
            except AttributeError:
                # 如果都失败，打印警告信息，可能是Open3D版本问题
                print("[警告] 无法获取迭代次数，可能是Open3D版本问题")
        # 打印配准的内点比例
        print(f"内点比例: {reg_result.fitness:.2%}")
        # 打印匹配误差（均方根误差）
        print(f"匹配误差 (RMSE): {reg_result.inlier_rmse:.6f}")
        # 打印ICP配准的总耗时
        print(f"总耗时: {elapsed_time:.2f}秒")

        # 将ICP配准得到的变换矩阵应用到原始源点云
        source.transform(reg_result.transformation)

        '''==================== 6.可视化控制 ========================'''
        # 如果启用了可视化
        if enable_visualization:
            # 将待配准点云的颜色设置为红色
            source.paint_uniform_color([1, 0, 0])
            # 将目标源点云的颜色设置为绿色
            target.paint_uniform_color([0, 1, 0])
            # 可视化配准后的源点云和目标点云
            o3d.visualization.draw_geometries(
                [source, target],
                window_name="ICP配准结果",
                width=1024,
                height=768
            )

        # 返回配准得到的变换矩阵和内点比例
        return reg_result.transformation, reg_result.fitness

    except FileNotFoundError as e:
        # 如果文件未找到，打印错误信息和完整错误堆栈
        print(f"[错误] 文件未找到: {str(e)}")
        traceback.print_exc()
        # 返回None和None表示配准失败
        return None, None
    except Exception as e:
        # 如果出现其他异常，打印错误信息和完整错误堆栈
        print(f"[错误] 配准失败: {str(e)}")
        traceback.print_exc()
        # 返回None和None表示配准失败
        return None, None


# 测试调用
if __name__ == "__main__":
    # 可使用此代码输出Open3D版本，例如：0.17.0
    # print(o3d.__version__)

    # 调用icp_registration函数进行点云配准
    trans_matrix, fitness = icp_registration(
        # 源点云路径，需替换为实际路径
        source_path="data/test/Stanford Bunny/data/bun000.ply",
        # 目标点云路径，需替换为实际路径
        target_path="data/test/Stanford Bunny/reconstruction/bun_zipper.ply",
        # 体素降采样尺寸
        voxel_size=0.05
    )
    # 打印配准得到的变换矩阵
    print("变换矩阵:\n", trans_matrix)
    # 打印配准的内点比例
    print("内点比例:", fitness)
