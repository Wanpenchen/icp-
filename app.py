# ---------- 后端代码 ----------
import time
from flask import Flask, render_template, request, jsonify
import open3d as o3d
import numpy as np
import os
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def compute_fpfh(pcd, voxel_size):
    # 计算FPFH特征
    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )


def icp(source_path, target_path):
    # 加载点云
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    # 预处理（降采样）
    voxel_size = 0.05
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    # 法线估计
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    source_down.estimate_normals(search_param)
    target_down.estimate_normals(search_param)

    # 粗配准部分
    source_fpfh = compute_fpfh(source_down, voxel_size)
    target_fpfh = compute_fpfh(target_down, voxel_size)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    initial_trans = result_ransac.transformation

    # 执行ICP
    start_time = time.time()
    reg_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, 0.2, initial_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )
    elapsed_time = time.time() - start_time

    # 计算结果指标
    fitness = reg_result.fitness * 100
    inlier_rmse = reg_result.inlier_rmse

    # 应用变换
    source.transform(reg_result.transformation)

    return {
        "matrix": reg_result.transformation.tolist(),
        "source_points": np.asarray(source.points).reshape(-1).tolist(),
        "target_points": np.asarray(target.points).reshape(-1).tolist(),
        "time": f"{elapsed_time * 1000:.2f} ms",
        "accuracy": {
            "fitness": f"{fitness:.2f}%",
            "rmse": f"{inlier_rmse:.4f}"
        }
    }


@app.route('/')
def index():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    try:
        source = request.files['source']
        target = request.files['target']
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source.filename)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target.filename)
        source.save(source_path)
        target.save(target_path)

        result = icp(source_path, target_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
