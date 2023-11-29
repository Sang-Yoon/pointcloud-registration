import os
import copy
import numpy as np
import open3d as o3d
import time


class PointCloudRegistration:
    voxel_size = 0.002
    max_correspondence_distance = voxel_size * 1.5
    voxel_sizes = o3d.utility.DoubleVector([voxel_size * 4, voxel_size * 2, voxel_size])
    max_correspondence_distances = o3d.utility.DoubleVector(
        [max_correspondence_distance * 4, max_correspondence_distance * 2, max_correspondence_distance]
    )
    def __init__(self) -> None:

        self.treg = o3d.t.pipelines.registration
        self.criteria = self.treg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)
        self.criteria_list = [
            self.treg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=200),
            self.treg.ICPConvergenceCriteria(relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=100),
            self.treg.ICPConvergenceCriteria(relative_fitness=1e-8, relative_rmse=1e-8, max_iteration=50),
        ]
        self.GLOBAL_REGISTRATION = False
        self.LOCAL_REGISTRATION = True
        self.CUDA = False
        self.RANSAC = False
        self.FGR = False
        self.VANILLA_ICP = False
        self.MULTI_SCALE_ICP = True

    def load_point_clouds(self, pcd_np):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        return pcd

    def remove_hidden_points(self, pcd):
        # All points not visible from that location will be removed
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())) * 2
        camera = np.array([0, 0, diameter]).astype(np.float32)
        radius = diameter * 100  # The radius of the sperical projection

        pcd = o3d.geometry.PointCloud(pcd)
        _, pt_map = pcd.hidden_point_removal(camera, radius)
        pcd = pcd.select_by_index(pt_map)
        return pcd

    def draw_registration_result(self, src, dst, result, window_name="After registration"):
        src_temp = copy.deepcopy(src)
        dst_temp = copy.deepcopy(dst)

        src_temp.paint_uniform_color([1, 0.706, 0])
        dst_temp.paint_uniform_color([0, 0.651, 0.929])

        transformation = result if result is not None else np.identity(4)
        transformation = transformation.cpu().numpy() if isinstance(transformation, o3d.core.Tensor) else transformation

        src_temp.transform(transformation)

        src_temp = src_temp.to_legacy() if isinstance(src_temp, o3d.t.geometry.PointCloud) else src_temp
        dst_temp = dst_temp.to_legacy() if isinstance(dst_temp, o3d.t.geometry.PointCloud) else dst_temp

        # # lineset of correspondences
        # lines = o3d.geometry.LineSet()
        # if isinstance(result, o3d.t.pipelines.registration.RegistrationResult):
        #     corr = result.correspondences_.cpu().numpy()
        # else:
        #     corr = np.asarray(result.correspondence_set)
        # corr = [(int(i), corr[i]) for i in range(len(corr))]
        # print(corr)

        # corr_lines = lines.create_from_point_cloud_correspondences(src, dst, corr)
        # corr_lines.paint_uniform_color([1, 0, 0])
        # corr_lines_temp = lines.create_from_point_cloud_correspondences(src_temp, dst_temp, corr)
        # corr_lines_temp.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([src, dst, corr_lines], window_name="Before registration", width=720, height=540)
        # o3d.visualization.draw_geometries([src_temp, dst_temp, corr_lines_temp], window_name, width=720, height=540)

        o3d.visualization.draw_geometries([src_temp, dst_temp], window_name, width=720, height=540)

    def preprocess_point_cloud(self, pcd, voxel_size):
        # print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        # print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return pcd_down, pcd_fpfh

    def execute_fast_global_registration(self, src_down, dst_down, src_fpfh, dst_fpfh, voxel_size):
        start_time = time.time()
        max_correspondence_distance = voxel_size * 3.0
        # print(":: Apply fast global registration with distance threshold %.3f" % max_correspondence_distance)
        result_global_refinement = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            src_down,
            dst_down,
            src_fpfh,
            dst_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=max_correspondence_distance
            ),
        )
        running_time = time.time() - start_time
        return result_global_refinement, running_time

    def execute_global_registration(self, src_down, dst_down, src_fpfh, dst_fpfh, voxel_size):
        start_time = time.time()
        max_correspondence_distance = voxel_size * 3.0
        # print(":: RANSAC registration on downsampled point clouds.")
        # print("   Since the downsampling voxel size is %.3f," % voxel_size)
        # print("   we use a liberal distance threshold %.3f." % max_correspondence_distance)
        result_global_refinement = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            dst_down,
            src_fpfh,
            dst_fpfh,
            True,
            max_correspondence_distance,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        running_time = time.time() - start_time
        return result_global_refinement, running_time

    def execute_ICP_registration(self, src, dst, voxel_size, init_src_temp_to_target):
        start_time = time.time()
        max_correspondence_distance = voxel_size * 3.0
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=10000
        )
        # print(":: Point-to-plane ICP registration is applied on original point")
        # print("   clouds to refine the alignment. This time we use a strict")
        # print("   distance threshold %.3f." % max_correspondence_distance)
        callback_after_iteration = lambda loss_log_map: print(
            "Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
                loss_log_map["iteration_index"].item(),
                loss_log_map["scale_index"].item(),
                loss_log_map["scale_iteration_index"].item(),
                loss_log_map["fitness"].item(),
                loss_log_map["inlier_rmse"].item(),
            )
        )
        result_local_refinement = o3d.pipelines.registration.registration_icp(
            src,
            dst,
            max_correspondence_distance,
            init_src_temp_to_target,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria,
        )
        running_time = time.time() - start_time
        return result_local_refinement, running_time

    def execute_ICP_registration_CUDA(self, src, dst, voxel_size, init_src_temp_to_target):
        start_time = time.time()
        max_correspondence_distance = voxel_size * 3.0
        src.estimate_normals()
        dst.estimate_normals()
        estimation = self.treg.TransformationEstimationPointToPlane()
        criteria = self.treg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=10000)
        callback_after_iteration = lambda loss_log_map: print(
            "Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
                loss_log_map["iteration_index"].item(),
                loss_log_map["scale_index"].item(),
                loss_log_map["scale_iteration_index"].item(),
                loss_log_map["fitness"].item(),
                loss_log_map["inlier_rmse"].item(),
            )
        )
        result_local_refinement = self.treg.icp(
            src,
            dst,
            max_correspondence_distance,
            init_src_temp_to_target,
            estimation,
            criteria,
            voxel_size,
            # callback_after_iteration,
        )
        running_time = time.time() - start_time
        return result_local_refinement, running_time

    def execute_multi_scale_ICP_registration(self, src, dst, voxel_sizes, init_src_temp_to_target=np.identity(4)):
        start_time = time.time()
        src = o3d.t.geometry.PointCloud.from_legacy(src)
        dst = o3d.t.geometry.PointCloud.from_legacy(dst)
        if self.CUDA:
            src, dst = src.cuda(0), dst.cuda(0)

        voxel_sizes = self.voxel_sizes
        max_correspondence_distances = self.max_correspondence_distances
        criteria_list = self.criteria_list

        init_src_temp_to_target = o3d.core.Tensor(init_src_temp_to_target)

        estimation = self.treg.TransformationEstimationPointToPlane()
        src.estimate_normals()
        dst.estimate_normals()

        callback_after_iteration = lambda loss_log_map: print(
            "Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
                loss_log_map["iteration_index"].item(),
                loss_log_map["scale_index"].item(),
                loss_log_map["scale_iteration_index"].item(),
                loss_log_map["fitness"].item(),
                loss_log_map["inlier_rmse"].item(),
            )
        )

        result_local_refinement = self.treg.multi_scale_icp(
            src,
            dst,
            voxel_sizes,
            criteria_list,
            max_correspondence_distances,
            init_src_temp_to_target,
            estimation,
            # callback_after_iteration,
        )
        running_time = time.time() - start_time
        return result_local_refinement, running_time


def main():

    pcd_reg = PointCloudRegistration()

    pcd_path = os.path.join(os.path.dirname(__file__), "pvn3d/datasets/ycb/YCB_Video_Dataset/models/102_valve_model/points.xyz")
    pcd_np = np.loadtxt(pcd_path)
    dst = pcd_reg.load_point_clouds(pcd_np)

    roll, pitch, yaw = np.pi / 4, 0, 0
    euler_angle = np.array([roll, pitch, yaw])
    R = o3d.geometry.get_rotation_matrix_from_xyz(euler_angle)
    src = copy.deepcopy(dst)
    src.rotate(R)

    pcd_reg.remove_hidden_points(src)

    voxel_size = pcd_reg.voxel_size
    voxel_sizes = pcd_reg.voxel_sizes
    src_down, src_down_fpfh = pcd_reg.preprocess_point_cloud(src, voxel_size)
    dst_down, dst_down_fpfh = pcd_reg.preprocess_point_cloud(dst, voxel_size)

    ##############################
    # Global registration
    ##############################
    if pcd_reg.GLOBAL_REGISTRATION:
        start_global_registration = time.time()
        if pcd_reg.RANSAC:
            result_ransac, _ = pcd_reg.execute_global_registration(src_down, dst_down, src_down_fpfh, dst_down_fpfh, voxel_size)
            print("Global registration took %.3f sec.\n" % (time.time() - start_global_registration))
            print(result_ransac)
            # pcd_reg.draw_registration_result(src_down, dst_down, result_ransac)
        elif pcd_reg.FGR:
            result_fast, _ = pcd_reg.execute_fast_global_registration(src_down, dst_down, src_down_fpfh, dst_down_fpfh, voxel_size)
            print("Fast global registration took %.3f sec.\n" % (time.time() - start_global_registration))
            print(result_fast)
            # pcd_reg.draw_registration_result(src_down, dst_down, result_fast)

        result_global_refinement = result_ransac if pcd_reg.RANSAC else result_fast

    init_src_temp_to_target = result_global_refinement if pcd_reg.GLOBAL_REGISTRATION else np.identity(4)

    ##############################
    # Local registration
    ##############################
    if pcd_reg.LOCAL_REGISTRATION:
        start_local_registration = time.time()
        if pcd_reg.VANILLA_ICP:
            if pcd_reg.CUDA:
                device = o3d.core.Device("cuda:0")
                dtype = o3d.core.float32

                pcd_src_cuda = o3d.t.geometry.PointCloud.from_legacy(src, device=device, dtype=dtype)
                pcd_dst_cuda = o3d.t.geometry.PointCloud.from_legacy(dst, device=device, dtype=dtype)

                result_local_refinement = pcd_reg.execute_ICP_registration_CUDA(
                    pcd_src_cuda, pcd_dst_cuda, voxel_size, init_src_temp_to_target
                )
                print("CUDA Accelerated ICP registration took %.3f sec.\n" % (time.time() - start_local_registration))
                pcd_reg.draw_registration_result(
                    pcd_src_cuda.to_legacy(), pcd_dst_cuda.to_legacy(), result_local_refinement.transformation
                )
            else:
                result_local_refinement = pcd_reg.execute_ICP_registration(src, dst, voxel_size, init_src_temp_to_target)
                print("Local registration took %.3f sec.\n" % (time.time() - start_local_registration))
                print(result_local_refinement)
                pcd_reg.draw_registration_result(src, dst, result_local_refinement.transformation)
        elif pcd_reg.MULTI_SCALE_ICP:
            result_local_refinement, _ = pcd_reg.execute_multi_scale_ICP_registration(
                src, dst, voxel_sizes, init_src_temp_to_target
            )
            # print("result_local_refinement", result_local_refinement)
            print("Multi-scale ICP registration took %.3f sec.\n" % (time.time() - start_local_registration))
            # pcd_reg.draw_registration_result(src, dst, result_local_refinement.transformation)


if __name__ == "__main__":
    main()
