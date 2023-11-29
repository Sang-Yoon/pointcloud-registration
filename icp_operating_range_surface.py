"""
This script is used to find the operating range of ICP algorithm about given point cloud.
"""


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pickle
import pandas as pd
import seaborn as sns
import time
import argparse
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from pcd_registration_o3d import PointCloudRegistration


pcd_reg = PointCloudRegistration()

parser = argparse.ArgumentParser()
parser.add_argument("--global_registration", action="store_true")
parser.add_argument("--local_registration", action="store_true")
parser.add_argument("--ransac", action="store_true")
parser.add_argument("--fgr", action="store_true")
parser.add_argument("--vanilla", action="store_true")
parser.add_argument("--multi_scale", action="store_true")
parser.add_argument("--total_figures", action="store_true")
args = parser.parse_args()

if any(vars(args).values()):
    print("Use arguments of args.")
    GLOBAL_REGISTRATION = args.global_registration
    LOCAL_REGISTRATION = args.local_registration
    RANSAC = args.ransac
    FGR = args.fgr
    VANILLA_ICP = args.vanilla
    MULTI_SCALE_ICP = args.multi_scale
else:
    print("Use default arguments.")
    GLOBAL_REGISTRATION = pcd_reg.GLOBAL_REGISTRATION
    LOCAL_REGISTRATION = pcd_reg.LOCAL_REGISTRATION
    RANSAC = pcd_reg.RANSAC
    FGR = pcd_reg.FGR
    VANILLA_ICP = pcd_reg.VANILLA_ICP
    MULTI_SCALE_ICP = pcd_reg.MULTI_SCALE_ICP

print("===============================")
print("GLOBAL_REGISTRATION:", GLOBAL_REGISTRATION)
print("LOCAL_REGISTRATION:", LOCAL_REGISTRATION)
print("RANSAC:", RANSAC)
print("FGR:", FGR)
print("VANILLA_ICP:", VANILLA_ICP)
print("MULTI_SCALE_ICP:", MULTI_SCALE_ICP)
print("===============================")


def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class ICPOperatingRangePlotter:
    def __init__(self):
        self.interval = 5
        self.num_interval = int(180 / self.interval)
        self.figsize = (15, 8)
        self.label_font_size = 15
        self.tick_font_size = 13
        self.marker_font_size = 13
        # self.legend_font_size = 13
        self.title_font_size = 20

    def get_pkl_data(self, pickle_data):
        convergence, relative_rotation_angle, ratio_diameter_to_distance, running_time = zip(*pickle_data)
        relative_rotation_angle = np.rad2deg(relative_rotation_angle)
        running_time = [time for time, conv in zip(running_time, convergence)]
        return convergence, relative_rotation_angle, ratio_diameter_to_distance, running_time

    def plot_rotation_angle_distribution(self, pickle_data_list: list, save_path=None):
        if len(pickle_data_list) == 1:
            pickle_data = pickle_data_list[0]
        else:
            config_list = [os.path.basename(path).split("_")[-1].split(",")[:6] for path in pickle_data_list]
            for config in config_list:
                global_registration = config[0][1:]
                local_registration = config[1][1:]
                ransac = config[2][1:]
                fgr = config[3][1:]
                vanilla_icp = config[4][1:]
                multi_scale_icp = config[5][1:]
                if (global_registration == "False") and (local_registration == "True") and multi_scale_icp == "True":
                    icp_pickle_data = pickle.load(open(pickle_data_list[config_list.index(config)], "rb"))
                elif global_registration == "True" and ransac == "True":
                    ransac_pickle_data = pickle.load(open(pickle_data_list[config_list.index(config)], "rb"))
                elif global_registration == "True" and fgr == "True":
                    fgr_pickle_data = pickle.load(open(pickle_data_list[config_list.index(config)], "rb"))

        if len(pickle_data_list) == 1:
            convergence, relative_rotation_angle, ratio_dd, _ = self.get_pkl_data(pickle_data)
        else:
            icp_convergence, icp_relative_rotation_angle, icp_ratio_dd, _ = self.get_pkl_data(icp_pickle_data)
            ransac_convergence, ransac_relative_rotation_angle, ransac_ratio_dd, _ = self.get_pkl_data(ransac_pickle_data)
            fgr_convergence, fgr_relative_rotation_angle, fgr_ratio_dd, _ = self.get_pkl_data(fgr_pickle_data)

        def get_success_rate_by_interval(convergence, relative_rotation_angle):
            success_by_interval = np.zeros(self.num_interval)
            fail_by_interval = np.zeros(self.num_interval)
            for i in range(self.num_interval):
                for j, angle in enumerate(relative_rotation_angle):
                    if i * self.interval <= angle < (i + 1) * self.interval:
                        if convergence[j]:
                            success_by_interval[i] += 1
                        else:
                            fail_by_interval[i] += 1

            success_rate_by_interval = []
            for i in range(self.num_interval):
                if success_by_interval[i] + fail_by_interval[i] == 0:
                    success_rate_by_interval.append(0)
                else:
                    success_rate_by_interval.append(
                        success_by_interval[i] / (success_by_interval[i] + fail_by_interval[i]) * 100
                    )
            return success_rate_by_interval

        fig = plt.figure(figsize=self.figsize)
        ax1 = fig.add_subplot(111)
        sns.set_style("whitegrid")
        x = np.arange(self.num_interval)

        if len(pickle_data_list) == 1:
            success_rate_by_interval = get_success_rate_by_interval(convergence, relative_rotation_angle)
            sns.barplot(x=x + 0.5, y=success_rate_by_interval, ax=ax1, color="C0")
            for i in range(self.num_interval):
                if success_rate_by_interval[i] == 0 or success_rate_by_interval[i] == 100:
                    ax1.text(
                        i,
                        success_rate_by_interval[i] + 0.05,
                        f"{success_rate_by_interval[i]:.0f}",
                        ha="center",
                        size=self.marker_font_size,
                    )
                else:
                    ax1.text(
                        i,
                        success_rate_by_interval[i] + 0.05,
                        f"{success_rate_by_interval[i]:.0f}",
                        ha="center",
                        size=self.marker_font_size,
                    )
        else:
            icp_success_rate_by_interval = get_success_rate_by_interval(icp_convergence, icp_relative_rotation_angle)
            ransac_success_rate_by_interval = get_success_rate_by_interval(ransac_convergence, ransac_relative_rotation_angle)
            fgr_success_rate_by_interval = get_success_rate_by_interval(fgr_convergence, fgr_relative_rotation_angle)
            sns.barplot(x=x + 0.5, y=ransac_success_rate_by_interval, ax=ax1, color="C1", label="RANSAC")
            sns.barplot(x=x + 0.5, y=fgr_success_rate_by_interval, ax=ax1, color="C2", label="FGR")
            sns.barplot(x=x + 0.5, y=icp_success_rate_by_interval, ax=ax1, color="C0", label="ICP")

            for i in range(self.num_interval)[4:]:
                if icp_success_rate_by_interval[i] == 0 or icp_success_rate_by_interval[i] == 100:
                    ax1.text(
                        i,
                        icp_success_rate_by_interval[i] + 0.1,
                        f"{icp_success_rate_by_interval[i]:.0f}",
                        ha="center",
                        size=self.marker_font_size,
                    )
                else:
                    ax1.text(
                        i,
                        icp_success_rate_by_interval[i] + 0.1,
                        f"{icp_success_rate_by_interval[i]:.0f}",
                        ha="center",
                        size=self.marker_font_size,
                    )
                if ransac_success_rate_by_interval[i] == 0 or ransac_success_rate_by_interval[i] == 100:
                    ax1.text(
                        i,
                        ransac_success_rate_by_interval[i] + 2.1,
                        f"{ransac_success_rate_by_interval[i]:.0f}",
                        ha="center",
                        size=self.marker_font_size,
                    )
                else:
                    ax1.text(
                        i,
                        ransac_success_rate_by_interval[i] + 2.1,
                        f"{ransac_success_rate_by_interval[i]:.0f}",
                        ha="center",
                        size=self.marker_font_size,
                    )
                if fgr_success_rate_by_interval[i] == 0 or fgr_success_rate_by_interval[i] == 100:
                    ax1.text(
                        i,
                        fgr_success_rate_by_interval[i] + 0.1,
                        f"{fgr_success_rate_by_interval[i]:.0f}",
                        ha="center",
                        size=self.marker_font_size,
                    )
                else:
                    ax1.text(
                        i,
                        fgr_success_rate_by_interval[i] + 0.1,
                        f"{fgr_success_rate_by_interval[i]:.0f}",
                        ha="center",
                        size=self.marker_font_size,
                    )
            # # ax1.legend(fontsize=self.legend_font_size, loc="lower left")

        xticks = [f"{int(self.interval * i)}°" for i in range(self.num_interval)]
        ax1.set_xticks(np.arange(self.num_interval) - 0.5)
        ax1.set_xticklabels(xticks, fontsize=self.tick_font_size, rotation=45)
        ax1.set_xlabel("The angle of rotation for any axis of rotation", fontdict={"fontsize": self.label_font_size})
        yticks = [f"{int(20 * i)}" for i in range(6)]
        ax1.set_yticks(np.arange(0, 120, 20))
        ax1.set_yticklabels(yticks, fontsize=self.tick_font_size)
        ax1.set_ylabel("Success rate (%)", fontdict={"fontsize": self.label_font_size})
        ax1.set_title(
            "Success rate of point cloud registration by interval of rotation angle for any axis of rotation",
            fontdict={"fontsize": self.title_font_size},
        )

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        if len(pickle_data_list) == 1:
            return success_rate_by_interval

    def plot_running_time_distribution(self, pickle_data_list: list, save_path=None):
        print(len(pickle_data_list))
        if len(pickle_data_list) == 1:
            pickle_data = pickle_data_list[0]
        else:
            config_list = [os.path.basename(path).split("_")[-1].split(",")[:6] for path in pickle_data_list]
            for config in config_list:
                global_registration = config[0][1:]
                local_registration = config[1][1:]
                ransac = config[2][1:]
                fgr = config[3][1:]
                vanilla_icp = config[4][1:]
                multi_scale_icp = config[5][1:]
                if (global_registration == "False") and (local_registration == "True") and multi_scale_icp == "True":
                    icp_pickle_data = pickle.load(open(pickle_data_list[config_list.index(config)], "rb"))
                elif global_registration == "True" and ransac == "True":
                    ransac_pickle_data = pickle.load(open(pickle_data_list[config_list.index(config)], "rb"))
                elif global_registration == "True" and fgr == "True":
                    fgr_pickle_data = pickle.load(open(pickle_data_list[config_list.index(config)], "rb"))

        if len(pickle_data_list) == 1:
            convergence, relative_rotation_angle, ratio_dd, running_time = self.get_pkl_data(pickle_data)
        else:
            icp_convergence, icp_relative_rotation_angle, icp_ratio_dd, icp_running_time = self.get_pkl_data(icp_pickle_data)
            ransac_convergence, ransac_relative_rotation_angle, ransac_ratio_dd, ransac_running_time = self.get_pkl_data(
                ransac_pickle_data
            )
            fgr_convergence, fgr_relative_rotation_angle, fgr_ratio_dd, fgr_running_time = self.get_pkl_data(fgr_pickle_data)

        def get_running_time_by_interval(convergence, running_time, relative_rotation_angle):
            running_time_by_interval = []
            for i in range(self.num_interval):
                running_time_interval = []
                for conv, angle, time in zip(convergence, relative_rotation_angle, running_time):
                    if (i * self.interval <= angle < (i + 1) * self.interval) and conv:
                        time_ransac = time["ransac"]
                        time_fgr = time["fgr"]
                        time_vanilla = time["vanilla"]
                        time_multi_scale = time["multi_scale"]
                        time_by_type = time_ransac + time_fgr + time_vanilla + time_multi_scale
                        running_time_interval.append(time_by_type)
                running_time_by_interval.append(running_time_interval)
            return running_time_by_interval

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        plt.xlim(-0.5, self.num_interval - 0.5)
        plt.ylim(0, 0.3)
        sns.set_style("whitegrid")
        x = np.arange(self.num_interval)

        if len(pickle_data_list) == 1:
            running_time_by_interval = get_running_time_by_interval(convergence, running_time, relative_rotation_angle)
            mean_time, median_time, std_time = [], [], []
            for i, running_time in enumerate(running_time_by_interval):
                if running_time != []:
                    mean_time.append(np.mean(running_time))
                    median_time.append(np.median(running_time))
                    std_time.append(np.std(running_time))
                elif running_time == [] and i == 0:
                    mean_time.append(np.mean(running_time_by_interval[i + 1]))
                    median_time.append(np.median(running_time_by_interval[i + 1]))
                    std_time.append(np.std(running_time_by_interval[i + 1]))
                elif running_time == [] and i == self.num_interval - 1:
                    mean_time.append(np.mean(running_time_by_interval[i - 1]))
                    median_time.append(np.median(running_time_by_interval[i - 1]))
                    std_time.append(np.std(running_time_by_interval[i - 1]))
                else:
                    mean_time.append((np.mean(running_time_by_interval[i - 1]) + np.mean(running_time_by_interval[i + 1])) / 2)
                    median_time.append((np.median(running_time_by_interval[i - 1]) + np.median(running_time_by_interval[i + 1])) / 2)
                    std_time.append((np.std(running_time_by_interval[i - 1]) + np.std(running_time_by_interval[i + 1])) / 2)
            # sns.boxplot(data=running_time_by_interval, ax=ax)
            ax.plot(x, mean_time, color="C0", alpha=0.2)
            ax.fill_between(
                x,
                np.array(mean_time) - np.array(std_time),
                np.array(mean_time) + np.array(std_time),
                alpha=0.2,
                color="C0",
            )
            ax.plot(x, mean_time, color="C0", label="Mean")
            ax.plot(x, median_time, color="C0", label="Median")
            # ax.legend(fontsize=self.legend_font_size, loc="upper left")
        else:
            icp_running_time_by_interval = get_running_time_by_interval(
                icp_convergence, icp_running_time, icp_relative_rotation_angle
            )
            ransac_running_time_by_interval = get_running_time_by_interval(
                ransac_convergence, ransac_running_time, ransac_relative_rotation_angle
            )
            fgr_running_time_by_interval = get_running_time_by_interval(
                fgr_convergence, fgr_running_time, fgr_relative_rotation_angle
            )

            icp_mean_running_time, icp_std_running_time = [], []
            ransac_mean_running_time, ransac_std_running_time = [], []
            fgr_mean_running_time, fgr_std_running_time = [], []

            for icp_running_time, ransac_running_time, fgr_running_time in zip(
                icp_running_time_by_interval, ransac_running_time_by_interval, fgr_running_time_by_interval
            ):
                if icp_running_time != []:
                    icp_mean_running_time.append(np.mean(icp_running_time))
                    icp_std_running_time.append(np.std(icp_running_time))
                else:
                    icp_mean_running_time.append(0)
                    icp_std_running_time.append(0)

                if ransac_running_time != []:
                    ransac_mean_running_time.append(np.mean(ransac_running_time))
                    ransac_std_running_time.append(np.std(ransac_running_time))
                else:
                    ransac_mean_running_time.append(0)
                    ransac_std_running_time.append(0)

                if fgr_running_time != []:
                    fgr_mean_running_time.append(np.mean(fgr_running_time))
                    fgr_std_running_time.append(np.std(fgr_running_time))
                else:
                    fgr_mean_running_time.append(0)
                    fgr_std_running_time.append(0)

            for mean_running_time in icp_mean_running_time:
                count = 0
                if mean_running_time == 0:
                    count += 1
                x_icp = np.arange(self.num_interval - count)
            icp_mean_running_time = [i for i in icp_mean_running_time if i != 0]
            icp_std_running_time = [i for i in icp_std_running_time if i != 0]

            ax.plot(x_icp, icp_mean_running_time, color="C0", label="ICP")
            ax.fill_between(
                x_icp,
                np.array(icp_mean_running_time) - np.array(icp_std_running_time),
                np.array(icp_mean_running_time) + np.array(icp_std_running_time),
                alpha=0.2,
                color="C0",
            )
            ax.plot(x, ransac_mean_running_time, color="C1", label="RANSAC")
            ax.fill_between(
                x,
                np.array(ransac_mean_running_time) - np.array(ransac_std_running_time),
                np.array(ransac_mean_running_time) + np.array(ransac_std_running_time),
                alpha=0.2,
                color="C1",
            )
            ax.plot(x, fgr_mean_running_time, color="C2", label="FGR")
            ax.fill_between(
                x,
                np.array(fgr_mean_running_time) - np.array(fgr_std_running_time),
                np.array(fgr_mean_running_time) + np.array(fgr_std_running_time),
                alpha=0.2,
                color="C2",
            )
            # # ax.legend(fontsize=self.legend_font_size)

        xticks = [f"{int(self.interval * i)}°" for i in range(self.num_interval)]
        ax.set_xticks(np.arange(self.num_interval) - 0.5)
        ax.set_xticklabels(xticks, fontsize=self.tick_font_size, rotation=45)
        ax.set_xlabel("The angle of rotation for any axis of rotation", fontdict={"fontsize": self.label_font_size})
        yticks = [f"{i:.2f}" for i in np.arange(0, 0.3, 0.05)]
        ax.set_yticks(np.arange(0, 0.3, 0.05))
        ax.set_yticklabels(yticks, fontsize=self.tick_font_size)
        ax.set_ylabel("Running time (s)", fontdict={"fontsize": self.label_font_size})
        ax.set_title(
            "Running time for each interval of rotation angle for any axis of rotation",
            fontdict={"fontsize": self.title_font_size},
        )

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


def generate_random_vectors(num_samples: int, angle_range: list([float, float])) -> np.ndarray:
    random_vectors = np.empty((0, 3))
    for _ in range(num_samples):
        x, y, z = np.random.uniform(-1, 1, 3)
        random_vectors = np.concatenate((random_vectors, np.array([[x, y, z]])), axis=0)

    random_vectors /= np.linalg.norm(random_vectors, axis=1)[:, np.newaxis]
    return random_vectors


def generate_low_dispersion_random_uniform(n, a, b):
    intervals = []
    for i in range(n):
        intervals.append([a + (b - a) * i / n, a + (b - a) * (i + 1) / n])
    np.random.shuffle(intervals)

    random_numbers = []
    for interval in intervals:
        num = np.random.uniform(interval[0], interval[1])
        random_numbers.append(num)
    return np.array(random_numbers)


def get_random_uniform_rotvec(num_samples: int, angle_range: list([float, float])) -> np.ndarray:
    rotation_vectors = generate_random_vectors(num_samples, angle_range)
    min_angle, max_angle = angle_range
    rotation_angle = generate_low_dispersion_random_uniform(num_samples, min_angle, max_angle)
    rotation_vectors *= rotation_angle[:, np.newaxis]
    return rotation_vectors


def get_random_uniform_transvec(num_samples: int, distance_range: list([float, float])) -> np.ndarray:
    translation_vectors = generate_random_vectors(num_samples, distance_range)
    min_distance, max_distance = distance_range
    translation_distance = generate_low_dispersion_random_uniform(num_samples, min_distance, max_distance)
    translation_vectors *= translation_distance[:, np.newaxis]
    return translation_vectors


def pcd_registration_analysis(pickle_savedir: str):
    import copy

    global VISUALIZE
    VISUALIZE = False

    pcl_file = os.path.join(os.path.dirname(__file__), "pvn3d/datasets/ycb/YCB_Video_Dataset/models/102_valve_model/points.xyz")
    pcl_fix = np.loadtxt(pcl_file)
    pcl_mov = copy.deepcopy(pcl_fix)
    pcl_fix = pcd_reg.load_point_clouds(pcl_fix)
    pcl_mov = pcd_reg.load_point_clouds(pcl_mov)

    check_directory(pickle_savedir)
    file_name = f"102_valve_model_{GLOBAL_REGISTRATION, LOCAL_REGISTRATION, RANSAC, FGR, VANILLA_ICP, MULTI_SCALE_ICP, PointCloudRegistration.voxel_size, PointCloudRegistration.max_correspondence_distance}"
    angle_distribution_img_file = os.path.join(pickle_savedir, f"angle_distribution_{file_name}.png")
    running_time_distribution_img_file = os.path.join(pickle_savedir, f"running_time_distribution_{file_name}.png")
    pickle_file_name = f"{file_name}.pkl"
    pickle_data_file = os.path.join(pickle_savedir, pickle_file_name)
    success_by_hyperparam = []

    if os.path.exists(pickle_data_file):
        with open(pickle_data_file, "rb") as pickle_data_file:
            pickle_data = pickle.load(pickle_data_file)
    else:
        pickle_data = []
        num_samples_rotvec = 100
        min_angle_deg, max_angle_deg = 0, 180
        min_angle, max_angle = np.deg2rad(min_angle_deg), np.deg2rad(max_angle_deg)
        angle_range = [min_angle, max_angle]
        random_rotations_fix = get_random_uniform_rotvec(num_samples_rotvec, angle_range)
        random_rotations_mov = get_random_uniform_rotvec(num_samples_rotvec, angle_range)

        num_samples_transvec = num_samples_rotvec
        diameter = np.linalg.norm(np.asarray(pcl_fix.get_max_bound()) - np.asarray(pcl_fix.get_min_bound()))
        min_distance, max_distance = 0, diameter
        distance_range = [min_distance, max_distance]
        random_translations_fix = get_random_uniform_transvec(num_samples_transvec, distance_range)
        random_translations_mov = get_random_uniform_transvec(num_samples_transvec, distance_range)

        for i in tqdm(range(num_samples_rotvec), desc="Fix", position=0):
            rotmat_fix = o3d.geometry.PointCloud.get_rotation_matrix_from_axis_angle(random_rotations_fix[i])
            pcl_fix_copy = copy.deepcopy(pcl_fix)
            rotated_pcl_fix = pcl_fix_copy.rotate(rotmat_fix)  # .translate(random_translations_fix[i])
            surface_pcl_fix = pcd_reg.remove_hidden_points(rotated_pcl_fix)

            for j in tqdm(range(num_samples_rotvec), desc="Mov", position=1, leave=False):
                rotmat_mov = o3d.geometry.PointCloud.get_rotation_matrix_from_axis_angle(random_rotations_mov[j])
                pcl_mov_copy = copy.deepcopy(pcl_mov)  # .translate(random_translations_mov[j])
                rotated_pcl_mov = pcl_mov_copy.rotate(rotmat_mov)

                running_time = {"ransac": 0, "fgr": 0, "vanilla": 0, "multi_scale": 0}
                ##############################
                # Global registration
                ##############################
                voxel_size = PointCloudRegistration.voxel_size
                mov_down, mov_down_fpfh = pcd_reg.preprocess_point_cloud(rotated_pcl_mov, voxel_size)
                fix_down, fix_down_fpfh = pcd_reg.preprocess_point_cloud(surface_pcl_fix, voxel_size)
                if GLOBAL_REGISTRATION:
                    if RANSAC:
                        result_ransac, running_time_ransac = pcd_reg.execute_global_registration(
                            mov_down, fix_down, mov_down_fpfh, fix_down_fpfh, voxel_size
                        )
                        running_time["ransac"] += running_time_ransac
                        pcd_reg.draw_registration_result(mov_down, fix_down, result_ransac, "RANSAC") if VISUALIZE else None
                    elif FGR:
                        result_fast, running_time_fgr = pcd_reg.execute_fast_global_registration(
                            mov_down, fix_down, mov_down_fpfh, fix_down_fpfh, voxel_size
                        )
                        running_time["fgr"] += running_time_fgr
                        # pcd_reg.draw_registration_result(
                        #     mov_down, fix_down, result_fast, "Fast Global Registration"
                        # ) if VISUALIZE else None

                    result_global_refinement = result_ransac if RANSAC else result_fast

                init_src_temp_to_target = result_global_refinement.transformation if GLOBAL_REGISTRATION else np.identity(4)

                ##############################
                # Local registration
                ##############################
                if LOCAL_REGISTRATION:
                    if VANILLA_ICP:
                        if pcd_reg.CUDA:
                            device = o3d.core.Device("cuda:0")
                            dtype = o3d.core.float32

                            pcd_src_cuda = o3d.t.geometry.PointCloud.from_legacy(rotated_pcl_mov, device=device, dtype=dtype)
                            pcd_dst_cuda = o3d.t.geometry.PointCloud.from_legacy(surface_pcl_fix, device=device, dtype=dtype)

                            result_local_refinement, running_time_vanilla_cuda = pcd_reg.execute_ICP_registration_CUDA(
                                pcd_src_cuda, pcd_dst_cuda, voxel_size, init_src_temp_to_target
                            )
                            pcd_reg.draw_registration_result(
                                pcd_src_cuda.to_legacy(),
                                pcd_dst_cuda.to_legacy(),
                                result_local_refinement,
                                "Vanilla ICP with CUDA",
                            ) if VISUALIZE else None
                        else:
                            result_local_refinement, running_time_vanilla = pcd_reg.execute_ICP_registration(
                                rotated_pcl_mov, surface_pcl_fix, voxel_size, init_src_temp_to_target
                            )
                            running_time["vanilla"] += running_time_vanilla
                            # pcd_reg.draw_registration_result(
                            #     rotated_pcl_mov, surface_pcl_fix, result_local_refinement, "Vanilla ICP"
                            # ) if VISUALIZE else None
                    elif MULTI_SCALE_ICP:
                        result_local_refinement, running_time_multi_scale = pcd_reg.execute_multi_scale_ICP_registration(
                            rotated_pcl_mov, surface_pcl_fix, PointCloudRegistration.voxel_sizes, init_src_temp_to_target
                        )
                        running_time["multi_scale"] += running_time_multi_scale
                        # pcd_reg.draw_registration_result(
                        #     rotated_pcl_mov, surface_pcl_fix, result_local_refinement, "Multi-scale ICP"
                        # ) if VISUALIZE else None

                transformation = result_global_refinement.transformation if GLOBAL_REGISTRATION else result_local_refinement.transformation
                transformation = transformation.cpu().numpy() if isinstance(transformation, o3d.core.Tensor) else transformation
                result_rotmat = transformation[:3, :3]
                ground_truth_rotmat = np.dot(rotmat_fix, rotmat_mov.T)
                rotmat_diff = result_rotmat - ground_truth_rotmat
                relative_rotation_vector = R.from_matrix(ground_truth_rotmat).as_rotvec()
                relative_rotation_angle = np.linalg.norm(relative_rotation_vector)
                relative_translation_vector = random_translations_fix[i] - random_translations_mov[j]
                ratio_diameter_to_distance = np.linalg.norm(relative_translation_vector) / diameter

                result_rotmat_rotvec = R.from_matrix(np.array(result_rotmat)).as_rotvec()
                ground_truth_rotmat_rotvec = R.from_matrix(np.array(ground_truth_rotmat)).as_rotvec()
                norm_result_rotmat_rotvec = np.linalg.norm(result_rotmat_rotvec)
                norm_ground_truth_rotmat_rotvec = np.linalg.norm(ground_truth_rotmat_rotvec)
                cos_similarity = np.dot(result_rotmat_rotvec, ground_truth_rotmat_rotvec) / (
                    norm_result_rotmat_rotvec * norm_ground_truth_rotmat_rotvec
                ) if norm_result_rotmat_rotvec * norm_ground_truth_rotmat_rotvec != 0 else 0
                convergence = True if cos_similarity > 0.97 else False

                if VISUALIZE and convergence:
                    pcd_reg.draw_registration_result(rotated_pcl_mov, surface_pcl_fix, result_local_refinement)
                pickle_data.append([convergence, relative_rotation_angle, ratio_diameter_to_distance, running_time])

        with open(pickle_data_file, "wb") as pickle_data_file:
            pickle.dump(pickle_data, pickle_data_file)

    icp_operating_range_plotter = ICPOperatingRangePlotter()
    success_rate_by_interval = icp_operating_range_plotter.plot_rotation_angle_distribution(
        [pickle_data], angle_distribution_img_file
    )
    icp_operating_range_plotter.plot_running_time_distribution([pickle_data], running_time_distribution_img_file)
    mean_success_rate = np.mean(success_rate_by_interval)
    success_by_hyperparam.append(
        [PointCloudRegistration.voxel_size, PointCloudRegistration.max_correspondence_distance, mean_success_rate]
    )

    return success_by_hyperparam


def main():
    pickle_savedir = os.path.join(os.path.dirname(__file__), "icp_pickle")
    if not args.total_figures:
        pcd_registration_analysis(pickle_savedir)

    else:
        pickle_files = glob.glob(os.path.join(pickle_savedir, "*.pkl"))
        icp_operating_range_plotter = ICPOperatingRangePlotter()
        icp_operating_range_plotter.plot_rotation_angle_distribution(
            pickle_files, pickle_savedir + "/comparison_rotation_angle_distribution.png"
        )
        icp_operating_range_plotter.plot_running_time_distribution(
            pickle_files, pickle_savedir + "/comparison_running_time_distribution.png"
        )


if __name__ == "__main__":
    main()
