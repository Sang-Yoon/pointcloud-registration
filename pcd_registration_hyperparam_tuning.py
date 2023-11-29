import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from pcd_registration_o3d import PointCloudRegistration
from icp_operating_range_surface import ICPOperatingRangePlotter, pcd_registration_analysis


class HyperparamTuningPlotter:
    def __init__(self):
        pass

    def plot_success_by_hyperparam(self, success_by_hyperparam_data):
        success_by_hyperparam_data = np.array(success_by_hyperparam_data)
        success_by_hyperparam_data = success_by_hyperparam_data.reshape(-1, 3)

        voxel_size = success_by_hyperparam_data[:, 0]
        max_correspondence_distance = success_by_hyperparam_data[:, 1]
        mean_success_rate_by_hyperparam = success_by_hyperparam_data[:, 2]

        optimal_idx = np.argmax(mean_success_rate_by_hyperparam)
        optimal_voxel_size = voxel_size[optimal_idx]
        optimal_max_correspondence_distance = max_correspondence_distance[optimal_idx]
        print(f"optimal_voxel_size: {optimal_voxel_size}")
        print(f"optimal_max_correspondence_distance: {optimal_max_correspondence_distance}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(voxel_size, max_correspondence_distance, mean_success_rate_by_hyperparam)
        ax.set_xlabel("voxel_size")
        ax.set_ylabel("max_correspondence_distance")
        ax.set_zlabel("mean_success_rate_by_hyperparam")
        save_path = os.path.join(os.path.dirname(__file__), "hyperparam_tuning.png")
        print(f"Saving hyperparam tuning plot to {save_path}")
        plt.savefig(save_path)


def main():
    pcd_reg = PointCloudRegistration()

    pickle_savedir = os.path.join(os.path.dirname(__file__), "icp_pickle")
    hyperparam_file_name = f"102_valve_model_hyperparam"
    hyperparam_pickle_file = os.path.join(pickle_savedir, f"{hyperparam_file_name}.pkl")

    ###############################################
    # Hyperparameters Tuning
    ###############################################
    if os.path.exists(hyperparam_pickle_file):
        with open(hyperparam_pickle_file, "rb") as f:
            success_by_hyperparam_data = pickle.load(f)
    else:
        success_by_hyperparam_data = []
        voxel_size_space = np.linspace(0.001, 0.004, 10)
        max_correspondence_distance_space = np.linspace(1.0, 4.0, 10)
        for voxel_size in voxel_size_space:
            for max_correspondence_distance in max_correspondence_distance_space:
                PointCloudRegistration.voxel_size = voxel_size
                PointCloudRegistration.max_correspondence_distance = (
                    PointCloudRegistration.voxel_size * max_correspondence_distance
                )

                success_by_hyperparam = pcd_registration_analysis(pickle_savedir)
                success_by_hyperparam_data.append(success_by_hyperparam)
        with open(hyperparam_pickle_file, "wb") as f:
            pickle.dump(success_by_hyperparam_data, f)

    ###############################################
    # Hyperparameters Tuning Plotting
    ###############################################
    hyperparam_tuning_plotter = HyperparamTuningPlotter()
    hyperparam_tuning_plotter.plot_success_by_hyperparam(success_by_hyperparam_data)


if __name__ == "__main__":
    main()
