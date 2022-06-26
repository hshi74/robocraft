import copy
import cv2
import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import open3d as o3d
import os, sys
import pdb
import pymeshfix
import pyvista as pv
import scipy
import torch

from datetime import datetime
from pysdf import SDF
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2mat


n_instance = 1
gravity = 1
draw_mesh = 0
scene_params = np.zeros(3)
scene_params[0] = n_instance
scene_params[1] = gravity
scene_params[-1] = draw_mesh

n_cam = 4
n_shapes = 3
aug_n_shapes = 31
n_points = 300
n_frame = 120

data_names = ['positions', 'shape_quats', 'scene_params']
floor_pos = np.array([0.5, 0, 0.5])
floor_dim = 9
primitive_dim = 11
floor_size = 0.05
voxel_size = 0.01
gripper_h = 0.2
gripper_r = 0.045

task_name = 'ngrip_fixed'
algo = 'crop'

visualize = False
o3d_write = False


def chamfer_distance(x, y):
    x = x[:, None, :].repeat(1, y.size(0), 1) # x: [N, M, D]
    y = y[None, :, :].repeat(x.size(0), 1, 1) # y: [N, M, D]
    dis = torch.norm(torch.add(x, -y), 2, dim=2)    # dis: [N, M]
    dis_xy = torch.mean(torch.min(dis, dim=1)[0])   # dis_xy: mean over N
    dis_yx = torch.mean(torch.min(dis, dim=0)[0])   # dis_yx: mean over M

    return dis_xy + dis_yx


def em_distance(x, y):
    x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
    y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
    dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
    cost_matrix = dis.numpy()
    try:
        ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
    except:
        print("Error in linear sum assignment!")
    emd = torch.mean(torch.norm(torch.add(x[ind1], -y[ind2]), 2, dim=1))
    
    return emd


def o3d_visualize(display_list):
    if o3d_write:
        cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
        image_path = os.path.join(cd, '..', '..', 'images', f'{time_now}.png')

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geo in display_list:
            vis.add_geometry(geo)
            vis.update_geometry(geo)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(image_path)
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries(display_list, mesh_show_back_face=True)


def visualize_points(ax, all_points, n_points):
    points = ax.scatter(all_points[:n_points, 0], all_points[:n_points, 2], all_points[:n_points, 1], c='b', s=10)
    shapes = ax.scatter(all_points[n_points:, 0], all_points[n_points:, 2], all_points[n_points:, 1], c='r', s=10)
    
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    ax.invert_yaxis()

    return points, shapes


def plt_render(particles_set, n_particle, render_path):
    n_frames = particles_set[0].shape[0]
    rows = 2
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 6))
    row_titles = ['GT', 'Sample']
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        big_axes[i].set_title(row_titles[i], fontweight='semibold')
        big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, particles_set[i][0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    plt.tight_layout()
    # plt.show()

    def update(step):
        outputs = []
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle:, 0], states[step, n_particle:, 2], states[step, n_particle:, 1])
                outputs.append(points)
                outputs.append(shapes)
        return outputs

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), blit=False)
    
    # plt.show()
    anim.save(render_path, writer=animation.PillowWriter(fps=10))


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def update_position(n_shapes, prim_pos, positions=None, pts=None, floor=None, n_points=300, task_name='gripper'):
    if positions is None:
        positions = np.zeros([n_points + n_shapes, 3])
    if pts is not None:
        positions[:n_points, :3] = pts
    if floor is not None:
        positions[n_points, :3] = floor

    if task_name == 'gripper':
        positions[n_points+1, :3] = prim_pos[0]
        positions[n_points+2, :3] = prim_pos[1]
        # gt_positions[n_points+1, :3] = prim_pos1
        # gt_positions[n_points+2, :3] = prim_pos2
    else:
        positions[n_points+1, :3] = prim_pos[0]
        # gt_positions[n_points+1, :3] = prim_pos
    return positions


def shape_aug(states, n_points):
    states_tmp = states[:n_points]
    prim1 = states[n_points + 1]
    prim2 = states[n_points + 2]
    new_floor = np.array([[0.25, 0., 0.25], [0.25, 0., 0.5], [0.25, 0., 0.75],
                          [0.5, 0., 0.25], [0.5, 0., 0.5], [0.5, 0., 0.75],
                          [0.75, 0., 0.25], [0.75, 0., 0.5], [0.75, 0., 0.75]])
    new_prim1 = []
    for j in range(11):
        prim1_tmp = np.array([prim1[0], prim1[1] + 0.018 * (j - 5), prim1[2]])
        new_prim1.append(prim1_tmp)
    new_prim1 = np.stack(new_prim1)

    new_prim2 = []
    for j in range(11):
        prim2_tmp = np.array([prim2[0], prim2[1] + 0.018 * (j - 5), prim2[2]])
        new_prim2.append(prim2_tmp)
    new_prim2 = np.stack(new_prim2)
    new_states = np.concatenate([states_tmp, new_floor, new_prim1, new_prim2])
    return new_states


def shape_aug_3D(states, rot1, rot2, n_points):
    states_tmp = states[:n_points]
    prim1 = states[n_points + 1]
    prim2 = states[n_points + 2]
    new_floor = np.array([[0.25, 0., 0.25], [0.25, 0., 0.5], [0.25, 0., 0.75],
                          [0.5, 0., 0.25], [0.5, 0., 0.5], [0.5, 0., 0.75],
                          [0.75, 0., 0.25], [0.75, 0., 0.5], [0.75, 0., 0.75]])
    new_prim1 = []
    for j in range(11):
        prim1_tmp = np.array([prim1[0], prim1[1] + 0.018 * (j - 5), prim1[2]])
        new_prim1.append(prim1_tmp)
    new_prim1 = np.stack(new_prim1)
    new_prim1 = (quat2mat(rot1) @ (new_prim1-prim1).T).T + prim1

    new_prim2 = []
    for j in range(11):
        prim2_tmp = np.array([prim2[0], prim2[1] + 0.018 * (j - 5), prim2[2]])
        new_prim2.append(prim2_tmp)
    new_prim2 = np.stack(new_prim2)
    new_prim2 = (quat2mat(rot2) @ (new_prim2-prim2).T).T + prim2
    
    new_states = np.concatenate([states_tmp, new_floor, new_prim1, new_prim2])
    return new_states


def gen_3D_za(intrinsic, extrinsic, rgb, depth, w=512, h=512):
    fx = fy = intrinsic[0, 0]
    cx = cy = intrinsic[0, 2]
    cam = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    # extrinsic = get_ext(camera_rot, np.array(camera_pos))
    RGB = o3d.geometry.Image(np.ascontiguousarray(np.rot90(rgb,0,(0,1))).astype(np.uint8))
    DEPTH = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(RGB, DEPTH, depth_scale=1., depth_trunc=np.inf, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)
    pcd.transform(np.linalg.inv(extrinsic))

    return pcd


def im2threed(rgb, depth, cam_params):
    n_cam = 4
    pcds = []
    for i in range(n_cam):
        pcd = gen_3D_za(intrinsic=cam_params['intrinsic'], extrinsic=cam_params[f'cam{i+1}_ext'], 
                        rgb=rgb[i], depth=depth[i])
        pcds.append(pcd)
    pcd_all = pcds[0] + pcds[1] + pcds[2] + pcds[3]
    return pcd_all


def flip_inward_normals(pcd, center, threshold=0.7):
    # Flip normal if normal points inwards by changing vertex order
    # https://math.stackexchange.com/questions/3114932/determine-direction-of-normal-vector-of-convex-polyhedron-in-3d
    
    # Get vertices and triangles from the mesh
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    # For each triangle in the mesh
    flipped_count = 0
    for i, n in enumerate(normals):
        # Compute vector from 1st vertex of triangle to center
        norm_ref = points[i] - center
        # Compare normal to the vector
        if np.dot(norm_ref, n) < 0:
            # Change vertex order to flip normal direction
            flipped_count += 1 
            if flipped_count > threshold * normals.shape[0]:
                normals = np.negative(normals)
                break

    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def poisson_mesh_reconstruct(rest, cube, prim_pos, visualize=False):
    gripper_label = np.where(np.array(rest.colors)[:, 2] >= 0.6)
    grippers = rest.select_by_index(gripper_label[0])

    labels = np.array(grippers.cluster_dbscan(eps=0.03, min_points=100))
    gripper1 = grippers.select_by_index(np.where(labels == 0)[0])
    gripper2 = grippers.select_by_index(np.where(labels > 0)[0])
    for gripper in [gripper1, gripper2]:
        gripper.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        gripper.estimate_normals()
        gripper.orient_normals_consistent_tangent_plane(100)
        
        center = gripper.get_center()
        if np.dot(center - prim_pos[0], center - prim_pos[0]) < np.dot(center - prim_pos[1], center - prim_pos[1]):
            center = prim_pos[0]
        else:
            center = prim_pos[1]
        gripper = flip_inward_normals(gripper, center)

    cube.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    cube.estimate_normals()
    cube.orient_normals_consistent_tangent_plane(100)
    center = cube.get_center()
    cube = flip_inward_normals(cube, center)

    raw_pcd = gripper1 + gripper2 + cube

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(raw_pcd, depth=8)

    if visualize:
        o3d.visualization.draw_geometries([raw_pcd, mesh], mesh_show_back_face=True)
    
    return mesh


def mesh_reconstruct(pcd, algo="filter", alpha=0.5, depth=8, visualize=False):
    if algo == "filter":
        point_cloud = pv.PolyData(np.asarray(pcd.points))
        surf = point_cloud.reconstruct_surface()

        mf = pymeshfix.MeshFix(surf)
        mf.repair()
        pymesh = mf.mesh

        if visualize:
            pl = pv.Plotter()
            pl.add_mesh(point_cloud, color='k', point_size=10)
            pl.add_mesh(pymesh)
            pl.add_title('Reconstructed Surface')
            pl.show()

        mesh = pymesh
    else:
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(100)
        center = pcd.get_center()
        pcd = flip_inward_normals(pcd, center)

        if algo == "ball_pivot":
            radii = [0.005, 0.01, 0.02, 0.04, 0.08]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
        elif algo == "alpha_shape":
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha, tetra_mesh, pt_map)
        elif algo == "poisson":
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        else:
            raise NotImplementedError

        mesh.paint_uniform_color([0,1,0])
        mesh.compute_vertex_normals()

        if visualize:
            o3d_visualize([pcd, mesh])
    
    return mesh


def length(x_arr):
    return np.array([np.sqrt(x.dot(x) + 1e-8) for x in x_arr])


def is_inside(pt_pos, tool_pos, tool_rot, task="gripper"):
    if task=="gripper":
        pt_pos = pt_pos - np.tile(tool_pos, (pt_pos.shape[0], 1))
        pt_pos = (quat2mat(tool_rot) @ pt_pos.T).T
        p2 = copy.copy(pt_pos)
        for i in range(p2.shape[0]):
            p2[i, 1] += gripper_h / 2 + 0.01
            p2[i, 1] -= min(max(p2[i, 1], 0.0), gripper_h)
        return length(p2) - gripper_r
    else:
        raise NotImplementedError


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def fps(pts, K=300):
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts


def crop(rest, cube, prev_pcd, grippers, prim_pos, prim_rot, n_points, back, visualize=False):
    if back:
        # print(f'touching status: {is_touching[0]} and {is_touching[1]}')
        prev_points = np.asarray(prev_pcd[-1].points)
        sampled_points = prev_points
    else:
        lower = cube.get_min_bound()
        upper = cube.get_max_bound()
        sample_size = round(20 * n_points)
        sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower

        selected_mesh = poisson_mesh_reconstruct(rest, cube, prim_pos, visualize=visualize)
        f = SDF(selected_mesh.vertices, selected_mesh.triangles)

        sdf = f(sampled_points)
        sampled_points = sampled_points[-sdf < 0, :]

    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    if visualize:
        sampled_pcd.paint_uniform_color([0,0,0])
        o3d_visualize([sampled_pcd])

    if not back:
        for tool_pos, tool_rot in zip(prim_pos, prim_rot):
            # if not in the tool, then it's valid
            inside_idx = is_inside(sampled_points, tool_pos, tool_rot)
            sampled_points = sampled_points[inside_idx > 0]  

        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
        if visualize:
            sampled_pcd.paint_uniform_color([0,0,0])
            o3d_visualize([sampled_pcd])

        cl, inlier_ind = sampled_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
        sampled_pcd = sampled_pcd.select_by_index(inlier_ind)

        if visualize:
            sampled_pcd.paint_uniform_color([0,0,0])
            o3d_visualize([sampled_pcd])

    selected_points = fps(np.asarray(sampled_pcd.points), n_points)

    if visualize:
        fps_pcd = o3d.geometry.PointCloud()
        fps_pcd.points = o3d.utility.Vector3dVector(selected_points)
        fps_pcd.paint_uniform_color([0,0,0])
        o3d_visualize([fps_pcd])

    return sampled_pcd, selected_points


def patch(cube, prev_pcd, grippers, n_points, back, looking_ahead=5, surface=True, visualize=False):
    if visualize:
        cube.paint_uniform_color([0,0,0])
        o3d_visualize([cube])

    selected_points = np.asarray(cube.points)

    is_touching = [False, False]
    if not back:
        if len(prev_pcd) < looking_ahead:
            bounding_mesh = mesh_reconstruct(cube, algo='alpha_shape', alpha=0.5, visualize=visualize)
            f = SDF(bounding_mesh.vertices, bounding_mesh.triangles)
        else:
            curr_mesh = mesh_reconstruct(cube, algo='filter', visualize=visualize)
            f_curr = SDF(curr_mesh.points, curr_mesh.faces.reshape(curr_mesh.n_faces, -1)[:, 1:])

            prev_mesh = mesh_reconstruct(prev_pcd[-looking_ahead], algo='filter', visualize=visualize)
            f_prev = SDF(prev_mesh.points, prev_mesh.faces.reshape(prev_mesh.n_faces, -1)[:, 1:])

        for i, gripper_pcd in enumerate(grippers):
            gripper_points = np.asarray(gripper_pcd.points)
            if len(prev_pcd) < looking_ahead:
                sdf = f(gripper_points)
                gripper_points_in = gripper_points[-sdf < 0, :]
            else:
                sdf = f_curr(gripper_points)
                gripper_points_in = gripper_points[-sdf < 0, :]
                sdf = f_prev(gripper_points_in)
                gripper_points_in = gripper_points_in[-sdf < 0, :]
            print(f"Number of gripper points inside: {gripper_points_in.size}")
            if gripper_points_in.size > 0:
                is_touching[i] = True
                selected_points = np.concatenate((selected_points, gripper_points_in))

    if visualize:
        selected_pcd = o3d.geometry.PointCloud()
        selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
        selected_pcd.paint_uniform_color([0,0,0])
        o3d_visualize([selected_pcd])
    
    # if not touching
    if len(prev_pcd) > 0 and not is_touching[0] and not is_touching[1]:
        # print(f'touching status: {is_touching[0]} and {is_touching[1]}')
        prev_points = np.asarray(prev_pcd[-1].points)
        selected_points = np.concatenate((selected_points, prev_points))

    if visualize:
        selected_pcd = o3d.geometry.PointCloud()
        selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
        selected_pcd.paint_uniform_color([0,0,0])
        o3d_visualize([selected_pcd])

    selected_pcd = o3d.geometry.PointCloud()
    selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
    selected_pcd = selected_pcd.voxel_down_sample(voxel_size=voxel_size)

    if not back:
        cl, inlier_ind = selected_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
        selected_pcd = selected_pcd.select_by_index(inlier_ind)

    if visualize:
        selected_pcd.paint_uniform_color([0,0,0])
        o3d_visualize([selected_pcd])

    if surface:
        selected_mesh = mesh_reconstruct(selected_pcd, algo='alpha_shape', alpha=0.03, visualize=visualize)
        selected_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(selected_mesh, n_points)
        if visualize:
            o3d_visualize([selected_surface])

        selected_points = np.asarray(selected_surface.points)
    else:
        lower = selected_pcd.get_min_bound()
        upper = selected_pcd.get_max_bound()
        sample_size = round(5 * n_points)
        sampled_points = np.random.rand(sample_size, 3) * (upper - lower) + lower
        
        selected_mesh = mesh_reconstruct(selected_pcd, algo='filter', visualize=visualize)
        f = SDF(selected_mesh.points, selected_mesh.faces.reshape(selected_mesh.n_faces, -1)[:, 1:])

        sdf = f(sampled_points)
        sampled_points = sampled_points[-sdf < 0, :]

        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        sampled_pcd = sampled_pcd.voxel_down_sample(voxel_size=voxel_size)

        cl, inlier_ind = sampled_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
        sampled_pcd = sampled_pcd.select_by_index(inlier_ind)

        selected_points = fps(np.asarray(sampled_pcd.points), K=n_points)

        if visualize:
            fps_pcd = o3d.geometry.PointCloud()
            fps_pcd.points = o3d.utility.Vector3dVector(selected_points)
            fps_pcd.paint_uniform_color([0,0,0])
            o3d_visualize([fps_pcd])

    return selected_pcd, selected_points


def process_raw_pcd(pcd_all, visualize=False):
    segment_models, inliers = pcd_all.segment_plane(distance_threshold=0.01,ransac_n=3,num_iterations=100)
    # cube + gripper
    rest = pcd_all.select_by_index(inliers, invert=True)

    if visualize:
        o3d_visualize([rest])

    rest_colors = np.asarray(rest.colors)
    cube_label = np.where(rest_colors[:, 2] < 0.1)
    cube = rest.select_by_index(cube_label[0])

    cl, inlier_ind = cube.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.5)
    cube = cube.select_by_index(inlier_ind)

    if visualize:
        o3d_visualize([cube])

    return rest, cube

def gen_data_one_frame(rgb, depth, cam_params, prim_pos, prim_rot, tool_info, back, prev_pcd, n_points=300):
    pcd_all = im2threed(rgb, depth, cam_params)
    rest, cube = process_raw_pcd(pcd_all)

    grippers = []
    for k in range(len(prim_pos)):
        gripper = o3d.geometry.TriangleMesh.create_cylinder(gripper_r, gripper_h)
        gripper_pcd = gripper.sample_points_poisson_disk(500)
        gripper_pcd.paint_uniform_color([0,0,0])                                       
        gripper_points = np.asarray(gripper_pcd.points)
        gripper_points = (quat2mat(prim_rot[k]) @ euler2mat(np.pi / 2, 0, 0) @ gripper_points.T).T + prim_pos[k]
        gripper_pcd.points = o3d.utility.Vector3dVector(gripper_points)
        grippers.append(gripper_pcd)
    selected_pcd, selected_points = crop(rest, cube, prev_pcd, grippers, prim_pos, prim_rot, n_points, back, visualize=False)
    return selected_points
    

def main():
    time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")

    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    output_path = os.path.join(cd, '..', '..', 'dataset', f'sample_{task_name}_{time_now}')

    ##### REPLACE with the output dir from last step #####
    data_dir = os.path.join(cd, '..', '..', 'dataset', 'ngrip_fixed_25-Jun-2022-12:32:18.300663')
    dir_list = sorted(glob.glob(os.path.join(data_dir, '*')))

    for vid_idx in range(0, len(dir_list)):
        print(f'========== Video {vid_idx} ==========')
        data_path = dir_list[vid_idx]
        rollout_path = os.path.join(output_path, f"{vid_idx:03d}")
        os.system('mkdir -p ' + rollout_path)

        all_positions = []
        all_gt_positions = []
        prev_pcd = []
        emd_loss_list = []
        chamfer_loss_list = []
        for j in range(0, n_frame):
            back = (j % 40) >= 30

            print(f'+++++ Frame {j} +++++')
            rgb = []
            for k in range(n_cam):
                rgb.append(cv2.imread(data_path + f"/{j:03d}_rgb_{k}.png", cv2.IMREAD_COLOR)[..., ::-1])
            
            d_and_p = np.load(data_path + f"/{j:03d}_depth_prim.npy", allow_pickle=True)
            cam_params = np.load(data_path + f"/cam_params.npy", allow_pickle=True).item()
            gt_pos = np.load(data_path + f"/{j:03d}_gtp.npy", allow_pickle=True)
            depth = d_and_p[:4]

            pcd_all = im2threed(rgb, depth, cam_params)
            rest, cube = process_raw_pcd(pcd_all, visualize=visualize)

            prim_pos = [np.array(d_and_p[4][:3]), np.array(d_and_p[5][:3])]
            prim_rot = [np.array(d_and_p[4][3:]), np.array(d_and_p[5][3:])]

            grippers = []
            for k in range(len(prim_pos)):
                gripper = o3d.geometry.TriangleMesh.create_cylinder(gripper_r, gripper_h)

                gripper_pcd = gripper.sample_points_poisson_disk(500)
                gripper_pcd.paint_uniform_color([0,0,0])
                gripper_points = np.asarray(gripper_pcd.points)

                gripper_points = (quat2mat(prim_rot[k]) @ euler2mat(np.pi / 2, 0, 0) @ gripper_points.T).T + prim_pos[k]

                gripper_pcd.points = o3d.utility.Vector3dVector(gripper_points)

                grippers.append(gripper_pcd)

            if visualize:
                print("Visualize grippers...")
                cube.paint_uniform_color([0,0,0])
                o3d_visualize([cube, grippers[0], grippers[1]])

            if algo == 'crop':
                selected_pcd, selected_points = crop(rest, cube, prev_pcd, grippers, prim_pos, prim_rot, n_points, back, visualize=visualize)
            else:
                cube_points = np.asarray(cube.points)
                for tool_pos, tool_rot in zip(prim_pos, prim_rot):
                    inside_idx = is_inside(cube_points, tool_pos, tool_rot)
                    cube_points = cube_points[inside_idx > 0]
                
                cube_new = o3d.geometry.PointCloud()
                cube_new.points = o3d.utility.Vector3dVector(cube_points)

                selected_pcd, selected_points = patch(cube_new, prev_pcd, grippers, n_points, back, surface=False, visualize=visualize)

            prev_pcd.append(selected_pcd)

            emd_loss = em_distance(torch.tensor(selected_points), torch.tensor(gt_pos))
            chamfer_loss = chamfer_distance(torch.tensor(selected_points), torch.tensor(gt_pos))
            emd_loss_list.append(emd_loss)
            chamfer_loss_list.append(chamfer_loss)

            if j >= 1:
                prev_positions = update_position(n_shapes, prim_pos, positions=prev_positions, n_points=n_points)
                prev_gt_positions = update_position(n_shapes, prim_pos, positions=prev_gt_positions, n_points=n_points)
                if 'ngrip_3d' in task_name:
                    prev_shape_positions = shape_aug_3D(prev_positions, prev_prim_ori1, prev_prim_ori2, n_points)
                    prev_shape_gt_positions = shape_aug_3D(prev_gt_positions, prev_prim_ori1, prev_prim_ori2, n_points)
                elif 'ngrip' in task_name:
                    prev_shape_positions = shape_aug(prev_positions, n_points)
                    prev_shape_gt_positions = shape_aug(prev_gt_positions, n_points)
                else:
                    raise NotImplementedError

                all_positions.append(prev_shape_positions)
                all_gt_positions.append(prev_shape_gt_positions)

                shape_shape_quats = np.zeros((aug_n_shapes, 4), dtype=np.float32)
                shape_shape_quats[floor_dim:floor_dim+primitive_dim] = prev_prim_ori1
                shape_shape_quats[floor_dim+primitive_dim:floor_dim+2*primitive_dim] = prev_prim_ori2

                shape_data = [prev_shape_positions, shape_shape_quats, scene_params]
                shape_gt_data = [prev_shape_gt_positions, shape_shape_quats, scene_params]

                store_data(data_names, shape_data, os.path.join(rollout_path, 'shape_' + str(j - 1) + '.h5'))
                store_data(data_names, shape_gt_data, os.path.join(rollout_path, 'shape_gt_' + str(j - 1) + '.h5'))

            positions = update_position(n_shapes, prim_pos, pts=selected_points, floor=floor_pos, n_points=n_points)
            gt_positions = update_position(n_shapes, prim_pos, pts=gt_pos, floor=floor_pos, n_points=n_points)
            
            prev_positions = positions
            prev_gt_positions = gt_positions
            prev_prim_ori1 = prim_rot[0]
            prev_prim_ori2 = prim_rot[1]

        emd_loss_array = np.stack(emd_loss_list)
        chamfer_loss_array = np.stack(chamfer_loss_list)
        print(f"EMD: {np.mean(emd_loss_array)} +- {np.std(emd_loss_array)}")
        print(f"Chamfer: {np.mean(chamfer_loss_array)} +- {np.std(chamfer_loss_array)}")
        plt_render([np.array(all_gt_positions), np.array(all_positions)], n_points, os.path.join(rollout_path, 'plt.gif'))


if __name__ == '__main__':
    main()
