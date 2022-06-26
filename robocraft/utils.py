import cv2
import glob
import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import open3d as o3d
import os
import random
import scipy
import sys
import time
import torch

from collections import defaultdict
from itertools import product
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.utils.data import Dataset


def em_distance(self, x, y):
    x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)  # x: [B, N, M, D]
    y_ = y[:, None, :, :].repeat(1, x.size(1), 1, 1)  # y: [B, N, M, D]
    dis = torch.norm(torch.add(x_, -y_), 2, dim=3)  # dis: [B, N, M]
    x_list = []
    y_list = []
    # x.requires_grad = True
    # y.requires_grad = True
    for i in range(dis.shape[0]):
        cost_matrix = dis[i].detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
        except:
            import pdb;
            pdb.set_trace()
        x_list.append(x[i, ind1])
        y_list.append(y[i, ind2])
        # x[i] = x[i, ind1]
        # y[i] = y[i, ind2]
    new_x = torch.stack(x_list)
    new_y = torch.stack(y_list)
    emd = torch.mean(torch.norm(torch.add(new_x, -new_y), 2, dim=2))
    return emd


def matched_motion(p_cur, p_prev, n_particles):
    x = p_cur[:, :n_particles, :]
    y = p_prev[:, :n_particles, :]

    x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)
    y_ = y[:, None, :, :].repeat(1, y.size(1), 1, 1)
    dis = torch.norm(torch.add(x_, -y_), 2, dim=3)
    x_list = []
    y_list = []
    for i in range(dis.shape[0]):
        cost_matrix = dis[i].detach().cpu().numpy()
        ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
        x_list.append(x[i, ind1])
        y_list.append(y[i, ind2])
    new_x = torch.stack(x_list)
    new_y = torch.stack(y_list)
    p_cur_new = torch.cat((new_x, p_cur[:, n_particles:, :]), dim=1)
    p_prev_new = torch.cat((new_y, p_prev[:, n_particles:, :]), dim=1)
    dist = torch.add(p_cur_new, -p_prev_new)
    return dist


def my_collate(batch):
    len_batch = len(batch[0])
    len_rel = 3

    ret = []
    for i in range(len_batch - len_rel - 1):
        d = [item[i] for item in batch]
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)

    # processing relations
    # R: B x seq_length x n_rel x (n_p + n_s)
    for i in range(len_rel):
        R = [item[-len_rel + i - 1] for item in batch]
        max_n_rel = 0
        seq_length, _, N = R[0].size()
        for j in range(len(R)):
            max_n_rel = max(max_n_rel, R[j].size(1))
        for j in range(len(R)):
            r = R[j]
            r = torch.cat([r, torch.zeros(seq_length, max_n_rel - r.size(1), N)], 1)
            R[j] = r

        R = torch.FloatTensor(torch.stack(R))

        ret.append(R)

    # std reg
    d = [item[-1] for item in batch]
    if d[0] is not None:
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)
    else:
        ret.append(None)
    return tuple(ret)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### from DPI

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


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + \
                (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def normalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.
            s = Variable(torch.FloatTensor(stat[i]).cuda())

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].size(1) / stat_dim)
            data[i] = data[i].view(-1, n_rep, stat_dim)

            data[i] = (data[i] - s[:, 0]) / s[:, 1]

            data[i] = data[i].view(-1, n_rep * stat_dim)

    else:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].shape[1] / stat_dim)
            data[i] = data[i].reshape((-1, n_rep, stat_dim))

            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]

            data[i] = data[i].reshape((-1, n_rep * stat_dim))

    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]

    return data


def calc_rigid_transform(XX, YY):
    X = XX.copy().T
    Y = YY.copy().T

    mean_X = np.mean(X, 1, keepdims=True)
    mean_Y = np.mean(Y, 1, keepdims=True)
    X = X - mean_X
    Y = Y - mean_Y
    C = np.dot(X, Y.T)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
    R = np.dot(Vt.T, np.dot(D, U.T))
    T = mean_Y - np.dot(R, mean_X)

    '''
    YY_fitted = (np.dot(R, XX.T) + T).T
    print("MSE fit", np.mean(np.square(YY_fitted - YY)))
    '''

    return R, T


def normalize_scene_param(scene_params, param_idx, param_range, norm_range=(-1, 1)):
    normalized = np.copy(scene_params[:, param_idx])
    low, high = param_range
    if low == high:
        return normalized
    nlow, nhigh = norm_range
    normalized = nlow + (normalized - low) * (nhigh - nlow) / (high - low)
    return normalized


def gen_PyFleX(info):
    env, env_idx = info['env'], info['env_idx']
    thread_idx, data_dir, data_names = info['thread_idx'], info['data_dir'], info['data_names']
    n_rollout, time_step = info['n_rollout'], info['time_step']
    shape_state_dim, dt = info['shape_state_dim'], info['dt']

    gen_vision = info['gen_vision']
    vision_dir, vis_width, vis_height = info['vision_dir'], info['vis_width'], info['vis_height']

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    # positions
    stats = [init_stat(3)]

    import pyflex
    pyflex.init()

    for i in range(n_rollout):

        if i % 10 == 0:
            print("%d / %d" % (i, n_rollout))

        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system('mkdir -p ' + rollout_dir)

        if env == 'RigidFall':
            g_low, g_high = info['physics_param_range']
            gravity = rand_float(g_low, g_high)
            print("Generated RigidFall rollout {} with gravity {} from range {} ~ {}".format(
                i, gravity, g_low, g_high))

            n_instance = 3
            draw_mesh = 1
            scene_params = np.zeros(n_instance * 3 + 3)
            scene_params[0] = n_instance
            scene_params[1] = gravity
            scene_params[-1] = draw_mesh

            low_bound = 0.09
            for j in range(n_instance):
                x = rand_float(0., 0.1)
                y = rand_float(low_bound, low_bound + 0.01)
                z = rand_float(0., 0.1)

                scene_params[j * 3 + 2] = x
                scene_params[j * 3 + 3] = y
                scene_params[j * 3 + 4] = z

                low_bound += 0.21

            pyflex.set_scene(env_idx, scene_params, thread_idx)
            pyflex.set_camPos(np.array([0.2, 0.875, 2.0]))

            n_particles = pyflex.get_n_particles()
            n_shapes = 1    # the floor

            positions = np.zeros((time_step, n_particles + n_shapes, 3), dtype=np.float32)
            shape_quats = np.zeros((time_step, n_shapes, 4), dtype=np.float32)

            for j in range(time_step):
                positions[j, :n_particles] = pyflex.get_positions().reshape(-1, 4)[:, :3]

                ref_positions = positions[0]

                for k in range(n_instance):
                    XX = ref_positions[64*k:64*(k+1)]
                    YY = positions[j, 64*k:64*(k+1)]

                    X = XX.copy().T
                    Y = YY.copy().T

                    mean_X = np.mean(X, 1, keepdims=True)
                    mean_Y = np.mean(Y, 1, keepdims=True)
                    X = X - mean_X
                    Y = Y - mean_Y
                    C = np.dot(X, Y.T)
                    U, S, Vt = np.linalg.svd(C)
                    D = np.eye(3)
                    D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
                    R = np.dot(Vt.T, np.dot(D, U.T))
                    t = mean_Y - np.dot(R, mean_X)

                    YY_fitted = (np.dot(R, XX.T) + t).T
                    # print("MSE fit", np.mean(np.square(YY_fitted - YY)))

                    positions[j, 64*k:64*(k+1)] = YY_fitted

                if gen_vision:
                    pyflex.step(capture=True, path=os.path.join(rollout_dir, str(j) + '.tga'))
                else:
                    pyflex.step()

                data = [positions[j], shape_quats[j], scene_params]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + '.h5'))

            if gen_vision:
                images = np.zeros((time_step, vis_height, vis_width, 3), dtype=np.uint8)
                for j in range(time_step):
                    img_path = os.path.join(rollout_dir, str(j) + '.tga')
                    img = scipy.misc.imread(img_path)[:, :, :3][:, :, ::-1]
                    img = cv2.resize(img, (vis_width, vis_height), interpolation=cv2.INTER_AREA)
                    images[j] = img
                    os.system('rm ' + img_path)

                store_data(['positions', 'images', 'scene_params'], [positions, images, scene_params],
                           os.path.join(vision_dir, str(rollout_idx) + '.h5'))

        elif env == 'MassRope':
            s_low, s_high = info['physics_param_range']
            stiffness = rand_float(s_low, s_high)
            print("Generated MassRope rollout {} with gravity {} from range {} ~ {}".format(
                i, stiffness, s_low, s_high))

            x = 0.
            y = 1.0
            z = 0.
            length = 0.7
            draw_mesh = 1.

            scene_params = np.array([x, y, z, length, stiffness, draw_mesh])

            pyflex.set_scene(env_idx, scene_params, 0)
            pyflex.set_camPos(np.array([0.13, 2.0, 3.2]))

            action = np.zeros(3)

            # the last particle is the pin, regarded as shape
            n_particles = pyflex.get_n_particles() - 1
            n_shapes = 1    # the mass at the top of the rope

            positions = np.zeros((time_step + 1, n_particles + n_shapes, 3), dtype=np.float32)
            shape_quats = np.zeros((time_step + 1, n_shapes, 4), dtype=np.float32)

            action = np.zeros(3)
            for j in range(time_step + 1):
                positions[j] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                if j >= 1:
                    # append the action (position of the pin) to the previous time step
                    positions[j - 1, -1, :] = positions[j, -1, :]

                ref_positions = positions[0]

                # apply rigid projection to the rigid object
                # cube: [0, 81)
                # rope: [81, 95)
                # pin: [95, 96)
                XX = ref_positions[:81]
                YY = positions[j, :81]

                X = XX.copy().T
                Y = YY.copy().T

                mean_X = np.mean(X, 1, keepdims=True)
                mean_Y = np.mean(Y, 1, keepdims=True)
                X = X - mean_X
                Y = Y - mean_Y
                C = np.dot(X, Y.T)
                U, S, Vt = np.linalg.svd(C)
                D = np.eye(3)
                D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
                R = np.dot(Vt.T, np.dot(D, U.T))
                t = mean_Y - np.dot(R, mean_X)

                YY_fitted = (np.dot(R, XX.T) + t).T

                positions[j, :81] = YY_fitted

                scale = 0.1
                action[0] += rand_float(-scale, scale) - positions[j, -1, 0] * 0.1
                action[2] += rand_float(-scale, scale) - positions[j, -1, 2] * 0.1

                if gen_vision:
                    pyflex.step(action * dt, capture=True, path=os.path.join(rollout_dir, str(j) + '.tga'))
                else:
                    pyflex.step(action * dt)

                if j >= 1:
                    data = [positions[j - 1], shape_quats[j - 1], scene_params]
                    store_data(data_names, data, os.path.join(rollout_dir, str(j - 1) + '.h5'))

            if gen_vision:
                images = np.zeros((time_step, vis_height, vis_width, 3), dtype=np.uint8)
                for j in range(time_step):
                    img_path = os.path.join(rollout_dir, str(j) + '.tga')
                    img = scipy.misc.imread(img_path)[:, :, :3][:, :, ::-1]
                    img = cv2.resize(img, (vis_width, vis_height), interpolation=cv2.INTER_AREA)
                    images[j] = img
                    os.system('rm ' + img_path)

                store_data(['positions', 'images', 'scene_params'], [positions, images, scene_params],
                           os.path.join(vision_dir, str(rollout_idx) + '.h5'))

        else:
            raise AssertionError("Unsupported env")

        # change dtype for more accurate stat calculation
        # only normalize positions
        datas = [positions[:time_step].astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0] * datas[j].shape[1]
            stats[j] = combine_stat(stats[j], stat)

    pyflex.clean()

    return stats


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def visualize_neighbors(anchors, queries, idx, neighbors, leaf=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # import pdb; pdb.set_trace()
    # green is the first thing in positions
    # red is all the neighbors
    # shaded is all the particles
    # blue is the ball
    if not leaf:
        ax.scatter(queries[-1, 0], queries[-1, 2], queries[-1, 1], c='b', s=80)
        ax.scatter(queries[-2, 0], queries[-2, 2], queries[-2, 1], c='b', s=80)
        ax.scatter(queries[-3, 0], queries[-3, 2], queries[-3, 1], c='g', s=80)
    # ax.scatter(queries[idx, 0], queries[idx, 1], queries[idx, 2], c='g', s=80)
    ax.scatter(anchors[neighbors, 0], anchors[neighbors, 2], anchors[neighbors, 1], c='r', s=80)
    ax.scatter(anchors[:, 0], anchors[:, 2], anchors[:, 1], alpha=0.2)
    axisEqual3D(ax)

    plt.show()


def find_relations_neighbor(pos, query_idx, anchor_idx, radius, order, var=False):
    if np.sum(anchor_idx) == 0:
        return []

    point_tree = scipy.spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    # for i in range(len(neighbors)):
    #     # import pdb; pdb.set_trace()
    #     visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])

    relations = []
    for i in range(len(neighbors)):
        count_neighbors = len(neighbors[i])
        if count_neighbors == 0:
            continue

        receiver = np.ones(count_neighbors, dtype=np.int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender], axis=1))

    return relations


def find_k_relations_neighbor(k, positions, query_idx, anchor_idx, radius, order, var=False):
    """
    Same as find_relations_neighbor except that each point is only connected to the k nearest neighbors

    For each particle, only take the first min_neighbor neighbors, where
    min_neighbor = minimum number of neighbors among all particle's numbers of neighbors
    """
    if np.sum(anchor_idx) == 0:
        return []

    pos = positions.data.cpu().numpy() if var else positions

    point_tree = scipy.spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    '''
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    '''

    relations = []
    min_neighbors = None
    for i in range(len(neighbors)):
        if min_neighbors is None:
            min_neighbors = len(neighbors[i])
        elif len(neighbors[i]) < min_neighbors:
            min_neighbors = len(neighbors[i])
        else:
            pass

    for i in range(len(neighbors)):
        receiver = np.ones(min_neighbors, dtype=np.int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i][:min_neighbors]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender], axis=1))

    return relations


def get_scene_info(data):
    """
    A subset of prepare_input() just to get number of particles
    for initialization of grouping
    """
    positions, shape_quats, scene_params = data
    n_shapes = shape_quats.shape[0]
    count_nodes = positions.shape[0]
    n_particles = count_nodes - n_shapes

    return n_particles, n_shapes, scene_params


def get_env_group(args, n_particles, scene_params, use_gpu=False):
    # n_particles (int)
    # scene_params: B x param_dim
    B = scene_params.shape[0]

    p_rigid = torch.zeros(B, args.n_instance)
    p_instance = torch.zeros(B, n_particles, args.n_instance)
    physics_param = torch.zeros(B, n_particles)

    if args.env == 'Pinch':
        norm_g = normalize_scene_param(scene_params, 1, args.physics_param_range)
        physics_param[:] = torch.FloatTensor(norm_g).view(B, 1)
        p_rigid[:] = 0
        for i in range(args.n_instance):
            p_instance[:, :, i] = 1
    elif args.env == 'Gripper':
        norm_g = normalize_scene_param(scene_params, 1, args.physics_param_range)
        physics_param[:] = torch.FloatTensor(norm_g).view(B, 1)
        p_rigid[:] = args.p_rigid
        for i in range(args.n_instance):
            p_instance[:, :, i] = 1

    elif args.env == 'RigidFall':
        norm_g = normalize_scene_param(scene_params, 1, args.physics_param_range)
        physics_param[:] = torch.FloatTensor(norm_g).view(B, 1)

        p_rigid[:] = 1

        for i in range(args.n_instance):
            p_instance[:, 64 * i:64 * (i + 1), i] = 1

    elif args.env == 'MassRope':
        norm_stiff = normalize_scene_param(scene_params, 4, args.physics_param_range)
        physics_param[:] = torch.FloatTensor(norm_stiff).view(B, 1)

        n_rigid_particle = 81

        p_rigid[:, 0] = 1
        p_instance[:, :n_rigid_particle, 0] = 1
        p_instance[:, n_rigid_particle:, 1] = 1

    else:
        raise AssertionError("Unsupported env")

    if use_gpu:
        p_rigid = p_rigid.cuda()
        p_instance = p_instance.cuda()
        physics_param = physics_param.cuda()

    # p_rigid: B x n_instance
    # p_instance: B x n_p x n_instance
    # physics_param: B x n_p
    return [p_rigid, p_instance, physics_param]


def prepare_input(positions, n_particle, n_shape, args, var=False, stdreg=0):
    # positions: (n_p + n_s) x 3

    verbose = args.verbose_data

    count_nodes = n_particle + n_shape

    cluster_onehot = None

    if verbose:
        print("prepare_input::positions", positions.shape)
        print("prepare_input::n_particle", n_particle)
        print("prepare_input::n_shape", n_shape)

    ### object attributes
    attr = np.zeros((count_nodes, args.attr_dim))

    ##### add env specific graph components
    rels = []
    rels2 = []
    if args.env == 'Pinch':
        attr[n_particle, 1] = 1
        attr[n_particle+1, 2] = 1
        pos = positions.data.cpu().numpy() if var else positions

        # floor to points
        dis = pos[:n_particle, 1] - pos[n_particle, 1]
        nodes = np.nonzero(dis < args.neighbor_radius)[0]
        # print('visualize floor neighbors')
        # visualize_neighbors(pos, pos, 0, nodes)
        # print(np.sort(dis)[:10])

        floor = np.ones(nodes.shape[0], dtype=np.int) * n_particle
        rels += [np.stack([nodes, floor], axis=1)]

        # to primitive
        disp = np.sqrt(np.sum((pos[:n_particle] - pos[n_particle+1])**2, 1))  #np.sqrt(np.sum((pos[:n_particle, :] - pos[n_particle+1, :])**2, axis=1))
        nodes = np.nonzero(disp < (args.neighbor_radius+0.02))[0]
        # print('visualize prim neighbors')
        # visualize_neighbors(pos, pos, 0, nodes)
        # print(np.sort(dis)[:10])
        prim = np.ones(nodes.shape[0], dtype=np.int) * (n_particle+1)
        rels += [np.stack([nodes, prim], axis=1)]

        """Start to do K-Means"""
        if stdreg:
            kmeans = KMeans(n_clusters=10, random_state=0).fit(pos[:n_particle])
            cluster_label = kmeans.labels_
            cluster_onehot = np.zeros((cluster_label.size, cluster_label.max() + 1))
            cluster_onehot[np.arange(cluster_label.size), cluster_label] = 1

    elif args.env == 'Gripper':
        if args.shape_aug:
            attr[n_particle: n_particle + 9, 1] = 1
            attr[n_particle + 9:, 2] = 1
            pos = positions.data.cpu().numpy() if var else positions
            # floor to points
            for ind in range(9):
                dis = pos[:n_particle, 1] - pos[n_particle+ind, 1]
                #np.linalg.norm(pos[:n_particle] - pos[n_particle + ind], 2, axis=1)
                nodes = np.nonzero(dis < args.neighbor_radius)[0]
                # print(nodes)
                # if ind == 8:
                #     import pdb; pdb.set_trace()
                #     visualize_neighbors(pos, pos, 0, nodes)
                floor = np.ones(nodes.shape[0], dtype=np.int) * (n_particle + ind)
                rels += [np.stack([nodes, floor], axis=1)]
                rels2 += [np.stack([nodes, floor], axis=1)]
            for ind in range(22):
                # to primitive
                disp1 = np.sqrt(np.sum((pos[:n_particle] - pos[n_particle + 9 + ind]) ** 2, 1))
                nodes1 = np.nonzero(disp1 < (args.neighbor_radius + args.gripper_extra_neighbor_radius))[0]
                # detect how many grippers touching
                nodes2 = np.nonzero(disp1 < args.neighbor_radius)[0]
                # print('visualize prim1 neighbors')
                # print(nodes1)
                # if ind == 15:
                    # import pdb; pdb.set_trace()
                # visualize_neighbors(pos, pos, 0, nodes1)
                prim1 = np.ones(nodes1.shape[0], dtype=np.int) * (n_particle + 9 + ind)
                rels += [np.stack([nodes1, prim1], axis=1)]
                prim2 = np.ones(nodes2.shape[0], dtype=np.int) * (n_particle + 9 + ind)
                rels2 += [np.stack([nodes2, prim2], axis=1)]

                # disp2 = np.sqrt(np.sum((pos[:n_particle, [0, 2]] - pos[n_particle + 2, [0, 2]]) ** 2, 1))
                # disp2 = np.sqrt(np.sum((pos[:n_particle] - pos[n_particle + 2]) ** 2, 1))
                # np.sqrt(np.sum((pos[:n_particle, :] - pos[n_particle+1, :])**2, axis=1))
                # nodes2 = np.nonzero(disp2 < (args.neighbor_radius + args.gripper_extra_neighbor_radius))[0]
                # print('visualize prim neighbors')
                # visualize_neighbors(pos, pos, 0, nodes)
                # print(np.sort(dis)[:10])
                # prim2 = np.ones(nodes2.shape[0], dtype=np.int) * (n_particle + 2)
                # rels += [np.stack([nodes2, prim2], axis=1)]
            # import pdb; pdb.set_trace()
        else:
            attr[n_particle, 1] = 1
            attr[n_particle + 1, 2] = 1
            attr[n_particle + 2, 2] = 1
            pos = positions.data.cpu().numpy() if var else positions

            # floor to points
            dis = pos[:n_particle, 1] - pos[n_particle, 1]
            nodes = np.nonzero(dis < args.neighbor_radius)[0]
            # print('visualize floor neighbors')
            # visualize_neighbors(pos, pos, 0, nodes)
            # print(np.sort(dis)[:10])

            floor = np.ones(nodes.shape[0], dtype=np.int) * n_particle
            rels += [np.stack([nodes, floor], axis=1)]

            # to primitive
            disp1 = np.sqrt(np.sum((pos[:n_particle, [0,2]] - pos[n_particle + 1, [0,2]]) ** 2, 1))
            # np.sqrt(np.sum((pos[:n_particle] - pos[n_particle + 1]) ** 2,1))
            # np.sqrt(np.sum((pos[:n_particle, :] - pos[n_particle+1, :])**2, axis=1))
            nodes1 = np.nonzero(disp1 < (args.neighbor_radius + args.gripper_extra_neighbor_radius))[0]
            # print('visualize prim1 neighbors')

            # print(args.neighbor_radius); import pdb; pdb.set_trace()
            # visualize_neighbors(pos, pos, 0, nodes1)
            # print(np.sort(dis)[:10])
            # print(np.sort(dis)[:10])
            prim1 = np.ones(nodes1.shape[0], dtype=np.int) * (n_particle + 1)
            rels += [np.stack([nodes1, prim1], axis=1)]

            disp2 = np.sqrt(np.sum((pos[:n_particle, [0,2]] - pos[n_particle + 2, [0,2]]) ** 2, 1))
            # disp2 = np.sqrt(np.sum((pos[:n_particle] - pos[n_particle + 2]) ** 2, 1))
            # np.sqrt(np.sum((pos[:n_particle, :] - pos[n_particle+1, :])**2, axis=1))
            nodes2 = np.nonzero(disp2 < (args.neighbor_radius + args.gripper_extra_neighbor_radius))[0]
            # print('visualize prim neighbors')
            # visualize_neighbors(pos, pos, 0, nodes)
            # print(np.sort(dis)[:10])
            prim2 = np.ones(nodes2.shape[0], dtype=np.int) * (n_particle + 2)
            rels += [np.stack([nodes2, prim2], axis=1)]


            """Start to do K-Means"""
            if stdreg:
                kmeans = KMeans(n_clusters=10, random_state=0).fit(pos[:n_particle])
                cluster_label = kmeans.labels_
                cluster_onehot = np.zeros((cluster_label.size, cluster_label.max() + 1))
                cluster_onehot[np.arange(cluster_label.size), cluster_label] = 1


    elif args.env == 'RigidFall':
        # object attr:
        # [particle, floor]
        attr[n_particle, 1] = 1
        pos = positions.data.cpu().numpy() if var else positions

        # conncetion between floor and particles when they are close enough
        dis = pos[:n_particle, 1] - pos[n_particle, 1]
        nodes = np.nonzero(dis < args.neighbor_radius)[0]
        '''
        if verbose:
            visualize_neighbors(pos, pos, 0, nodes)
            print(np.sort(dis)[:10])
        '''

        floor = np.ones(nodes.shape[0], dtype=np.int) * n_particle
        rels += [np.stack([nodes, floor], axis=1)]

    elif args.env == 'MassRope':
        pos = positions.data.cpu().numpy() if var else positions
        dis = np.sqrt(np.sum((pos[n_particle] - pos[:n_particle])**2, 1))
        nodes = np.nonzero(dis < args.neighbor_radius)[0]
        '''
        if verbose:
            visualize_neighbors(pos, pos, 0, nodes)
            print(np.sort(dis)[:10])
        '''

        pin = np.ones(nodes.shape[0], dtype=np.int) * n_particle
        rels += [np.stack([nodes, pin], axis=1)]
    else:
        AssertionError("Unsupported env %s" % args.env)

    ##### add relations between leaf particles

    if args.env in ['RigidFall', 'MassRope', 'Pinch', 'Gripper']:
        queries = np.arange(n_particle)
        anchors = np.arange(n_particle)

    rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
    rels2 += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
    # rels += find_k_relations_neighbor(args.neighbor_k, pos, queries, anchors, args.neighbor_radius, 2, var)

    if len(rels) > 0:
        rels = np.concatenate(rels, 0)

    if len(rels2) > 0:
        rels2 = np.concatenate(rels2, 0)

    if verbose:
        print("Relations neighbor", rels.shape)

    n_rel = rels.shape[0]
    Rr = torch.zeros(n_rel, n_particle + n_shape)
    Rs = torch.zeros(n_rel, n_particle + n_shape)
    Rr[np.arange(n_rel), rels[:, 0]] = 1
    Rs[np.arange(n_rel), rels[:, 1]] = 1

    Rn = torch.zeros(rels2.shape[0], n_particle + n_shape)
    Rn[np.arange(rels2.shape[0]), rels2[:, 1]] = 1

    if verbose:
        print("Object attr:", np.sum(attr, axis=0))
        print("Particle attr:", np.sum(attr[:n_particle], axis=0))
        print("Shape attr:", np.sum(attr[n_particle:n_particle + n_shape], axis=0))

    if verbose:
        print("Particle positions stats")
        print("  Shape", positions.shape)
        print("  Min", np.min(positions[:n_particle], 0))
        print("  Max", np.max(positions[:n_particle], 0))
        print("  Mean", np.mean(positions[:n_particle], 0))
        print("  Std", np.std(positions[:n_particle], 0))

    if var:
        particle = positions
    else:
        particle = torch.FloatTensor(positions)

    if verbose:
        for i in range(count_nodes - 1):
            if np.sum(np.abs(attr[i] - attr[i + 1])) > 1e-6:
                print(i, attr[i], attr[i + 1])

    attr = torch.FloatTensor(attr)
    if stdreg:
        cluster_onehot = torch.FloatTensor(cluster_onehot)
    else:
        cluster_onehot = None
    assert attr.size(0) == count_nodes
    assert attr.size(1) == args.attr_dim

    # attr: (n_p + n_s) x attr_dim
    # particle (unnormalized): (n_p + n_s) x state_dim
    # Rr, Rs: n_rel x (n_p + n_s)
    return attr, particle, Rr, Rs, Rn, cluster_onehot


def real_sim_remap(args, data, n_particle):
    points = data[0]
    # print(np.mean(points[:n_particle], axis=0), np.std(points[:n_particle], axis=0))
    points = (points - np.mean(points[:n_particle], axis=0)) / np.std(points[:n_particle], axis=0)
    points = np.array([points.T[0], points.T[2], points.T[1]]).T * np.array([0.06, args.std_p[1], 0.06]) \
        + np.array([0.5, args.mean_p[1], 0.5])

    n_shapes_floor = 9
    n_shapes_per_gripper = 11
    prim1 = points[n_particle + n_shapes_floor + 2] # + n_shapes_per_gripper // 2
    prim2 = points[n_particle + n_shapes_floor + n_shapes_per_gripper + 2]
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
    new_state = np.concatenate([points[:n_particle], new_floor, new_prim1, new_prim2])

    return new_state


class PhysicsFleXDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.vision_dir = self.data_dir + '_vision'
        self.stat_path = os.path.join(self.args.dataf, '..', 'stats.h5')

        self.n_frame_list = []
        vid_list = sorted(glob.glob(os.path.join(self.data_dir, '*')))
        # print(vid_list)
        for vid_idx in range(len(vid_list)):
            frame_list = sorted(glob.glob(os.path.join(vid_list[vid_idx], 'shape_*.h5')))
            gt_frame_list = sorted(glob.glob(os.path.join(vid_list[vid_idx], 'shape_gt_*.h5')))
            self.n_frame_list.append(len(frame_list) - len(gt_frame_list))
        print(f"#frames list: {self.n_frame_list}")

        if args.gen_data:
            os.system('mkdir -p ' + self.data_dir)
        if args.gen_vision:
            os.system('mkdir -p ' + self.vision_dir)

        if args.env in ['RigidFall', 'MassRope', 'Pinch', 'Gripper']:
            self.data_names = ['positions', 'shape_quats', 'scene_params']
        else:
            raise AssertionError("Unsupported env")

        ratio = self.args.train_valid_ratio
        if phase == 'train':
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == 'valid':
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

    def __len__(self):
        """
        Each data point is consisted of a whole trajectory
        """
        args = self.args
        return self.n_rollout * (args.time_step - args.sequence_length + 1)

    def load_data(self, name):
        print("Loading stat from %s ..." % self.stat_path)
        self.stat = load_data(self.data_names[:1], self.stat_path)

    def gen_data(self, name):
        # if the data hasn't been generated, generate the data
        print("Generating data ... n_rollout=%d, time_step=%d" % (self.n_rollout, self.args.time_step))

        infos = []
        for i in range(self.args.num_workers):
            info = {
                'env': self.args.env,
                'thread_idx': i,
                'data_dir': self.data_dir,
                'data_names': self.data_names,
                'n_rollout': self.n_rollout // self.args.num_workers,
                'time_step': self.args.time_step,
                'dt': self.args.dt,
                'shape_state_dim': self.args.shape_state_dim,
                'physics_param_range': self.args.physics_param_range,

                'gen_vision': self.args.gen_vision,
                'vision_dir': self.vision_dir,
                'vis_width': self.args.vis_width,
                'vis_height': self.args.vis_height}

            if self.args.env == 'RigidFall':
                info['env_idx'] = 3
            elif self.args.env == 'MassRope':
                info['env_idx'] = 9
            else:
                raise AssertionError("Unsupported env")

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)
        data = pool.map(gen_PyFleX, infos)

        print("Training data generated, wrapping up stats ...")

        if self.phase == 'train' and self.args.gen_stat:
            # positions [x, y, z]
            self.stat = [init_stat(3)]
            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])
            store_data(self.data_names[:1], self.stat, self.stat_path)
        else:
            print("Loading stat from %s ..." % self.stat_path)
            self.stat = load_data(self.data_names[:1], self.stat_path)

    def __getitem__(self, idx):
        """
        Load a trajectory of length sequence_length
        """
        args = self.args

        idx_curr = idx
        idx_rollout = 0
        offset = self.n_frame_list[idx_rollout] - args.sequence_length + 1
        while idx_curr >= offset:
            idx_curr -= offset
            idx_rollout = (idx_rollout + 1) % len(self.n_frame_list)
            offset = self.n_frame_list[idx_rollout] - args.sequence_length + 1
        
        # offset = args.time_step - args.sequence_length + 1
        # idx_rollout = idx // offset
        # st_idx = idx % offset
        st_idx = idx_curr
        ed_idx = st_idx + args.sequence_length

        if args.stage in ['dy']:
            # load ground truth data
            attrs, particles, Rrs, Rss, Rns, cluster_onehots= [], [], [], [], [], []
            # sdf_list = []
            max_n_rel = 0
            for t in range(st_idx, ed_idx):
                # load data
                if self.args.env == 'Pinch' or self.args.env == 'Gripper':
                    frame_name = str(t) + '.h5'

                    if self.args.gt_particles:
                        frame_name = 'gt_' + frame_name
                    if self.args.shape_aug:
                        frame_name = 'shape_' + frame_name
                        # data_path = os.path.join(self.data_dir, str(idx_rollout).zfill(3), 'gt_' + frame_name)
                    # else:
                    #     pass
                        # data_path = os.path.join(self.data_dir, str(idx_rollout).zfill(3), str(t) + '.h5')
                    data_path = os.path.join(self.data_dir, str(idx_rollout).zfill(3), frame_name)
                else:
                    data_path = os.path.join(self.data_dir, str(idx_rollout), str(t) + '.h5')
                data = load_data(self.data_names, data_path)
                # sdf_data = load_data(['sdf'], os.path.join(self.data_dir, str(idx_rollout).zfill(3), 'sdf_' + str(t) + '.h5')), 

                # load scene param
                if t == st_idx:
                    n_particle, n_shape, scene_params = get_scene_info(data)

                # attr: (n_p + n_s) x attr_dim
                # particle (unnormalized): (n_p + n_s) x state_dim
                # Rr, Rs: n_rel x (n_p + n_s)
                if 'robot' in args.data_type:
                    new_state = real_sim_remap(args, data, n_particle)
                else:
                    new_state = data[0]

                attr, particle, Rr, Rs, Rn, cluster_onehot = prepare_input(new_state, n_particle, n_shape, self.args, stdreg=self.args.stdreg)
                max_n_rel = max(max_n_rel, Rr.size(0))

                attrs.append(attr)
                particles.append(particle.numpy())
                Rrs.append(Rr)
                Rss.append(Rs)
                Rns.append(Rn)
                # sdf_data = np.array(sdf_data).squeeze()
                # print(np.array(sdf_data.shape)
                # sdf_list.append(sdf_data)

                if cluster_onehot is not None:
                    cluster_onehots.append(cluster_onehot)


        '''
        add augmentation
        '''
        if args.stage in ['dy']:
            for t in range(args.sequence_length):
                if t == args.n_his - 1:
                    # set anchor for transforming rigid objects
                    particle_anchor = particles[t].copy()

                if t < args.n_his:
                    # add noise to observation frames - idx smaller than n_his
                    noise = np.random.randn(n_particle, 3) * args.std_d * args.augment_ratio
                    particles[t][:n_particle] += noise

                else:
                    # for augmenting rigid object,
                    # make sure the rigid transformation is the same before and after augmentation
                    if args.env == 'RigidFall':
                        for k in range(args.n_instance):
                            XX = particle_anchor[64*k:64*(k+1)]
                            XX_noise = particles[args.n_his - 1][64*k:64*(k+1)]

                            YY = particles[t][64*k:64*(k+1)]

                            R, T = calc_rigid_transform(XX, YY)

                            particles[t][64*k:64*(k+1)] = (np.dot(R, XX_noise.T) + T).T

                            '''
                            # checking the correctness of the implementation
                            YY_noise = particles[t][64*k:64*(k+1)]
                            RR, TT = calc_rigid_transform(XX_noise, YY_noise)
                            print(R, T)
                            print(RR, TT)
                            '''

                    elif args.env == 'MassRope':
                        n_rigid_particle = 81

                        XX = particle_anchor[:n_rigid_particle]
                        XX_noise = particles[args.n_his - 1][:n_rigid_particle]
                        YY = particles[t][:n_rigid_particle]

                        R, T = calc_rigid_transform(XX, YY)

                        particles[t][:n_rigid_particle] = (np.dot(R, XX_noise.T) + T).T

                        '''
                        # checking the correctness of the implementation
                        YY_noise = particles[t][:n_rigid_particle]
                        RR, TT = calc_rigid_transform(XX_noise, YY_noise)
                        print(R, T)
                        print(RR, TT)
                        '''

        else:
            AssertionError("Unknown stage %s" % args.stage)


        # attr: (n_p + n_s) x attr_dim
        # particles (unnormalized): seq_length x (n_p + n_s) x state_dim
        # scene_params: param_dim
        # sdf_list: seq_length x 64 x 64 x 64
        attr = torch.FloatTensor(attrs[0])
        particles = torch.FloatTensor(np.stack(particles))
        scene_params = torch.FloatTensor(scene_params)
        # sdf_list = torch.FloatTensor(np.stack(sdf_list))

        # pad the relation set
        # Rr, Rs: seq_length x n_rel x (n_p + n_s)
        if args.stage in ['dy']:
            for i in range(len(Rrs)):
                Rr, Rs, Rn = Rrs[i], Rss[i], Rns[i]
                Rr = torch.cat([Rr, torch.zeros(max_n_rel - Rr.size(0), n_particle + n_shape)], 0)
                Rs = torch.cat([Rs, torch.zeros(max_n_rel - Rs.size(0), n_particle + n_shape)], 0)
                Rn = torch.cat([Rn, torch.zeros(max_n_rel - Rn.size(0), n_particle + n_shape)], 0)
                Rrs[i], Rss[i], Rns[i] = Rr, Rs, Rn
            Rr = torch.FloatTensor(np.stack(Rrs))
            Rs = torch.FloatTensor(np.stack(Rss))
            Rn = torch.FloatTensor(np.stack(Rns))
            if cluster_onehots:
                cluster_onehot = torch.FloatTensor(np.stack(cluster_onehots))
            else:
                cluster_onehot = None
        if args.stage in ['dy']:
            return attr, particles, n_particle, n_shape, scene_params, Rr, Rs, Rn, cluster_onehot


def p2g(x, size=64, p_mass=1.):
    if x.dim() == 2:
        x = x[None, :]
    batch = x.shape[0]
    grid_m = torch.zeros(batch, size * size * size, dtype=x.dtype, device=x.device)
    inv_dx = size
    # base = (self.x[f, p] * self.inv_dx - 0.5).cast(int)
    # fx = self.x[f, p] * self.inv_dx - base.cast(self.dtype)
    fx = x * inv_dx
    base = (x * inv_dx - 0.5).long()
    fx = fx - base.float()
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    #for offset in ti.static(ti.grouped(self.stencil_range())):
    #    weight = ti.cast(1.0, self.dtype)
    #    for d in ti.static(range(self.dim)):
    #        weight *= w[offset[d]][d]
    #    self.grid_m[base + offset] += weight * self.p_mass
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[i][..., 0] * w[j][..., 1] * w[k][..., 2] * p_mass
                target = (base + torch.tensor(np.array([i, j, k]), dtype=torch.long, device='cuda:0')).clamp(0, size-1)
                idx = (target[..., 0] * size + target[..., 1]) * size + target[..., 2]
                grid_m.scatter_add_(1, idx, weight)
    grid_m = (grid_m > 0.0001).float()
    return grid_m.reshape(batch, size, size, size)


def compute_sdf(density, eps=1e-4, inf=1e10):
    if density.dim() == 3:
        density = density[None, :, :]
    dx = 1./density.shape[1]
    with torch.no_grad():
        nearest_points = torch.stack(torch.meshgrid(
            torch.arange(density.shape[1]),
            torch.arange(density.shape[2]),
            torch.arange(density.shape[3]),
        ), axis=-1)[None, :].to(density.device).expand(density.shape[0], -1, -1, -1, -1) * dx
        mesh_points = nearest_points.clone()

        is_object = (density <= eps) * inf
        sdf = is_object.clone()

        for i in range(density.shape[1] * 2): # np.sqrt(1^2+1^2+1^2)
            for x, y, z in product(range(3), range(3), range(3)):
                if x + y + z == 0: continue
                def get_slice(a):
                    if a == 0: return slice(None), slice(None)
                    if a == 1: return slice(0, -1), slice(1, None)
                    return slice(1, None), slice(0, -1)
                f1, t1 = get_slice(x)
                f2, t2 = get_slice(y)
                f3, t3 = get_slice(z)
                fr = (slice(None), f1, f2, f3)
                to = (slice(None), t1, t2, t3)
                dist = ((mesh_points[to] - nearest_points[fr])**2).sum(axis=-1)**0.5
                dist += (sdf[fr] >= inf) * inf
                sdf_to = sdf[to]
                mask = (dist < sdf_to).float()
                sdf[to] = mask * dist + (1-mask) * sdf_to
                mask = mask[..., None]
                nearest_points[to] = (1-mask) * nearest_points[to] + mask * nearest_points[fr]
        return sdf

def p2v(xyz):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(xyz.shape[0]):
    pcd = o3d.geometry.PointCloud()
    # import pdb; pdb.set_trace()
    # print(xyz.shape)
    pcd.points = o3d.utility.Vector3dVector(xyz.cpu().numpy())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.04)
    # o3d.visualization.draw_geometries([voxel_grid])
    # data = voxel_grid.create_dense(origin=[0,0,0], color=[0,0,0], voxel_size=0.03, width=1, height=1, depth=1)
    my_voxel = np.zeros((32, 32, 32))
    for j, d in enumerate(voxel_grid.get_voxels()):
        # print(j)
        my_voxel[d.grid_index[0], d.grid_index[1], d.grid_index[2]] = 1
        # z, x, y = my_voxel.nonzero()
        # ax.scatter(x, y, z, c=z, alpha=1)
        # plt.show()
    return torch.from_numpy(my_voxel).cuda()


def alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape [B, M, D] points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """
    tetra = scipy.spatial.Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs 
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra.vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))
    
    # Find tetrahedrals
    tetras = tetra.vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles: 
        TrianglesDict[tuple(tri)] += 1
    Triangles = np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    # edges
    EdgeComb = np.array([(0, 1), (0, 2), (1, 2)])
    Edges = Triangles[:,EdgeComb].reshape(-1,2)
    Edges = np.sort(Edges,axis=1)
    Edges = np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)

    return Vertices


def farthest_point_sampling(points, K_ratio=0.5):
    K = int(K_ratio * points.shape[0])
    fp_idx = np.zeros(K, dtype=np.int)
    fp_idx[0] = np.random.randint(points.shape[0])
    distances = ((points[fp_idx[0]] - points)**2).sum(axis=1)
    for i in range(1, K):
        fp_idx[i] = np.argmax(distances)
        d = ((points[fp_idx[i]] - points)**2).sum(axis=1)
        distances = np.minimum(distances, d)
    return fp_idx
