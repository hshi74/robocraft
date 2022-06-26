import glob
import numpy as np
import os
import torch

from config import gen_args
from model import Model, EarthMoverLoss, ChamferLoss, HausdorffLoss
from utils import set_seed, Tee, count_parameters
from utils import load_data, get_env_group, get_scene_info, prepare_input
from visualize import train_plot_curves, eval_plot_curves, eval_plot_curves, plt_render

use_gpu = True
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def evaluate(args):
    set_seed(args.random_seed)
    args.outf = os.path.dirname(args.model_path)

    eval_out_path = os.path.join(args.outf, 'eval')
    os.system('mkdir -p ' + os.path.join(eval_out_path, 'plot'))
    os.system('mkdir -p ' + os.path.join(eval_out_path, 'render'))

    tee = Tee(os.path.join(eval_out_path, 'eval.log'), 'w')

    data_names = args.data_names

    # create model and load weights
    model = Model(args, use_gpu)
    print("model_kp #params: %d" % count_parameters(model))

    print("Loading network from %s" % args.model_path)

    if args.stage == 'dy':
        pretrained_dict = torch.load(args.model_path, map_location=device)
        model_dict = model.state_dict()
        # only load parameters in dynamics_predictor
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() \
            if 'dynamics_predictor' in k and k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        AssertionError("Unsupported stage %s, using other evaluation scripts" % args.stage)

    model.eval()

    model = model.to(device)

    emd_loss = EarthMoverLoss()
    chamfer_loss = ChamferLoss()
    h_loss = HausdorffLoss()

    loss_list_over_episodes = []

    for idx_episode in range(args.n_rollout):
        loss_list = []

        print("Rollout %d / %d" % (idx_episode, args.n_rollout))

        n_particle, n_shape = 0, 0

        # load data
        gt_data_list = []
        data_list = []
        p_gt = []
        p_sample = []
        frame_list = sorted(glob.glob(os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), 'shape_*.h5')))
        gt_frame_list = sorted(glob.glob(os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), 'shape_gt_*.h5')))
        args.time_step = (len(frame_list) - len(gt_frame_list))
        for step in range(args.time_step):
            gt_frame_name = 'gt_' + str(step) + '.h5'
            frame_name = str(step) + '.h5'
            if args.shape_aug:
                gt_frame_name = 'shape_' + gt_frame_name
                frame_name = 'shape_' + frame_name

            gt_data_path = os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), gt_frame_name)
            data_path = os.path.join(args.dataf, 'train', str(idx_episode).zfill(3), frame_name)

            try:
                gt_data = load_data(data_names, gt_data_path)
                load_gt = True
            except FileNotFoundError:
                load_gt = False
            
            data = load_data(data_names, data_path)

            if n_particle == 0 and n_shape == 0:
                n_particle, n_shape, scene_params = get_scene_info(data)
                scene_params = torch.FloatTensor(scene_params).unsqueeze(0)

            if args.verbose_data:
                print("n_particle", n_particle)
                print("n_shape", n_shape)

            if load_gt: 
                gt_data_list.append(gt_data)
            data_list.append(data)

            if load_gt: 
                p_gt.append(gt_data[0])

            new_state = data[0]

            p_sample.append(new_state)

        # p_sample: time_step x N x state_dim
        if load_gt: 
            p_gt = torch.FloatTensor(np.stack(p_gt))
        p_sample = torch.FloatTensor(np.stack(p_sample))
        p_pred = torch.zeros(args.time_step, n_particle + n_shape, args.state_dim)
        # initialize particle grouping
        group_info = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

        # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
        # for now, only used as a placeholder
        memory_init = model.init_memory(1, n_particle + n_shape)

        # model rollout
        # loss = 0.
        # loss_raw = 0.
        # loss_counter = 0
        st_idx = args.n_his
        ed_idx = args.time_step

        with torch.set_grad_enabled(False):
            for step_id in range(st_idx, ed_idx):
                # print(step_id)
                if step_id == st_idx:
                    if args.gt_particles:
                        # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
                        state_cur = p_gt[step_id - args.n_his:step_id]
                    else:
                        state_cur = p_sample[step_id - args.n_his:step_id]
                    state_cur = state_cur.to(device)

                # unsqueeze the batch dimension
                # attr: B x (n_p + n_s) x attr_dim
                # Rr_cur, Rs_cur: B x n_rel x (n_p + n_s)
                # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                attr, _, Rr_cur, Rs_cur, Rn_cur, cluster_onehot = prepare_input(state_cur[-1].cpu().numpy(), n_particle,
                                                                        n_shape, args, stdreg=args.stdreg)
                attr = attr.to(device).unsqueeze(0)
                Rr_cur = Rr_cur.to(device).unsqueeze(0)
                Rs_cur = Rs_cur.to(device).unsqueeze(0)
                Rn_cur = Rn_cur.to(device).unsqueeze(0)
                state_cur = state_cur.unsqueeze(0)
                if cluster_onehot:
                    cluster_onehot = cluster_onehot.unsqueeze(0)

                if args.stage in ['dy']:
                    inputs = [attr, state_cur, Rr_cur, Rs_cur, Rn_cur, memory_init, group_info, cluster_onehot]

                # pred_pos (unnormalized): B x n_p x state_dim
                # pred_motion_norm (normalized): B x n_p x state_dim
                if args.sequence_length > args.n_his + 1:
                    pred_pos_p, pred_motion_norm, std_cluster = model.predict_dynamics(inputs, (step_id - args.n_his))
                else:
                    pred_pos_p, pred_motion_norm, std_cluster = model.predict_dynamics(inputs)

                # concatenate the state of the shapes
                # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                sample_pos = p_sample[step_id].to(device).unsqueeze(0)
                sample_pos_p = sample_pos[:, :n_particle]
                pred_pos = torch.cat([pred_pos_p, sample_pos[:, n_particle:]], 1)

                # sample_motion_norm (normalized): B x (n_p + n_s) x state_dim
                # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                sample_motion = (p_sample[step_id] - p_sample[step_id - 1]).unsqueeze(0)
                sample_motion = sample_motion.to(device)

                mean_d, std_d = model.stat[2:]
                sample_motion_norm = (sample_motion - mean_d) / std_d
                pred_motion_norm = torch.cat([pred_motion_norm, sample_motion_norm[:, n_particle:]], 1)

                loss_emd = emd_loss(pred_pos_p, sample_pos_p)
                loss_chamfer = chamfer_loss(pred_pos_p, sample_pos_p)
                loss_h = h_loss(pred_pos_p, sample_pos_p)

                loss_list.append([step_id, loss_emd.item(), loss_chamfer.item(), loss_h.item()])
                # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
                state_cur = state_cur.detach()[0]

                # record the prediction
                p_pred[step_id] = state_cur[-1].detach().cpu()

        loss_list_over_episodes.append(loss_list)

        # visualization
        group_info = [d.data.cpu().numpy()[0, ...] for d in group_info]

        if args.gt_particles:
            p_pred = np.concatenate((p_gt.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))
        else:
            p_pred = np.concatenate((p_sample.numpy()[:st_idx], p_pred.numpy()[st_idx:ed_idx]))

        p_sample = p_sample.numpy()[:ed_idx]
        if load_gt: 
            p_gt = p_gt.numpy()[:ed_idx]

        # vid_path = os.path.join(args.dataf, 'vid', str(idx_episode).zfill(3))
        render_path = os.path.join(eval_out_path, 'render', f'vid_{idx_episode}_plt.gif')

        if args.vis == 'plt':
            plt_render([p_gt, p_sample, p_pred], n_particle, render_path)
        else:
            raise NotImplementedError

    # plot the loss curves for training and evaluating
    with open(os.path.join(args.outf, 'train.npy'), 'rb') as f:
        train_log = np.load(f, allow_pickle=True)
        train_log = train_log[None][0]
        train_plot_curves(train_log['iters'], train_log['loss'], path=os.path.join(eval_out_path, 'plot', 'train_loss_curves.png'))

    loss_list_over_episodes = np.array(loss_list_over_episodes)
    loss_mean = np.mean(loss_list_over_episodes, axis=0)
    loss_std = np.std(loss_list_over_episodes, axis=0)
    eval_plot_curves(loss_mean[:, :-1], loss_std[:, :-1], path=os.path.join(eval_out_path, 'plot', 'eval_loss_curves.png'))

    print(f"\nAverage emd loss at last frame: {np.mean(loss_list_over_episodes[:, -1, 1])} (+- {np.std(loss_list_over_episodes[:, -1, 1])})")
    print(f"Average chamfer loss at last frame: {np.mean(loss_list_over_episodes[:, -1, 2])} (+- {np.std(loss_list_over_episodes[:, -1, 2])})")
    print(f"Average hausdorff loss at last frame: {np.mean(loss_list_over_episodes[:, -1, 3])} (+- {np.std(loss_list_over_episodes[:, -1, 3])})")

    print(f"\nAverage emd loss over episodes: {np.mean(loss_list_over_episodes[:, :, 1])} (+- {np.std(loss_list_over_episodes[:, :, 1])})")
    print(f"Average chamfer loss over episodes: {np.mean(loss_list_over_episodes[:, :, 2])} (+- {np.std(loss_list_over_episodes[:, :, 2])})")
    print(f"Average hausdorff loss over episodes: {np.mean(loss_list_over_episodes[:, :, 3])} (+- {np.std(loss_list_over_episodes[:, :, 3])})")


if __name__ == '__main__':
    args = gen_args()
    evaluate(args)
