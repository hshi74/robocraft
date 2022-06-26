import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import gen_args
from tqdm import tqdm
from utils import PhysicsFleXDataset
from utils import prepare_input, get_scene_info, get_env_group
from model import Model, ChamferLoss, EarthMoverLoss, HausdorffLoss
from utils import set_seed, AverageMeter, get_lr, Tee, count_parameters, my_collate, matched_motion

from eval import evaluate

args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf)

tee = Tee(os.path.join(args.outf, 'train.log'), 'w')

def main():
    ### training

    # load training data

    phases = ['train'] if args.valid == 0 else ['valid']
    datasets = {phase: PhysicsFleXDataset(args, phase) for phase in phases}

    for phase in phases:
        if args.gen_data:
            datasets[phase].gen_data(args.env)
        else:
            datasets[phase].load_data(args.env)

    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers,
        collate_fn=my_collate) for phase in phases}

    # create model and train
    use_gpu = torch.cuda.is_available()
    model = Model(args, use_gpu)

    print("model #params: %d" % count_parameters(model))


    # checkpoint to reload model from
    model_path = None

    # resume training of a saved model (if given)
    if args.resume == 0:
        print("Randomly initialize the model's parameters")

    elif args.resume == 1:
        model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (
            args.resume_epoch, args.resume_iter))
        print("Loading saved ckp from %s" % model_path)

        if args.stage == 'dy':
            pretrained_dict = torch.load(model_path)
            model_dict = model.state_dict()

            # only load parameters in dynamics_predictor
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() \
                if 'dynamics_predictor' in k and k in model_dict}
            model.load_state_dict(pretrained_dict, strict=False)


    # optimizer
    if args.stage == 'dy':
        params = model.dynamics_predictor.parameters()
    else:
        raise AssertionError("unknown stage: %s" % args.stage)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params, lr=args.lr, betas=(args.beta1, 0.999))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=0.9)
    else:
        raise AssertionError("unknown optimizer: %s" % args.optimizer)

    # reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

    # define loss
    chamfer_loss = ChamferLoss()
    emd_loss = EarthMoverLoss()
    h_loss = HausdorffLoss()

    if use_gpu:
        model = model.cuda()

    # log args
    print(vars(args))

    # start training
    st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
    best_valid_loss = np.inf

    training_stats = {'args':vars(args), 'loss':[], 'loss_raw':[], 'iters': [], 'loss_emd': [], 'loss_motion': []}

    rollout_epoch = -1
    rollout_iter = -1
    for epoch in range(st_epoch, args.n_epoch):

        for phase in phases:

            model.train(phase == 'train')

            meter_loss = AverageMeter()
            meter_loss_raw = AverageMeter()

            meter_loss_ref = AverageMeter()
            meter_loss_nxt = AverageMeter()

            meter_loss_param = AverageMeter()

            for i, data in enumerate(tqdm(dataloaders[phase], desc=f'Epoch {epoch}/{args.n_epoch}')):
                # each "data" is a trajectory of sequence_length time steps

                if args.stage == 'dy':
                    # attrs: B x (n_p + n_s) x attr_dim
                    # particles: B x seq_length x (n_p + n_s) x state_dim
                    # n_particles: B
                    # n_shapes: B
                    # scene_params: B x param_dim
                    # Rrs, Rss: B x seq_length x n_rel x (n_p + n_s)
                    attrs, particles, n_particles, n_shapes, scene_params, Rrs, Rss, Rns, cluster_onehots = data

                    if use_gpu:
                        attrs = attrs.cuda()
                        particles = particles.cuda()
                        # sdf_list = sdf_list.cuda()
                        Rrs, Rss, Rns = Rrs.cuda(), Rss.cuda(), Rns.cuda()
                        if cluster_onehots is not None:
                            cluster_onehots = cluster_onehots.cuda()

                    # statistics
                    B = attrs.size(0)
                    n_particle = n_particles[0].item()
                    n_shape = n_shapes[0].item()

                    # p_rigid: B x n_instance
                    # p_instance: B x n_particle x n_instance
                    # physics_param: B x n_particle
                    groups_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

                    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
                    # for now, only used as a placeholder
                    memory_init = model.init_memory(B, n_particle + n_shape)
                    loss = 0
                    for j in range(args.sequence_length - args.n_his):
                        with torch.set_grad_enabled(phase == 'train'):
                            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
                            if j == 0:
                                state_cur = particles[:, :args.n_his]
                                # Rrs_cur, Rss_cur: B x n_rel x (n_p + n_s)
                                Rr_cur = Rrs[:, args.n_his - 1]
                                Rs_cur = Rss[:, args.n_his - 1]
                                Rn_cur = Rns[:, args.n_his - 1]
                            else: # elif pred_pos.size(0) >= args.batch_size:
                                Rr_cur = []
                                Rs_cur = []
                                Rn_cur = []
                                max_n_rel = 0
                                for k in range(pred_pos.size(0)):
                                    _, _, Rr_cur_k, Rs_cur_k, Rn_cur_k, _ = prepare_input(pred_pos[k].detach().cpu().numpy(), n_particle, n_shape, args, stdreg=args.stdreg)
                                    Rr_cur.append(Rr_cur_k)
                                    Rs_cur.append(Rs_cur_k)
                                    Rn_cur.append(Rn_cur_k)
                                    max_n_rel = max(max_n_rel, Rr_cur_k.size(0))
                                for w in range(pred_pos.size(0)):
                                    Rr_cur_k, Rs_cur_k, Rn_cur_k = Rr_cur[w], Rs_cur[w], Rn_cur[w]
                                    Rr_cur_k = torch.cat([Rr_cur_k, torch.zeros(max_n_rel - Rr_cur_k.size(0), n_particle + n_shape)], 0)
                                    Rs_cur_k = torch.cat([Rs_cur_k, torch.zeros(max_n_rel - Rs_cur_k.size(0), n_particle + n_shape)], 0)
                                    Rn_cur_k = torch.cat([Rn_cur_k, torch.zeros(max_n_rel - Rn_cur_k.size(0), n_particle + n_shape)], 0)
                                    Rr_cur[w], Rs_cur[w], Rn_cur[w] = Rr_cur_k, Rs_cur_k, Rn_cur_k
                                Rr_cur = torch.FloatTensor(np.stack(Rr_cur))
                                Rs_cur = torch.FloatTensor(np.stack(Rs_cur))
                                Rn_cur = torch.FloatTensor(np.stack(Rn_cur))
                                if use_gpu:
                                    Rr_cur = Rr_cur.cuda()
                                    Rs_cur = Rs_cur.cuda()
                                    Rn_cur = Rn_cur.cuda()
                                state_cur = torch.cat([state_cur[:,-3:], pred_pos.detach().unsqueeze(1)], dim=1)


                            if cluster_onehots is not None:
                                cluster_onehot = cluster_onehots[:, args.n_his - 1]
                            else:
                                cluster_onehot = None
                            # predict the velocity at the next time step
                            inputs = [attrs, state_cur, Rr_cur, Rs_cur, Rn_cur, memory_init, groups_gt, cluster_onehot]

                            # pred_pos (unnormalized): B x n_p x state_dim
                            # pred_motion_norm (normalized): B x n_p x state_dim
                            pred_pos_p, pred_motion_norm, std_cluster = model.predict_dynamics(inputs, j)

                            # concatenate the state of the shapes
                            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                            gt_pos = particles[:, args.n_his + j]
                            gt_pos_p = gt_pos[:, :n_particle]
                            # gt_sdf = sdf_list[:, args.n_his]
                            pred_pos = torch.cat([pred_pos_p, gt_pos[:, n_particle:]], 1)

                            # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
                            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
                            # gt_motion_norm should match then calculate if matched_motion enabled
                            if args.matched_motion:
                                gt_motion = matched_motion(particles[:, args.n_his], particles[:, args.n_his - 1], n_particles=n_particle)
                            else:
                                gt_motion = particles[:, args.n_his] - particles[:, args.n_his - 1]

                            mean_d, std_d = model.stat[2:]
                            gt_motion_norm = (gt_motion - mean_d) / std_d
                            pred_motion_norm = torch.cat([pred_motion_norm, gt_motion_norm[:, n_particle:]], 1)
                            if args.loss_type == 'emd_chamfer_h':
                                if args.emd_weight > 0:
                                    emd_l = args.emd_weight * emd_loss(pred_pos_p, gt_pos_p)
                                    loss += emd_l
                                if args.chamfer_weight > 0:
                                    chamfer_l = args.chamfer_weight * chamfer_loss(pred_pos_p, gt_pos_p)
                                    loss += chamfer_l
                                if args.h_weight > 0:
                                    h_l = args.h_weight * h_loss(pred_pos_p, gt_pos_p)
                                    loss += h_l
                                # print(f"EMD: {emd_l.item()}; Chamfer: {chamfer_l.item()}; H: {h_l.item()}")
                            else:
                                raise NotImplementedError

                            if args.stdreg:
                                loss += args.stdreg_weight * std_cluster
                            loss_raw = F.l1_loss(pred_pos_p, gt_pos_p)

                            meter_loss.update(loss.item(), B)
                            meter_loss_raw.update(loss_raw.item(), B)

                if i % args.log_per_iter == 0:
                    print()
                    print('%s epoch[%d/%d] iter[%d/%d] LR: %.6f, loss: %.6f (%.6f), loss_raw: %.8f (%.8f)' % (
                        phase, epoch, args.n_epoch, i, len(dataloaders[phase]), get_lr(optimizer),
                        loss.item(), meter_loss.avg, loss_raw.item(), meter_loss_raw.avg))
                    print('std_cluster', std_cluster)
                    if phase == 'train':
                        training_stats['loss'].append(loss.item())
                        training_stats['loss_raw'].append(loss_raw.item())
                        training_stats['iters'].append(epoch * len(dataloaders[phase]) + i)
                    # with open(args.outf + '/train.npy', 'wb') as f:
                    #     np.save(f, training_stats)

                # update model parameters
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if phase == 'train' and i > 0 and ((epoch * len(dataloaders[phase])) + i) % args.ckp_per_iter == 0:
                    model_path = '%s/net_epoch_%d_iter_%d.pth' % (args.outf, epoch, i)
                    torch.save(model.state_dict(), model_path)
                    rollout_epoch = epoch
                    rollout_iter = i

            print('%s epoch[%d/%d] Loss: %.6f, Best valid: %.6f' % (
                phase, epoch, args.n_epoch, meter_loss.avg, best_valid_loss))

            with open(args.outf + '/train.npy','wb') as f:
                np.save(f, training_stats)

            if phase == 'valid' and not args.eval:
                scheduler.step(meter_loss.avg)
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg
                    torch.save(model.state_dict(), '%s/net_best.pth' % (args.outf))
    
    if args.eval and model_path is not None:
        args.model_path = model_path
        evaluate(args)

if __name__ == '__main__':
    main()