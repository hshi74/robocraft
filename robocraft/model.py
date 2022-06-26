import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import optimize
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        s = x.size()
        x = self.model(x.view(-1, s[-1]))
        return x.view(list(s[:-1]) + [-1])


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, res=None):
        s_x = x.size()

        if self.residual:
            s_res = res.size()

        x = self.linear(x.view(-1, s_x[-1]))

        if self.residual:
            x += res.view(-1, s_res[-1])

        x = self.relu(x).view(list(s_x[:-1]) + [-1])
        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        s_x = x.size()

        x = x.view(-1, s_x[-1])
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x).view(list(s_x[:-1]) + [-1])


class DynamicsPredictor(nn.Module):
    def __init__(self, args, residual=False, use_gpu=False):

        super(DynamicsPredictor, self).__init__()

        self.args = args

        n_his = args.n_his
        attr_dim = args.attr_dim
        state_dim = args.state_dim
        mem_dim = args.nf_effect * args.mem_nlayer

        nf_particle = args.nf_particle
        nf_relation = args.nf_relation
        nf_effect = args.nf_effect
        self.nf_effect = nf_effect

        self.use_gpu = use_gpu
        self.residual = residual

        self.quat_offset = torch.FloatTensor([1., 0., 0., 0.])
        if use_gpu:
            self.quat_offset = self.quat_offset.cuda()

        # ParticleEncoder
        input_dim = attr_dim + 1 + n_his * state_dim * 2 + mem_dim
        self.particle_encoder = Encoder(input_dim, nf_particle, nf_effect)

        # RelationEncoder
        self.relation_encoder = Encoder(input_dim * 2 + 1, nf_relation, nf_effect)

        # ParticlePropagator
        self.particle_propagator = Propagator(nf_effect * 2, nf_effect, self.residual)

        # RelationPropagator
        self.relation_propagator = Propagator(nf_effect * 3, nf_effect)

        # ParticlePredictor
        self.rigid_predictor = ParticlePredictor(nf_effect, nf_effect, 7)
        self.non_rigid_predictor = ParticlePredictor(nf_effect, nf_effect, state_dim)

    def rotation_matrix_from_quaternion(self, params):
        # params: (B * n_instance) x 4
        # w, x, y, z

        one = torch.ones(1, 1)
        zero = torch.zeros(1, 1)
        if self.use_gpu:
            one = one.cuda()
            zero = zero.cuda()

        # multiply the rotation matrix from the right-hand side
        # the matrix should be the transpose of the conventional one

        # Reference
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

        params = params / torch.norm(params, dim=1, keepdim=True)
        w, x, y, z = \
                params[:, 0].view(-1, 1, 1), params[:, 1].view(-1, 1, 1), \
                params[:, 2].view(-1, 1, 1), params[:, 3].view(-1, 1, 1)

        rot = torch.cat((
            torch.cat((one - y * y * 2 - z * z * 2, x * y * 2 + z * w * 2, x * z * 2 - y * w * 2), 2),
            torch.cat((x * y * 2 - z * w * 2, one - x * x * 2 - z * z * 2, y * z * 2 + x * w * 2), 2),
            torch.cat((x * z * 2 + y * w * 2, y * z * 2 - x * w * 2, one - x * x * 2 - y * y * 2), 2)), 1)

        # rot: (B * n_instance) x 3 x 3
        return rot

    # @profile
    def forward(self, inputs, stat, verbose=0, j=0):
        args = self.args
        verbose = args.verbose_model
        mean_p, std_p, mean_d, std_d = stat

        # attrs: B x N x attr_dim
        # state (unnormalized): B x n_his x N x state_dim
        # Rr_cur, Rs_cur: B x n_rel x N
        # memory: B x mem_nlayer x N x nf_memory
        # group:
        #   p_rigid: B x n_instance
        #   p_instance: B x n_particle x n_instance
        #   physics_param: B x n_particle
        # cluster: num_of_particles (N) x num_of_clusters (k)
        attrs, state, Rr_cur, Rs_cur, Rn_cur, memory, group, cluster = inputs
        p_rigid, p_instance, physics_param = group
        # Rr_cur_t, Rs_cur_t: B x N x n_rel
        Rr_cur_t = Rr_cur.transpose(1, 2).contiguous()
        Rs_cur_t = Rs_cur.transpose(1, 2).contiguous()

        # number of particles that need prediction
        B, N = attrs.size(0), attrs.size(1)
        n_p = p_instance.size(1)
        n_s = attrs.size(1) - n_p

        n_his = args.n_his
        state_dim = args.state_dim
        # state_norm (normalized): B x n_his x N x state_dim
        # [0, n_his - 1): state_residual
        # [n_his - 1, n_his): the current position
        state_res_norm = (state[:, 1:] - state[:, :-1] - mean_d) / std_d
        state_res_norm[:, :, :301, :] = 0
        state_cur_norm = (state[:, -1:] - mean_p) / std_p
        state_norm = torch.cat([state_res_norm, state_cur_norm], 1)
        # state_norm_t (normalized): B x N x (n_his * state_dim)
        state_norm_t = state_norm.transpose(1, 2).contiguous().view(B, N, n_his * state_dim)

        # add offset to center-of-mass for rigids to attr
        # offset: B x N x (n_his * state_dim)
        offset = torch.zeros(B, N, n_his * state_dim)
        if self.use_gpu:
            offset = offset.cuda()

        # p_rigid_per_particle: B x n_p x 1
        p_rigid_per_particle = torch.sum(p_instance * p_rigid[:, None, :], 2, keepdim=True)  # this is trying to keep both instance label and rigidity label

        # instance_center: B x n_instance x (n_his * state_dim)
        instance_center = p_instance.transpose(1, 2).bmm(state_norm_t[:, :n_p])
        instance_center /= torch.sum(p_instance, 1).unsqueeze(-1) + args.eps

        # c_per_particle: B x n_p x (n_his * state_dim)
        # particle offset: B x n_p x (n_his * state_dim)
        c_per_particle = p_instance.bmm(instance_center)
        c = (1 - p_rigid_per_particle) * state_norm_t[:, :n_p] + p_rigid_per_particle * c_per_particle
        offset[:, :n_p] = state_norm_t[:, :n_p] - c

        # memory_t: B x N x (mem_nlayer * nf_memory)
        # physics_param: B x N x 1
        # attrs: B x N x (attr_dim + 1 + n_his * state_dim + mem_nlayer * nf_memory)
        memory_t = memory.transpose(1, 2).contiguous().view(B, N, -1)
        physics_param_s = torch.zeros(B, n_s, 1)
        if self.use_gpu:
            physics_param_s = physics_param_s.cuda()
        physics_param = torch.cat([physics_param[:, :, None], physics_param_s], 1)
        attrs = torch.cat([attrs, physics_param, offset, memory_t], 2)

        # group info
        # g: B x N x n_instance
        g = p_instance
        g_s = torch.zeros(B, n_s, args.n_instance)
        if self.use_gpu:
            g_s = g_s.cuda()
        g = torch.cat([g, g_s], 1)

        # receiver_attr, sender_attr
        # attrs_r: B x n_rel x -1
        # attrs_s: B x n_rel x -1
        attrs_r = Rr_cur.bmm(attrs)
        attrs_s = Rs_cur.bmm(attrs)
        # receiver_state, sender_state
        # state_norm_r: B x n_rel x -1
        # state_norm_s: B x n_rel x -1
        state_norm_r = Rr_cur.bmm(state_norm_t)
        state_norm_s = Rs_cur.bmm(state_norm_t)

        # receiver_group, sender_group
        # group_r: B x n_rel x -1
        # group_s: B x n_rel x -1
        group_r = Rr_cur.bmm(g)
        group_s = Rs_cur.bmm(g)
        group_diff = torch.sum(torch.abs(group_r - group_s), 2, keepdim=True)

        # particle encode
        if verbose:
            print('attrs_r', attrs_r.shape, 'state_norm_r', state_norm_r.shape)
        particle_encode = self.particle_encoder(torch.cat([attrs, state_norm_t], 2))
        particle_effect = particle_encode
        if verbose:
            print("particle encode:", particle_encode.size())

        # calculate relation encoding
        relation_encode = self.relation_encoder(
            torch.cat([attrs_r, attrs_s, state_norm_r, state_norm_s, group_diff], 2))
        if verbose:
            print("relation encode:", relation_encode.size())

        for i in range(args.pstep):
            if verbose:
                print("pstep", i)

            # effect_r, effect_s: B x n_rel x nf
            effect_r = Rr_cur.bmm(particle_effect)
            effect_s = Rs_cur.bmm(particle_effect)

            # calculate relation effect
            # effect_rel: B x n_rel x nf
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2))
            if verbose:
                print("relation effect:", effect_rel.size())

            # calculate particle effect by aggregating relation effect
            # effect_rel_agg: B x N x nf
            effect_rel_agg = Rr_cur_t.bmm(effect_rel)

            # calculate particle effect
            # particle_effect: B x N x nf
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2),
                res=particle_effect)
            if verbose:
                 print("particle effect:", particle_effect.size())

        # non_rigid_motion: B x n_p x state_dim
        non_rigid_motion = self.non_rigid_predictor(particle_effect[:, :n_p].contiguous())

        # rigid motion
        # instance effect: B x n_instance x nf_effect
        n_instance = p_instance.size(2)
        instance_effect = p_instance.transpose(1, 2).bmm(particle_effect[:, :n_p])

        # rigid motion
        # instance_rigid_params: (B * n_instance) x 7
        instance_rigid_params = self.rigid_predictor(instance_effect).view(B * n_instance, 7)

        # R: (B * n_instance) x 3 x 3
        R = self.rotation_matrix_from_quaternion(instance_rigid_params[:, :4] + self.quat_offset)
        if verbose:
            print("Rotation matrix", R.size(), "should be (B x n_instance, 3, 3)")

        b = instance_rigid_params[:, 4:] * std_d + mean_d
        b = b.view(B * n_instance, 1, state_dim)
        if verbose:
            print("b", b.size(), "should be (B x n_instance, 1, state_dim)")

        p_0 = state[:, -1:, :n_p]
        p_0 = p_0.repeat(1, n_instance, 1, 1).view(B * n_instance, n_p, state_dim)
        if verbose:
            print("p_0", p_0.size(), "should be (B x n_instance, n_p, state_dim)")

        c = instance_center[:, :, -3:] * std_p + mean_p
        c = c.view(B * n_instance, 1, state_dim)
        if verbose:
            print("c", c.size(), "should be (B x n_instance, 1, state_dim)")

        p_1 = torch.bmm(p_0 - c, R) + b + c
        if verbose:
            print("p_1", p_1.size(), "should be (B x n_instance, n_p, state_dim)")

        # rigid_motion: B x n_instance x n_p x state_dim
        rigid_motion = (p_1 - p_0).view(B, n_instance, n_p, state_dim)
        rigid_motion = (rigid_motion - mean_d) / std_d

        """regularize non regid motion within a cluster"""
        if cluster is not None:
            X = non_rigid_motion.transpose(1,2).bmm(cluster)
            X2 = torch.square(non_rigid_motion).transpose(1,2).bmm(cluster)
            n_count = cluster.sum(1).unsqueeze(1).repeat(1,3,1)  #counter.transpose(1,2).bmm(cluster)
            variance = (X2 / n_count)  - (X / n_count)**2
            std = torch.sqrt(variance + 1e-6)
            mean_std = torch.mean(std)
        else:
            mean_std = None

        n_gripper_touch = torch.zeros(B)
        if self.use_gpu:
            n_gripper_touch = n_gripper_touch.cuda()

        if self.args.stage == 'dy':
            neighbors = Rs_cur
        elif self.args.stage == 'control':
            neighbors = Rn_cur
        else:
            raise NotImplementedError

        n_gripper_touch[torch.count_nonzero(neighbors[:, :, 310:321], dim=(1, 2)) > 0] += 1
        n_gripper_touch[torch.count_nonzero(neighbors[:, :, 321:], dim=(1, 2)) > 0] += 1
        
        do_stationary = n_gripper_touch == 0
        do_rigid = n_gripper_touch == 1
        do_non_rigid = n_gripper_touch == 2
        # print(n_gripper_touch)

        pred_motion = torch.zeros(B, n_p, state_dim)
        if self.use_gpu:
            pred_motion = pred_motion.cuda()
        
        stationary_motion = (pred_motion - mean_d) / std_d
        
        # pdb.set_trace()
        if 'fixed' in self.args.data_type:
            pred_motion = non_rigid_motion
            pred_motion[do_stationary] = stationary_motion[do_stationary]
        else:
            pred_motion[do_stationary] = stationary_motion[do_stationary]
            pred_motion[do_non_rigid] = non_rigid_motion[do_non_rigid]
            pred_motion[do_rigid] = torch.sum(p_instance.transpose(1, 2)[:, :, :, None] * rigid_motion, 1)[do_rigid]

        pred_pos = state[:, -1, :n_p] + torch.clamp(pred_motion * std_d + mean_d, max=0.025, min=-0.025)
        # print(pred_motion * std_d + mean_d)
        # print(torch.max(torch.norm(torch.clamp(pred_motion * std_d + mean_d, max=0.025, min=-0.025), dim=2)))
        if verbose:
            print('pred_pos', pred_pos.size())

        # pred_pos (unnormalized): B x n_p x state_dim
        # pred_motion_norm (normalized): B x n_p x state_dim
        return pred_pos, pred_motion, mean_std   #, torch.sum(torch.mean(torch.stack(total_std, 0), 0))



class Model(nn.Module):
    def __init__(self, args, use_gpu):

        super(Model, self).__init__()

        self.args = args
        self.use_gpu = use_gpu

        self.dt = torch.FloatTensor([args.dt])
        mean_p = torch.FloatTensor(args.mean_p)
        std_p = torch.FloatTensor(args.std_p)
        mean_d = torch.FloatTensor(args.mean_d)
        std_d = torch.FloatTensor(args.std_d)

        if use_gpu:
            self.dt = self.dt.cuda()
            mean_p = mean_p.cuda()
            std_p = std_p.cuda()
            mean_d = mean_d.cuda()
            std_d = std_d.cuda()

        self.stat = [mean_p, std_p, mean_d, std_d]

        # PropNet to predict forward dynamics
        self.dynamics_predictor = DynamicsPredictor(args, use_gpu=use_gpu)

    def init_memory(self, B, N):
        """
        memory  (B, mem_layer, N, nf_memory)
        """
        mem = torch.zeros(B, self.args.mem_nlayer, N, self.args.nf_effect)
        if self.use_gpu:
            mem = mem.cuda()
        return mem

    def predict_dynamics(self, inputs, j=0):
        """
        return:
        ret - predicted position of all particles, shape (n_particles, 3)
        """
        ret = self.dynamics_predictor(inputs, self.stat, self.args.verbose_model, j=j)
        return ret


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x = x[:, :, None, :].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
        y = y[:, None, :, :].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
        dis_xy = torch.mean(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=1)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.chamfer_distance(pred, label)


class EarthMoverLoss(torch.nn.Module):
    def __init__(self):
        super(EarthMoverLoss, self).__init__()

    def em_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
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
                # pdb.set_trace()
                print("Error in linear sum assignment!")
            x_list.append(x[i, ind1])
            y_list.append(y[i, ind2])
            # x[i] = x[i, ind1]
            # y[i] = y[i, ind2]
        new_x = torch.stack(x_list)
        new_y = torch.stack(y_list)
        # print(f"EMD new_x shape: {new_x.shape}")
        # print(f"MAX: {torch.max(torch.norm(torch.add(new_x, -new_y), 2, dim=2))}")
        emd = torch.mean(torch.norm(torch.add(new_x, -new_y), 2, dim=2))
        return emd

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.em_distance(pred, label)


class HausdorffLoss(torch.nn.Module):
    def __init__(self):
        super(HausdorffLoss, self).__init__()

    def hausdorff_distance(self, x, y):
        # x: [B, N, D]
        # y: [B, M, D]
        x = x[:, :, None, :].repeat(1, 1, y.size(1), 1) # x: [B, N, M, D]
        y = y[:, None, :, :].repeat(1, x.size(1), 1, 1) # y: [B, N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=3)    # dis: [B, N, M]
        # print(dis.shape)
        dis_xy = torch.max(torch.min(dis, dim=2)[0])   # dis_xy: mean over N
        dis_yx = torch.max(torch.min(dis, dim=1)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        # pred: [B, N, D]
        # label: [B, M, D]
        return self.hausdorff_distance(pred, label)
