import numpy as np
import os
import sys
import torch

torch.set_default_tensor_type(torch.FloatTensor)

from dynamics.dy_utils import *
from dynamics.model import Model
from perception.pcd_utils import upsample
from pytorch3d.transforms import *
from transforms3d.axangles import axangle2mat
# from sim import Simulator
from config.config import gen_args
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


class GNN(object):
    def __init__(self, args, model_path):
        # load dynamics model
        self.args = args
        set_seed(args.random_seed)

        self.model = Model(args)
        # print("model_kp #params: %d" % count_parameters(self.model))

        self.device = args.device
        pretrained_dict = torch.load(model_path, map_location=self.device)

        model_dict = self.model.state_dict()
        
        # only load parameters in dynamics_predictor
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() \
            if 'dynamics_predictor' in k and k in model_dict
        }
        
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.eval()

        self.model = self.model.to(self.device)


    def prepare_shape(
        self,
        state_cur,      # [N, state_dim]
        init_pose_seqs, # [B, n_grip, n_shape, 14]
        act_seqs,       # [B, n_grip, n_steps, 12]
        rot_seqs,       # [B, n_grip]
    ):
        # preprocess the tensors
        if not torch.is_tensor(init_pose_seqs):
            init_pose_seqs = torch.tensor(init_pose_seqs)
        if not torch.is_tensor(act_seqs):
            act_seqs = torch.tensor(act_seqs)
        if not torch.is_tensor(state_cur):
            state_cur = torch.tensor(state_cur)

        init_pose_seqs = init_pose_seqs.float().to(self.device)
        act_seqs = act_seqs.float().to(self.device)
        
        if rot_seqs is not None:
            if not torch.is_tensor(rot_seqs):
                rot_seqs = torch.tensor(rot_seqs)
            rot_seqs = rot_seqs.float().to(self.device)
        
        B = init_pose_seqs.shape[0]
        if len(state_cur.shape) == 3:
            state_cur = expand(B, state_cur.float()).to(self.device)
        else:
            state_cur = expand(B, state_cur.float().unsqueeze(0)).to(self.device)

        if self.args.state_dim == 6:
            floor_state = torch.tensor(np.concatenate((self.args.floor_state, self.args.floor_normals), axis=1))
        else:
            floor_state = torch.tensor(self.args.floor_state)
        floor_state = expand(B, floor_state.float().unsqueeze(0)).to(self.device)

        return init_pose_seqs, act_seqs, rot_seqs, state_cur, floor_state


    def rotate_state(self, state_cur, theta, center_xy=[0.429, -0.008]):
        B = state_cur.shape[0]
        zero_padding = torch.zeros_like(theta, device=self.device)
        center_xy = torch.tensor(center_xy, device=self.device, dtype=torch.float32)
        center = torch.cat([torch.tile(center_xy, (B, 1, 1)), torch.mean(state_cur, dim=1)[:, 2:3].unsqueeze(1)], dim=2)

        state_rot = euler_angles_to_matrix(torch.stack((zero_padding, zero_padding, theta), dim=-1), 'XYZ')
        state_cur_new = (state_cur - center).bmm(state_rot) + center

        return state_cur_new


    # @profile
    def rollout(
        self,
        state_cur,      # [N, state_dim]
        init_pose_seqs, # [B, n_grip, n_shape, 14]
        act_seqs,       # [B, n_grip, n_steps, 12]
        rot_seqs=None,       # [B, n_grip]
    ):
        # reshape the tensors
        B = init_pose_seqs.shape[0]
        init_pose_seqs, act_seqs, rot_seqs, state_cur, floor_state = self.prepare_shape(
            state_cur, init_pose_seqs, act_seqs, rot_seqs)

        N = self.args.n_particles + sum(self.args.tool_dim[self.args.env]) + self.args.floor_dim
        memory_init = self.model.init_memory(init_pose_seqs.shape[0], N)
        group_gt = get_env_group(self.args, B)

        if self.args.batch_norm:
            mean_p, std_p, _, _ = compute_stats(self.args, state_cur)
        else:
            mean_p = torch.FloatTensor(self.args.mean_p).to(self.device)
            std_p = torch.FloatTensor(self.args.std_p).to(self.device)

        mean_d = torch.FloatTensor(self.args.mean_d).to(self.device)
        std_d = torch.FloatTensor(self.args.std_d).to(self.device)
        
        stats = [mean_p, std_p, mean_d, std_d]
        # print(stats)

        tool_start_idx = self.args.n_particles + self.args.floor_dim

        # object attributes
        attr_dim = 3
        attrs = torch.zeros((B, N, attr_dim), device=self.device)
        attrs[:, self.args.n_particles: tool_start_idx, 1] = 1
        # since the two tools in gripper_asym have different attributes
        
        if attr_dim == 3:
            attrs[:, tool_start_idx:, 2] = 1
        else:
            tool_idx = tool_start_idx
            for i in range(len(self.args.tool_dim[self.args.env])):
                tool_dim = self.args.tool_dim[self.args.env][i]
                attrs[:, tool_idx: tool_idx + tool_dim, i + 2] = 1
                tool_idx += tool_dim

        # rollout
        states_pred_list = []
        attn_mask_list = []
        t2p_rels_list = [[] for _ in range(B)]
        for i in range(act_seqs.shape[1]):
            rels_list_prev = None
            tool_pos_list = []
            tool_start = 0
            for k in range(len(self.args.tool_dim[self.args.env])):
                tool_dim = self.args.tool_dim[self.args.env][k]
                tool_pos = init_pose_seqs[:, i, tool_start:tool_start+tool_dim]
                if self.args.state_dim == 6:
                    tool_normals = get_normals(tool_pos, pkg='torch').to(self.device)
                    tool_pos_list.append(torch.cat((tool_pos, tool_normals), dim=2))
                else:
                    tool_pos_list.append(tool_pos)
                tool_start += tool_dim

            for j in range(0, act_seqs.shape[2], self.args.time_step):
                action_his_list = []
                for k in range(len(self.args.tool_dim[self.args.env])):
                    tool_his_list = [tool_pos_list[k].unsqueeze(1)]
                    for ii in range(j, j + self.args.n_his*self.args.time_step):
                        action_aug = act_seqs[:, i, ii:ii+1, 6*k:6*k+3].unsqueeze(2).expand(-1, -1, self.args.tool_dim[self.args.env][k], -1)
                        tool_pos = tool_pos_list[k][:, :, :3] + torch.sum(action_aug, dim=1)
                        if self.args.state_dim == 6:
                            tool_pos_list[k] = torch.cat((tool_pos, tool_pos_list[k][:, :, 3:]), dim=2)
                        else:
                            tool_pos_list[k] = tool_pos
                        tool_his_list.append(tool_pos_list[k].unsqueeze(1))

                    action_his_list.append(tool_his_list)

                if i == 0 and j == 0:
                    state_cur = state_cur[:, :self.args.n_particles, :3]
                else:
                    state_cur = pred_pos_p

                if j == 0 and rot_seqs is not None:
                    state_cur = self.rotate_state(state_cur, rot_seqs[:, i])

                if self.args.state_dim == 6:
                    normals_cur = get_normals(state_cur, pkg='torch').to(self.device)
                    state_cur = torch.cat([state_cur, normals_cur], dim=2)

                state_cur = torch.cat([state_cur, floor_state, *tool_pos_list], dim=1)
                action_his = torch.cat([torch.cat(x, dim=1) for x in action_his_list], dim=2)

                # import pdb; pdb.set_trace()
                Rr_curs, Rs_curs, rels_list_batch, t2p_rels_list_batch = prepare_input(
                    self.args, state_cur[:, :, :3].detach().cpu().numpy(), rels_list_prev=rels_list_prev, device=self.device)
                rels_list_prev = rels_list_batch
                # print(attrs.shape, state_cur.shape, Rr_curs.shape, Rs_curs.shape)
                inputs = [attrs, state_cur, action_his, Rr_curs, Rs_curs, memory_init, group_gt, stats]
                pred_pos_p, _, nrm_attn = self.model.predict_dynamics(inputs)

                states_pred_list.append(pred_pos_p)
                attn_mask_list.append(nrm_attn)
                for k in range(B):
                    t2p_rels_list[k].append(t2p_rels_list_batch[k])

        states_pred_array = torch.stack(states_pred_list, dim=1)
        attn_mask_arry = torch.stack(attn_mask_list, dim=1)
        # print(f"torch mem allocated: {torch.cuda.memory_allocated()}")

        return states_pred_array, attn_mask_arry, t2p_rels_list

