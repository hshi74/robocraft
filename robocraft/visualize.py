import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["legend.loc"] = 'lower right'


def train_plot_curves(iters, loss, path=''):
    plt.figure(figsize=[16,9])
    plt.plot(iters, loss)
    plt.xlabel('iterations', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Training Loss', fontsize=35)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()


def eval_plot_curves(loss_mean, loss_std, colors=['orange', 'royalblue'], 
    alpha_fill=0.3, ax=None, path=''):
    iters, loss_mean_emd, loss_mean_chamfer = loss_mean.T
    _, loss_std_emd, loss_std_chamfer = loss_std.T
    plt.figure(figsize=[16, 9])

    emd_min = loss_mean_emd - loss_std_emd
    emd_max = loss_mean_emd + loss_std_emd

    chamfer_min = loss_mean_chamfer - loss_std_chamfer
    chamfer_max = loss_mean_chamfer + loss_std_chamfer

    plt.plot(iters, loss_mean_emd, color=colors[0], linewidth=6, label='EMD')
    plt.fill_between(iters, emd_max, emd_min, color=colors[0], alpha=alpha_fill)

    plt.plot(iters, loss_mean_chamfer, color=colors[1], linewidth=6, label='Chamfer')
    plt.fill_between(iters, chamfer_max, chamfer_min, color=colors[1], alpha=alpha_fill)

    plt.xlabel('Time Steps', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.title('Dyanmics Model Evaluation Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)


    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()


def visualize_points(ax, all_points, n_particles):
    points = ax.scatter(all_points[:n_particles, 0], all_points[:n_particles, 2], all_points[:n_particles, 1], c='b', s=10)
    shapes = ax.scatter(all_points[n_particles+9:, 0], all_points[n_particles+9:, 2], all_points[n_particles+9:, 1], c='r', s=20)

    # ax.invert_yaxis()

    # mid_point = [0.5, 0.5, 0.1]
    # r = 0.25
    # ax.set_xlim(mid_point[0] - r, mid_point[0] + r)
    # ax.set_ylim(mid_point[1] - r, mid_point[1] + r)
    # ax.set_zlim(mid_point[2] - r, mid_point[2] + r)

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


    return points, shapes


def plt_render(particles_set, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    n_frames = particles_set[0].shape[0]
    rows = 3
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 9))
    row_titles = ['GT', 'Sample', 'Prediction']
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
                points._offsets3d = (
                states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (
                states[step, n_particle:, 0], states[step, n_particle:, 2], states[step, n_particle:, 1])
                outputs.append(points)
                outputs.append(shapes)
        return outputs

    anim = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), blit=False)

    # plt.show()
    anim.save(render_path, writer=animation.PillowWriter(fps=20))


def plt_render_frames_rm(particles_set, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    # pdb.set_trace()
    n_frames = particles_set[0].shape[0]
    rows = 2
    cols = 1

    fig, big_axes = plt.subplots(rows, 1, figsize=(3, 9))
    row_titles = ['Sample', 'Prediction']
    views = [(90, 90)]
    plot_info_all = {}
    for i in range(rows):
        states = particles_set[i]
        big_axes[i].set_title(row_titles[i], fontweight='semibold')
        big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.axis('off')
            ax.view_init(*views[j])
            points, shapes = visualize_points(ax, states[0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    for step in range(n_frames): # n_frames
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle+9:, 0], states[step, n_particle+9:, 2], states[step, n_particle+9:, 1])

        plt.tight_layout()
        plt.savefig(f'{render_path}/{str(step).zfill(3)}.pdf')


def plt_render_robot(particles_set, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    n_frames = particles_set[0].shape[0]
    rows = len(particles_set)
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, rows * 3))
    row_titles = ['Sample', 'Prediction']
    row_titles = row_titles[:rows]
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        if rows == 1: 
            big_axes.set_title(row_titles[i], fontweight='semibold')
            big_axes.axis('off')
        else:
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


def visualize_points_helper(ax, all_points, n_particles, p_color='b', alpha=1.0):
    points = ax.scatter(all_points[:n_particles, 0], all_points[:n_particles, 2], all_points[:n_particles, 1], c=p_color, s=10)
    shapes = ax.scatter(all_points[n_particles+9:, 0], all_points[n_particles+9:, 2], all_points[n_particles+9:, 1], c='r', s=20)

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = 0.25  # maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    # ax.invert_yaxis()

    return points, shapes


def plt_render_frames(particles_set, target_shape, n_particle, render_path):
    # particles_set[0] = np.concatenate((particles_set[0][:, :n_particle], particles_set[1][:, n_particle:]), axis=1)
    n_frames = particles_set[0].shape[0]
    rows = 1
    cols = 3

    fig, big_axes = plt.subplots(rows, 1, figsize=(9, 3))
    # plt.gca().invert_yaxis()
    row_titles = ['Simulator']
    # views = [(90, 90)]
    views = [(90, 90), (0, 90), (45, 135)]
    plot_info_all = {}
    for i in range(rows):
        states = particles_set[i]
        if rows == 1:
            big_axes.set_title(row_titles[i], fontweight='semibold')
            big_axes.axis('off')
        else:  
            big_axes[i].set_title(row_titles[i], fontweight='semibold')
            big_axes[i].axis('off')

        plot_info = []
        for j in range(cols):
            ax = fig.add_subplot(rows, cols, i * cols + j + 1, projection='3d')
            ax.axis('off')
            ax.view_init(*views[j])
            visualize_points_helper(ax, target_shape, n_particle, p_color='c', alpha=1.0)
            points, shapes = visualize_points_helper(ax, states[0], n_particle)
            plot_info.append((points, shapes))

        plot_info_all[row_titles[i]] = plot_info

    frame_list = [n_frames - 1]
    for g in range(n_frames // (task_params['len_per_grip'] + task_params['len_per_grip_back'])):
        frame_list.append(g * (task_params['len_per_grip'] + task_params['len_per_grip_back']) + 12)
        frame_list.append(g * (task_params['len_per_grip'] + task_params['len_per_grip_back']) + 15)
        frame_list.append(g * (task_params['len_per_grip'] + task_params['len_per_grip_back']) + task_params['len_per_grip'] - 1)

    for step in frame_list: # range(n_frames):
        for i in range(rows):
            states = particles_set[i]
            for j in range(cols):
                points, shapes = plot_info_all[row_titles[i]][j]
                points._offsets3d = (states[step, :n_particle, 0], states[step, :n_particle, 2], states[step, :n_particle, 1])
                shapes._offsets3d = (states[step, n_particle+9:, 0], states[step, n_particle+9:, 2], states[step, n_particle+9:, 1])

        plt.tight_layout()
        # plt.show()
        plt.savefig(f'{render_path}/{str(step).zfill(3)}.pdf')