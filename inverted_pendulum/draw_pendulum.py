import os
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np
import pandas


def draw_pendulum(theta=np.pi / 6, pend_color='b', show=False, friction_angles = None):
    theta += np.pi/2
    pend_len = 1
    sc = 1.2
    plt.figure(figsize=(8, 8))
    plt.plot([-sc * pend_len, sc * pend_len], [0, 0], 'k--', linewidth=0.5)
    plt.plot([0, 0], [-sc * pend_len, sc * pend_len], 'k--', linewidth=0.5)

    end_point_x = pend_len * np.cos(theta)
    end_point_y = pend_len * np.sin(theta)

    plt.plot([0, end_point_x], [0, end_point_y], color=pend_color, linewidth=4)
    hinge = patch.Circle((0, 0), radius=0.03, color=pend_color)
    weight = patch.Circle((end_point_x, end_point_y), radius=0.06, color=pend_color)
    if friction_angles is None:
        friction_angles = [[-2.0, -0.5], [0.5, 2.0]]
    friction_patches = []
    friction_patches_colors = ['lightgray', 'dimgray']
    labels = ['friction coefficient = x2', 'friction coefficient = x4']
    for ind, fang in enumerate(friction_angles):
        new_patch = patch.Wedge([0, 0], 2.0, np.degrees(fang[0] + np.pi/2), np.degrees(fang[1] + np.pi/2),
                                hatch='/', facecolor=friction_patches_colors[ind], label=labels[ind])
        friction_patches.append(new_patch)

    ax = plt.gca()

    for fpatch in friction_patches:
        ax.add_patch(fpatch)
    plt.legend(fontsize='medium')

    ax.add_patch(hinge)
    ax.add_patch(weight)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    if show:
        plt.show()
    return plt.gcf()


def animate_pendulum(theta_array:np.ndarray, filename = 'pendulum.gif', remove_stills=False):
    frames_folder = filename+'_frames'
    nframes = theta_array.size
    if not os.path.isdir(frames_folder):
        os.mkdir(frames_folder)

    repeat_last = 10
    for i in range(nframes):
        fig = draw_pendulum(theta_array[i])
        plt.savefig(frames_folder+'/'+str(i)+'.png', bbox_inches='tight')
        plt.close(fig)

    with imageio.get_writer(filename, mode='I', fps=15) as writer:
        for i in range(nframes):
            frame_name = frames_folder+'/'+str(i)+'.png'
            image = imageio.imread(frame_name)
            writer.append_data(image)
        for i in range(repeat_last):
            writer.append_data(image)
    # remove the folder with the frames
    if remove_stills:
        os.system("rm {}/*.png".format(frames_folder))
        os.system("rmdir {}/".format(frames_folder))


if __name__ == "__main__":
    filename = 'data/mpc_data/dataset_mpc_19.txt'
    df = pandas.read_csv(filename, sep=',', names=['theta', 'dot_theta', 'u', 'log'])
    theta_array = df['theta']
    animate_pendulum(theta_array, filename[:-4]+'.gif', remove_stills=False)
