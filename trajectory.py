

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('tkagg')

# ------------------------------------------------------------------------

def avoid_wall(position, hd, min_x, max_x, min_y, max_y, border_region):
    '''
    Compute distance and angle to nearest wall
    '''
    x = position[:, 0]
    y = position[:, 1]
    dists = [max_x - x, max_y - y, x - min_x, y - min_y]
    d_wall = np.min(dists, axis=0)
    angles = np.arange(4) * np.pi / 2
    theta = angles[np.argmin(dists, axis=0)]
    hd = np.mod(hd, 2 * np.pi)
    a_wall = hd - theta
    a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

    is_near_wall = (d_wall < border_region) * (np.abs(a_wall) < np.pi / 2)
    turn_angle = np.zeros_like(hd)
    turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (np.pi / 2 - np.abs(a_wall[is_near_wall]))

    return is_near_wall, turn_angle

# ------------------------------------------------------------------------

def trajectory_builder(samples, batch_size, min_x, max_x, min_y, max_y,
                        dt=0.02, start_x=None, start_y=None, position_pred_start_idx=2):

    '''Build a random walk in a rectangular box, with given inputs'''

    sigma = 5.76 * 2        # stdev rotation velocity (rads/sec)
    b = 0.13 * 2 * np.pi    # forward velocity rayleigh dist scale (m/sec)
    mu = 0                  # turn angle bias
    border_region = 0.03    # meters

    # Initialize variables
    position = np.zeros([batch_size, samples + 2, 2])
    head_dir = np.zeros([batch_size, samples + 2])
    if start_x is None:
        position[:, 0, 0] = np.random.uniform(low=min_x, high=max_x, size=batch_size)
    else:
        assert (len(start_x.shape) == 1)
        assert (start_x.shape[0] == batch_size)
        position[:, 0, 0] = start_x

    # at t = 0, we start in a random position and head direction, for each episode
    # we have batch_size total episodes
    if start_y is None:
        position[:, 0, 1] = np.random.uniform(low=min_y, high=max_y, size=batch_size)
    else:
        assert (len(start_y.shape) == 1)
        assert (start_y.shape[0] == batch_size)
        position[:, 0, 1] = start_y

    head_dir[:, 0] = np.random.uniform(low=0, high=2 * np.pi, size=batch_size)
    # Generate sequence of random boosts and turns
    random_turn = np.random.normal(mu, sigma, [batch_size, samples + 1])

    # at t = 0, we start at 0 velocity
    velocity = np.zeros([batch_size, samples + 2])

    random_vel = np.random.rayleigh(b, [batch_size, samples + 1])
    v = np.abs(np.random.normal(0, b * np.pi / 2, batch_size))

    for t in range(samples + 1):
        # Update velocity
        v = random_vel[:, t]
        turn_angle = np.zeros(batch_size)

        # If in border region, turn and slow down
        is_near_wall, turn_angle = avoid_wall(position=position[:, t],
                                              hd=head_dir[:, t],
                                              min_x=min_x, max_x=max_x,
                                              min_y=min_y, max_y=max_y, border_region=border_region)
        v[is_near_wall] *= 0.25

        # Update turn angle
        turn_angle += dt * random_turn[:, t]

        # Take a step
        velocity[:, t] = v * dt
        update = velocity[:, t, None] * np.stack([np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1)
        position[:, t + 1] = position[:, t] + update

        # Rotate head direction
        head_dir[:, t + 1] = head_dir[:, t] + turn_angle

    head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi  # Periodic variable

    assert (position_pred_start_idx >= 1)

    traj = {}
    # Input variables
    # we get head direction at t = 0
    # since we compute angular velocity from it,
    # whose first element is hd at t = 1 - hd at t = 0
    traj['init_hd'] = head_dir[:, 0, None]
    # we get the first position after moving one step
    # with nonzero velocity (since at t=0 our velocity is 0)
    #traj['init_x'] = position[:, position_pred_start_idx - 1, 0, None]
    #traj['init_y'] = position[:, position_pred_start_idx - 1, 1, None]
    traj['init_pos'] = np.concatenate([position[:, position_pred_start_idx - 1, 0, None],
                                      position[:, position_pred_start_idx - 1, 1, None]], axis=1)

    # get the first nonzero velocity
    # up to second to last velocity, which will be our input
    # since we predict position as our target is 1 timestep ahead of input
    #traj['ego_v'] = velocity[:, position_pred_start_idx - 1:-1]
    ang_v = np.diff(head_dir, axis=-1)
    bs = position.shape[0]
    #traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:,:-1], np.sin(ang_v)[:,:-1]
    traj['ego_vel'] = np.concatenate([velocity[:, position_pred_start_idx - 1:-1].reshape(bs, -1, 1),
                                      np.cos(ang_v)[:, :-1].reshape(bs, -1, 1),
                                      np.sin(ang_v)[:, :-1].reshape(bs, -1, 1)], axis=2)

    # Target variables
    traj['target_hd'] = head_dir[:, position_pred_start_idx - 1:-1].reshape(bs, -1, 1)
    #traj['target_x'] = position[:, position_pred_start_idx:, 0]
    #traj['target_y'] = position[:, position_pred_start_idx:, 1]
    traj['target_pos'] = position[:, position_pred_start_idx:, 0:2]

    return traj, position

# ------------------------------------------------------------------------

def graph_trajectories(traj, num_recs_per_plot):

    transforms = traj['target_pos']

    bs = 1.0

    num_rec = transforms.shape[0]-1
    num_plots = (num_rec + (num_recs_per_plot-1)) // num_recs_per_plot

    ncols = min(10, num_plots)
    nrows = ((num_plots+(ncols-1)) // ncols)

    ws = 21.0
    ebs = bs*1.25

    fig, ax = plt.subplots(nrows, ncols, figsize=(ws, ws * (nrows/ncols)))

    gx = ncols
    gy = -1

    for i in range(num_rec):

        if i % num_recs_per_plot == 0:
            gx += 1
            if gx >= ncols:
                gy += 1
                gx = 0

        if nrows == 1 and ncols == 1:
            ax_ = ax
        else:
            ax_ = ax[gy, gx] if nrows > 1 else ax[gx]

        b_line = np.array([[-bs, -bs, 0], [bs, -bs, 0], [bs, bs, 0], [-bs, bs, 0], [-bs, -bs, 0]], dtype=np.float32)
        ax_.plot(b_line[:, 0], b_line[:, 1])

        trajectory = transforms[i]

        ax_.set_xlim((-ebs, ebs))
        ax_.set_ylim((-ebs, ebs))
        ax_.plot(trajectory[:, 0], trajectory[:, 1])

    plt.tight_layout()
    plt.show()
