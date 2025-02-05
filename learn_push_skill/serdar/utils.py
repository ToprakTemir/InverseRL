import mujoco
import numpy as np
from scipy.interpolate import make_splprep
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

hover_dist_object = 0.2
hover_dist_box = 0.3

def generate_quat_traj_for_box_pick(num_step):
    target_quat = [0, 0.707, 0.707, 0]
    source_quat = [0, 0, 1, 0]

    q = R.from_quat([source_quat,target_quat])

    slerp = Slerp([0, 1],  q)

    quat_traj = [slerp(i/num_step).as_quat() for i in range(num_step+1)]

    return quat_traj


def generate_traj_for_flip():

    source_quat = [0, -0.707, -0.707, 0]
    target_quat = [0, -0.4871745, 0, -0.8733046]

    q = R.from_quat([source_quat,target_quat])

    slerp = Slerp([0, 1],  q)

    num_step = 5000

    quat_traj = [slerp(i/num_step).as_quat() for i in range(num_step+1)]

    return quat_traj


def generate_traj_for_pick_object(model, data):
    object_id = model.body("object").id
    end_eff_id = model.body("end_effector").id

    traj = []

    source = data.xpos[object_id] + [0, 0, hover_dist_object]
    goal = source - [0, 0, hover_dist_object + 0.03]

    num_step = 1000

    for i in range(num_step):
        traj.append(source)

    for i in range(num_step+1):
        traj.append((goal-source)*(i/num_step) + source)

    for i in range(num_step*2):
        traj.append(goal)

    return traj

def generate_traj_for_pick_box(source, goal):
    traj = []

    num_step = 1000

    for i in range(num_step+1):
        traj.append(source)

    for i in range(num_step+1):
        traj.append((goal-source)*(i/num_step) + source)

    for i in range(num_step*2):
        traj.append(goal)

    return traj

    


def generate_traj_for_box_hover(source, goal, num_step):

    hover = source + [0, 0, 0.2]
    points = np.vstack([source, hover, (hover + goal) / 2, goal]).T
    return generate_traj_with_spline(points, num_step)


def generate_traj_for_object_hover(source, goal, num_step):
    num_control_points = 1
    control_point_1 = np.random.uniform(low=np.minimum(source, goal), high= (source + goal)/2, size=(num_control_points, 3))
    control_point_2 = np.random.uniform(low=(source + goal)/2, high= np.maximum(source, goal), size=(num_control_points, 3))
    control_points = np.concat((control_point_1, control_point_2))

    points = np.vstack([source, control_points, goal]).T

    for i in range(len(points)):
        if points[i][-1] < points[i][0]:
            points[i] = np.flip(points[i])
            temp = points[i, 0]
            points[i, 0] = points[i, -1]
            points[i, -1] = temp

    return generate_traj_with_spline(points, num_step)



def generate_traj_with_spline(points, num_step):

    spl, u = make_splprep(points)

    traj = []
    for i in range(num_step+1):
        traj.append(spl(i/num_step))

    return traj

def set_goal(data, goal):
    data.mocap_pos = goal[:3]
    data.mocap_quat = goal[3:]

def set_pos_goal(model, data, goal, inverse=False):
    data.mocap_pos = goal
    if inverse:
        data.mocap_quat=[0, 0.707, 0.707, 0]
    

def set_quat_goal(data, goal):
    data.mocap_quat = goal

def randomize_env(model, data, forward):
    box_id = model.body("hex_container").id
    object_id = model.body("object").id

    box_x_low, box_x_high, box_y_low, box_y_high = -0.3, 0.3, -0.3, -0.5
    [box_random_x, box_random_y] = np.random.uniform(low=[box_x_low, box_y_low], high=[box_x_high, box_y_high])

    data.qpos[model.jnt_qposadr[model.body_jntadr[box_id]]] = box_random_x  # New x position
    data.qpos[model.jnt_qposadr[model.body_jntadr[box_id]] + 1] = box_random_y  # New y position

    if forward:
        object_x_low, object_x_high, object_y_low, object_y_high = -0.3, 0.3, -0.3, -0.55

        [object_random_x, object_random_y] = np.random.uniform(low=[object_x_low, object_y_low], high=[object_x_high, object_y_high])

        while ((box_random_x - object_random_x)**2 + (box_random_y - object_random_y)**2)**0.5 < 0.3:
            [object_random_x, object_random_y] = np.random.uniform(low=[object_x_low, object_y_low], high=[object_x_high, object_y_high])

        data.qpos[model.jnt_qposadr[model.body_jntadr[object_id]]] = object_random_x  # New x position
        data.qpos[model.jnt_qposadr[model.body_jntadr[object_id]] + 1] = object_random_y  # New y position

        return box_random_x, box_random_y, object_random_x, object_random_y
    else:
        data.qpos[model.jnt_qposadr[model.body_jntadr[object_id]]] = box_random_x
        data.qpos[model.jnt_qposadr[model.body_jntadr[object_id]] + 1] = box_random_y
        data.qpos[model.jnt_qposadr[model.body_jntadr[object_id]] + 2] = 0.02

        return box_random_x, box_random_y, box_random_x, box_random_y
 
def set_env(model, data, env_info):

    box_id = model.body("hex_container").id
    object_id = model.body("object").id

    box_x, box_y, _, __ = env_info

    data.qpos[model.jnt_qposadr[model.body_jntadr[box_id]]] = box_x  # New x position
    data.qpos[model.jnt_qposadr[model.body_jntadr[box_id]] + 1] = box_y  # New y position

    data.qpos[model.jnt_qposadr[model.body_jntadr[object_id]]] = box_x
    data.qpos[model.jnt_qposadr[model.body_jntadr[object_id]] + 1] = box_y
    data.qpos[model.jnt_qposadr[model.body_jntadr[object_id]] + 2] = 0.02

def set_joint_goal(model, data, goal):

    for i in range(1, 8):
        data.qpos[model.jnt_qposadr[model.joint(f"joint{i}").id]] = goal[i-1]






    



