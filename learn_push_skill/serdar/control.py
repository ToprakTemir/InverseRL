import mujoco
from utils import *

hover_dist_object = 0.2
hover_dist_box = 0.3

def move_to_initial(model, data):
    data.mocap_pos = [0.206, 0, 0.5]
    #data.mocap_quat = [0, 0.707, 0.707, 0]
    mujoco.mj_step(model, data)

def generate_path_forward(model, data):
    box_id = model.body("hex_container").id
    object_id = model.body("object").id
    end_eff_id = model.body("end_effector").id

    num_step = 5000
    initial_source = data.mocap_pos
    object_hover_goal = data.xpos[object_id] + [0, 0, hover_dist_object]
    source_object = data.xpos[object_id]
    box_hover_goal = data.xpos[box_id] + [0, 0, hover_dist_box]

    traj_object_hover = generate_traj_for_object_hover(initial_source, object_hover_goal, num_step)
    traj_pick = generate_traj_for_pick_object(model, data)
    traj_box_hover = generate_traj_for_box_hover(source_object, box_hover_goal, num_step)

    traj_pos = np.concat((traj_object_hover, traj_pick, traj_box_hover))

    traj_quat = np.tile(np.array([0,0,1,0]),(len(traj_pos), 1))

    traj = np.concat((traj_pos, traj_quat), axis=-1)

    return traj

def generate_path_inverse(model, data):
    box_id = model.body("hex_container").id
    
    num_step = 5000
    initial_source = data.mocap_pos
    box_pick_hover_goal = data.xpos[box_id] + [0.08, 0, hover_dist_object]
    source_box = box_pick_hover_goal - [0, 0, hover_dist_object - 0.04]
    goal_carry_box = [0.25, -0.4, 0.4]

    traj_box_pick_hover = generate_traj_for_object_hover(initial_source, box_pick_hover_goal, num_step) # 5000
    quat_traj_box_pick_hover = generate_quat_traj_for_box_pick(num_step)
    traj_pick = generate_traj_for_pick_box(box_pick_hover_goal, source_box) # 3000
    traj_box_hover = generate_traj_for_box_hover(source_box, goal_carry_box, num_step) #5000
    
    traj_quat_flip = generate_traj_for_flip() #1000
    traj_pos_flip = [goal_carry_box for _ in range(len(traj_quat_flip))]

    traj_pos = np.concat((traj_box_pick_hover, traj_pick, traj_box_hover, traj_pos_flip))

    num_step_until_flip = len(traj_pick) + len(traj_box_hover) 
    initial_quat = np.array([0, 0.707, 0.707, 0])

    traj_quat = np.concat((quat_traj_box_pick_hover, np.tile(initial_quat, (num_step_until_flip, 1)), traj_quat_flip))

    traj = np.concat((traj_pos, traj_quat), axis=-1)

    return traj