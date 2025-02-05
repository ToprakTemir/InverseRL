import mujoco
import mujoco.viewer
import utils
import control
import time
import numpy as np

def run(forward, env_info = None):
    model = mujoco.MjModel.from_xml_path("./xml/xarm7/peg_in_box.xml")
    data = mujoco.MjData(model)

    mujoco.mj_kinematics(model, data)
    control.move_to_initial(model, data)
    mujoco.mj_kinematics(model, data)

    ## wait for robot to move the initial position
    for i in range(1000):
        mujoco.mj_step(model, data)

    if not env_info:
        box_x, box_y, object_x, object_y = utils.randomize_env(model, data, forward)
        env_info = [box_x, box_y, object_x, object_y]
    else:
        utils.set_env(model, data, env_info)

    mujoco.mj_forward(model, data)

    traj = []
    if forward:
        traj = control.generate_path_forward(model, data)
    else:
        traj = control.generate_path_inverse(model, data)

    joint_data = []

    joint_addr_list = []
    for i in range(1,8):
        joint_addr_list.append(model.jnt_qposadr[model.joint(f"joint{i}").id])

    for _ in ["right", "left"]:
        joint_addr_list.append(model.jnt_qposadr[model.joint(f"{_}_driver_joint").id])
        joint_addr_list.append(model.jnt_qposadr[model.joint(f"{_}_finger_joint").id])
        joint_addr_list.append(model.jnt_qposadr[model.joint(f"{_}_inner_knuckle_joint").id])
    

    i = -1000 # wait to observe the initial state of the robot

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and i < len(traj) + 2:
            # control
            if i < len(traj) and i >= 0:
                
                utils.set_goal(data, traj[i])

                if i == 8000: #picking the object
                    data.ctrl[7] = 255
                
                if i == len(traj) - 1 and forward:
                    data.ctrl[7] = 0
                
            joint_data.append(data.qpos[joint_addr_list])
        
            mujoco.mj_step(model, data)
            mujoco.mj_kinematics(model, data)
            i += 1

            viewer.sync()

    joint_data = np.array(joint_data)

    return joint_data, env_info




