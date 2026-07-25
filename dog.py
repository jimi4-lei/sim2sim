from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UrdfFileCfg, RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg

DOG_CFG = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path="/home/jimi/IsaacLab/source/isaaclab_assets/data/robots/dog/dog.urdf",
        activate_contact_sensors=True,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # 关键：不要合并固定关节，保留足端
        merge_fixed_joints=False,
        fix_base=False,
        self_collision=True,
        # 明确指定根链接和足端
        root_link_name="base",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.4,
            "FR_thigh_joint": -0.4,
            "RL_thigh_joint": 0.4,
            "RR_thigh_joint": -0.4,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            velocity_limit=20.0,
            effort_limit=20.0,
            stiffness=20.0,
            damping=0.5,
        ),
    },
)
