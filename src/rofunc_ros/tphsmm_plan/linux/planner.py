import rospy
import numpy as np
import pickle as pkl

from geometry_msgs.msg import Pose


def get_arm_frames(arm, data, labels):
    # Only for immobile parameters
    if arm == 'left':
        arm_idx = 0
        arm_pts = ['t4', 't7', 'rl', 'rh']
    elif arm == 'right':
        arm_idx = 1
        arm_pts = ['t7', 't4', 'll', 'lh']
    else:
        raise ValueError(f'Invalid arm name {arm}')
    arm_pts = [f"{a}.pose.x" for a in arm_pts]
    if not set(arm_pts).issubset(labels) :
        print(f'ERROR: arm points not in labels. Your data should contain {arm_pts}')
        return None
    # left arm
    As = []
    bs = []
    pts = []
    for l in arm_pts:
        ptr = labels.index(l)
        has_nan = np.any(np.isnan(data[0, ptr:ptr+3]))
        if has_nan:
            print(f"{l} - Number of nans: {np.count_nonzero(np.isnan(data[0, ptr:ptr+3]))}")

        pts.append(data[0, ptr:ptr+3])
    for a in range(3):
        if a < 2:
            A = get_rotation_matrix(pts[a], pts[a+1], pts[a+2])
        else:
            A = get_rotation_matrix(pts[a-1], pts[a], pts[a+1])
        A = np.kron(np.eye(4), A)
        b = np.kron(np.array([1,1,0,0]),
                    pts[a+1])
        # [A, 0 ,0, 0] [b] xl
        # [0, A ,0, 0] [b] xr
        # [0 ,0, A, 0] [0] dxl
        # [0, 0 ,0, A] [0] dxr
        As.append(A)
        bs.append(b)

    return As, bs

class TPSHMMPLanner(object):
    def __init__(self, model_path, ot_topic, pose_topic, freq=1, horizon=200):
        rospy.init_node('tphsmm_planner', anonymous=True)
        self.model = pkl.load(open(model_path, 'rb'))
        self.freq = freq
        self.horizon = horizon
        self.ot_data = None
        self.pose_data = None
        self.ot_sub = rospy.Subscriber(ot_topic, Pose, self.ot_cb)
        self.pose_sub = rospy.Subscriber(pose_topic, Pose, self.pose_cb)

    def plan(self, ot_data: np.ndarray, pose_data: np.ndarray=None,
             horizon=200):

        if ot_data is None:
            return None

        labels = ['t4', 't7', 'rl', 'rh', 'll', 'lh']
        ltp = get_arm_frames('left', ot_data, labels)
        rtp = get_arm_frames('right', ot_data, labels)
        task_params = (np.stack(ltp[0] + rtp[0]),  # N_obs, xdx_dim, xdx_dim
                        np.stack(ltp[1] + rtp[1]))  # N_obs, xdx_dim
        if pose_data is None:
            pose_data = self.model.demos_tp[0][0, :]
        traj = self.model.generate(0, task_params, pose_data, horizon)

        traj /= 1000

        axshift = np.kron(np.eye(2), np.array([[1,0,0],[0,0,-1],[0,1,0]]))
        traj = (axshift @ traj[:, :6].T).T
        traj = np.repeat(traj, 10, axis=0)
        if traj[:, ::3].max() > 0.75 :
            print("X value over 0.75 ! Abort")
            return
        out_traj = np.zeros((2, traj.shape[0], 3))
        out_traj[0] = traj[:, :3]
        out_traj[1] = traj[:, 3:]
        return traj

    def ot_cb(self, msg):
        self.ot_data = np.array([msg.position.x, msg.position.y, msg.position.z,
                                 msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def pose_cb(self, msg):
        self.pose_data = np.array([msg.position.x, msg.position.y, msg.position.z,
                                   msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def run(self):
        rate = rospy.Rate(self.freq)
        while not rospy.is_shutdown():
            traj = self.plan(self.ot_data, self.pose_data)
            if traj is not None:
                print(traj)
            rate.sleep()
