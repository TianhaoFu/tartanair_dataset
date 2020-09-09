from azure.storage.blob import ContainerClient
import numpy as np
import io
import cv2
import time
from yonoarc_utils.image import to_ndarray, from_ndarray
from yonoarc_utils.header import set_timestamp, get_timestamp
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped


def depth2vis(depth, maxthresh=50):
    depthvis = np.clip(depth, 0, maxthresh)
    depthvis = depthvis/maxthresh*255
    depthvis = depthvis.astype(np.uint8)
    depthvis = np.tile(depthvis.reshape(depthvis.shape+(1,)), (1, 1, 3))

    return depthvis


def seg2vis(segnp):
    colors = [(205, 92, 92), (0, 255, 0), (199, 21, 133), (32, 178, 170), (233, 150, 122), (0, 0, 255), (128, 0, 0), (255, 0, 0), (255, 0, 255), (176, 196, 222), (139, 0, 139), (102, 205, 170), (128, 0, 128), (0, 255, 255), (0, 255, 255), (127, 255, 212), (222, 184, 135), (128, 128, 0), (255, 99, 71), (0, 128, 0), (218, 165, 32), (100, 149, 237), (30, 144, 255), (255, 0, 255), (112, 128, 144),
              (72, 61, 139), (165, 42, 42), (0, 128, 128), (255, 255, 0), (255, 182, 193), (107, 142, 35), (0, 0, 128), (135, 206, 235), (128, 0, 0), (0, 0, 255), (160, 82, 45), (0, 128, 128), (128, 128, 0), (25, 25, 112), (255, 215, 0), (154, 205, 50), (205, 133, 63), (255, 140, 0), (220, 20, 60), (255, 20, 147), (95, 158, 160), (138, 43, 226), (127, 255, 0), (123, 104, 238), (255, 160, 122), (92, 205, 92), ]
    segvis = np.zeros(segnp.shape+(3,), dtype=np.uint8)

    for k in range(256):
        mask = segnp == k
        colorind = k % len(colors)
        if np.sum(mask) > 0:
            segvis[mask, :] = colors[colorind]

    return segvis


def _calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2(dv, du)

    angleShift = np.pi

    if (True == flagDegree):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt(du * du + dv * dv)

    return a, d, angleShift


def flow2vis(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0):
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = _calculate_angle_distance_from_du_dv(
        flownp[:, :, 0], flownp[:, :, 1], flagDegree=False)

    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((ang.shape[0], ang.shape[1], 3), dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[:, :, 0] = np.remainder((ang + angShift) / (2*np.pi), 1)
    hsv[:, :, 1] = mag / maxF * n
    hsv[:, :, 2] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 1) * hueMax
    hsv[:, :, 1:3] = np.clip(hsv[:, :, 1:3], 0, 1) * 255
    hsv = hsv.astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if (mask is not None):
        mask = mask > 0
        rgb[mask] = np.array([0, 0, 0], dtype=np.uint8)

    return rgb


class TartanAir(object):
    def on_start(self):
        """This function is called once the Block is started
        """
        account_url = 'https://tartanair.blob.core.windows.net/'
        container_name = 'tartanair-release1'
        self.container_client = ContainerClient(account_url=account_url,
                                                container_name=container_name,
                                                credential=None)
        self.envlist = ['abandonedfactory/', 'abandonedfactory_night/', 'amusement/', 'carwelding/', 'endofworld/', 'gascola/', 'hospital/', 'japanesealley/',
                        'neighborhood/', 'ocean/', 'office/', 'office2/', 'oldtown/', 'seasidetown/', 'seasonsforest/', 'seasonsforest_winter/', 'soulcity/', 'westerndesert/']
        self.diff_level = ["Easy", "Hard"][int(
            self.get_property("diff_level"))]
        self.env_ind = self.get_property("env_ind")
        self.trajlist = self.get_trajectory_list(
            self.envlist[self.env_ind], easy_hard=self.diff_level)
        self.trajs_len = len(self.trajlist)
        self.traj_id = self.get_property("traj_id")
        self.alert("Selected Environment: {}".format(
            self.envlist[self.env_ind]), "INFO")
        self.alert("Difficulty Level: {}".format(self.diff_level), "INFO")
        self.alert("Number of available trajectories: {}".format(
            self.trajs_len), "INFO")
        if(self.traj_id >= self.trajs_len):
            self.alert("Trajectory id out of range[0, {}]".format(
                self.trajs_len-1), "ERROR")
        self.frequency = self.get_property("fps")

    def run(self):
        traj_dir = self.trajlist[self.traj_id]
        left_img_list = self.get_image_list(traj_dir, left_right='left')
        print('Find {} left images in {}'.format(len(left_img_list), traj_dir))

        right_img_list = self.get_image_list(traj_dir, left_right='right')
        print('Find {} right images in {}'.format(
            len(right_img_list), traj_dir))

        left_depth_list = self.get_depth_list(traj_dir, left_right='left')
        print('Find {} left depth files in {}'.format(
            len(left_depth_list), traj_dir))

        right_depth_list = self.get_depth_list(traj_dir, left_right='right')
        print('Find {} right depth files in {}'.format(
            len(right_depth_list), traj_dir))

        left_seg_list = self.get_seg_list(traj_dir, left_right='left')
        print('Find {} left segmentation files in {}'.format(
            len(left_seg_list), traj_dir))

        right_seg_list = self.get_seg_list(traj_dir, left_right='left')
        print('Find {} right segmentation files in {}'.format(
            len(right_seg_list), traj_dir))

        flow_list = self.get_flow_list(traj_dir)
        print('Find {} flow files in {}'.format(len(flow_list), traj_dir))

        flow_mask_list = self.get_flow_mask_list(traj_dir)
        print('Find {} flow mask files in {}'.format(
            len(flow_mask_list), traj_dir))

        left_pose_file = self.get_posefile(traj_dir, left_right='left')
        print('Left pose file: {}'.format(left_pose_file))

        right_pose_file = self.get_posefile(traj_dir, left_right='right')
        print('Right pose file: {}'.format(right_pose_file))
        # Load poses
        bc = self.container_client.get_blob_client(
            blob=left_pose_file)
        data = bc.download_blob()
        text_file = open("OutputL.txt", "w")
        text_file.write(data.content_as_text())
        text_file.close()
        pose_l = np.loadtxt("OutputL.txt")
        bc = self.container_client.get_blob_client(
            blob=left_pose_file)
        data = bc.download_blob()
        text_file = open("OutputR.txt", "w")
        text_file.write(data.content_as_text())
        text_file.close()
        pose_r = np.loadtxt("OutputR.txt")

        ltime = time.time()
        idx = 0
        while True:
            if(time.time() - ltime >= 1/self.frequency):
                if(idx == len(left_img_list)):
                    idx = 0
                # RGB Images
                left_img = self.read_image_file(left_img_list[idx])
                right_img = self.read_image_file(right_img_list[idx])
                header = Header()
                set_timestamp(header, time.time())
                header.frame_id = "left_img"
                left_msg = from_ndarray(left_img, header)
                self.publish("left_img", left_msg)
                header.frame_id = "right_img"
                right_msg = from_ndarray(right_img, header)
                self.publish("right_img", right_msg)
                # Depth Images
                left_depth = self.read_numpy_file(left_depth_list[idx])
                left_depth_vis = depth2vis(left_depth)
                header.frame_id = "left_depth"
                left_msg = from_ndarray(left_depth_vis, header)
                self.publish("left_depth", left_msg)
                right_depth = self.read_numpy_file(right_depth_list[idx])
                right_depth_vis = depth2vis(right_depth)
                header.frame_id = "right_depth"
                right_msg = from_ndarray(right_depth_vis, header)
                self.publish("right_depth", right_msg)
                # Semantic Segmentation
                left_seg = self.read_numpy_file(left_seg_list[idx])
                left_seg_vis = seg2vis(left_seg)
                header.frame_id = "left_segmentation"
                left_msg = from_ndarray(left_seg_vis, header)
                self.publish("left_segmentation", left_msg)
                right_seg = self.read_numpy_file(right_seg_list[idx])
                right_seg_vis = seg2vis(right_seg)
                header.frame_id = "right_segmentation"
                right_msg = from_ndarray(right_seg_vis, header)
                self.publish("right_segmentation", right_msg)
                # Left Camera Pose
                pose_stamped = PoseStamped()
                pose_stamped.header = header
                pose_stamped.header.frame_id = "left_camera"
                pose = Pose()
                pose.position.x = pose_l[idx][0]
                pose.position.y = pose_l[idx][1]
                pose.position.z = pose_l[idx][2]
                pose.orientation.x = pose_l[idx][3]
                pose.orientation.y = pose_l[idx][4]
                pose.orientation.z = pose_l[idx][5]
                pose.orientation.w = pose_l[idx][6]
                pose_stamped.pose = pose
                self.publish("left_pose", pose_stamped)
                # Right Camera Pose
                pose_stamped = PoseStamped()
                pose_stamped.header = header
                pose_stamped.header.frame_id = "right_camera"
                pose = Pose()
                pose.position.x = pose_r[idx][0]
                pose.position.y = pose_r[idx][1]
                pose.position.z = pose_r[idx][2]
                pose.orientation.x = pose_r[idx][3]
                pose.orientation.y = pose_r[idx][4]
                pose.orientation.z = pose_r[idx][5]
                pose.orientation.w = pose_r[idx][6]
                pose_stamped.pose = pose
                self.publish("right_pose", pose_stamped)

                if(idx > 0):
                    flow = self.read_numpy_file(flow_list[idx-1])
                    flow_vis = flow2vis(flow)
                    header.frame_id = "optical_flow"
                    left_msg = from_ndarray(flow_vis, header)
                    self.publish("optical_flow", left_msg)
                    flow_mask = self.read_numpy_file(flow_mask_list[idx-1])
                    flow_vis_w_mask = flow2vis(flow, mask=flow_mask)
                    header.frame_id = "optical_flow_mask"
                    right_msg = from_ndarray(flow_vis_w_mask, header)
                    self.publish("optical_flow_mask", right_msg)

                ltime = time.time()
                idx += 1

    def get_environment_list(self):
        '''
        List all the environments shown in the root directory
        '''
        env_gen = self.container_client.walk_blobs()
        envlist = []
        for env in env_gen:
            envlist.append(env.name)
        return envlist

    def get_trajectory_list(self, envname, easy_hard='Easy'):
        '''
        List all the trajectory folders, which is named as 'P0XX'
        '''
        assert(easy_hard == 'Easy' or easy_hard == 'Hard')
        traj_gen = self.container_client.walk_blobs(
            name_starts_with=envname + '/' + easy_hard+'/')
        trajlist = []
        for traj in traj_gen:
            trajname = traj.name
            trajname_split = trajname.split('/')
            trajname_split = [tt for tt in trajname_split if len(tt) > 0]
            if trajname_split[-1][0] == 'P':
                trajlist.append(trajname)
        return trajlist

    def _list_blobs_in_folder(self, folder_name):
        """
        List all blobs in a virtual folder in an Azure blob container
        """

        files = []
        generator = self.container_client.list_blobs(
            name_starts_with=folder_name)
        for blob in generator:
            files.append(blob.name)
        return files

    def get_image_list(self, trajdir, left_right='left'):
        assert(left_right == 'left' or left_right == 'right')
        files = self._list_blobs_in_folder(
            trajdir + '/image_' + left_right + '/')
        files = [fn for fn in files if fn.endswith('.png')]
        return files

    def get_depth_list(self, trajdir, left_right='left'):
        assert(left_right == 'left' or left_right == 'right')
        files = self._list_blobs_in_folder(
            trajdir + '/depth_' + left_right + '/')
        files = [fn for fn in files if fn.endswith('.npy')]
        return files

    def get_flow_list(self, trajdir, ):
        files = self._list_blobs_in_folder(trajdir + '/flow/')
        files = [fn for fn in files if fn.endswith('flow.npy')]
        return files

    def get_flow_mask_list(self, trajdir, ):
        files = self._list_blobs_in_folder(trajdir + '/flow/')
        files = [fn for fn in files if fn.endswith('mask.npy')]
        return files

    def get_posefile(self, trajdir, left_right='left'):
        assert(left_right == 'left' or left_right == 'right')
        return trajdir + '/pose_' + left_right + '.txt'

    def get_seg_list(self, trajdir, left_right='left'):
        assert(left_right == 'left' or left_right == 'right')
        files = self._list_blobs_in_folder(
            trajdir + '/seg_' + left_right + '/')
        files = [fn for fn in files if fn.endswith('.npy')]
        return files

    def read_numpy_file(self, numpy_file,):
        '''
        return a numpy array given the file path
        '''
        bc = self.container_client.get_blob_client(blob=numpy_file)
        data = bc.download_blob()
        ee = io.BytesIO(data.content_as_bytes())
        ff = np.load(ee)
        return ff

    def read_image_file(self, image_file,):
        '''
        return a uint8 numpy array given the file path  
        '''
        bc = self.container_client.get_blob_client(blob=image_file)
        data = bc.download_blob()
        ee = io.BytesIO(data.content_as_bytes())
        img = cv2.imdecode(np.asarray(bytearray(ee.read()),
                                      dtype=np.uint8), cv2.IMREAD_COLOR)
        # im_rgb = img[:, :, [2, 1, 0]]  # BGR2RGB
        return img
