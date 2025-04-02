import pyzed.sl as sl
import numpy as np

class ZEDCameraManager:
    def __init__(self, camera_id=1, serial_number = None, resolution='HD720', fps=15.0):
        self.camera_id = camera_id
        self.serial_number = serial_number
        self.resolution_str = resolution
        self.fps = fps
        self.camera = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.point_cloud = sl.Mat()
        self.depth_image = sl.Mat()
        self.image = sl.Mat()
        
    def open_camera(self):
        """Open the ZED camera with specified settings"""
        init_params = sl.InitParameters()
        init_params.set_from_serial_number(self.serial_number)

        
        # Set resolution
        if self.resolution_str == 'VGA':
            init_params.camera_resolution = sl.RESOLUTION.VGA
        elif self.resolution_str == 'HD720':
            init_params.camera_resolution = sl.RESOLUTION.HD720
        elif self.resolution_str == 'HD1080':
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        elif self.resolution_str == 'HD2K':
            init_params.camera_resolution = sl.RESOLUTION.HD2K
        else:
            init_params.camera_resolution = sl.RESOLUTION.HD720
            
        init_params.camera_fps = self.fps
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL

        init_params.camera_disable_self_calib = False  # Disable self-calibration for fixed cameras

        init_params.coordinate_units = sl.UNIT.METER  # Set units to meters
        
        # Try opening the camera
        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Error opening camera: {err}")
            return False

        return True
    
    def capture(self):
        """Capture RGB image, depth, and point cloud"""
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.image, sl.VIEW.LEFT)
            # Fix: use self.depth_image instead of self.depth
            self.camera.retrieve_measure(self.depth_image, sl.MEASURE.DEPTH)
            self.camera.retrieve_measure(self.point_cloud, sl.MEASURE.XYZBGRA)
            # Return the correct variables
            return self.image, self.depth_image, self.point_cloud
        return None, None, None

    def get_calibration_parameters(self):
        """Get camera calibration parameters"""
        if self.camera.is_opened():
            info = self.camera.get_camera_information()
            calib = info.camera_configuration.calibration_parameters
            return {
                'fx': calib.left_cam.fx,
                'fy': calib.left_cam.fy,
                'cx': calib.left_cam.cx,
                'cy': calib.left_cam.cy,
                'width': info.camera_configuration.resolution.width,
                'height': info.camera_configuration.resolution.height
            }
        return None

    def close_camera(self):
        self.camera.close()