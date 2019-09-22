import multiprocessing as mp
import time

class camera(mp.Process):
    """
    This class describes the object that will read the camera as a separate subprocess. The start/stop is regulated by an Event object.
    """
    def __init__(self, flag_camera_ready, flag_reading_data, s):

        mp.Process.__init__(self)  # Calling super constructor - mandatory
        self.s = s
        self.flag_camera_ready = flag_camera_ready
        self.flag_reading_data = flag_reading_data  # Read the state of the FPGA reading process
        self.frame_buffer = []

    def run(self):
        #import cv2  # Gettho but I have not there seems to be conflicts with matplotlib if outside local scope.
        from cv2 import imshow, VideoCapture, flip, waitKey, destroyAllWindows, VideoWriter, VideoWriter_fourcc
        # Create the camera object
        self.capture_object = VideoCapture(self.s['camera_address'])


        ret, frame = self.capture_object.read()
        imshow('frame', frame)  # Display the first image
        actual_resolution = (frame.shape[1], frame.shape[0])
        print("[INFO] Webcam frames are {} shape".format(frame.shape))

        counter = 0
        self.flag_camera_ready.set()
        self.t0 = time.perf_counter()
        while self.flag_camera_ready.is_set():  # Loop until parent sets the camera ready flag to False
            if self.flag_reading_data.is_set():
                ret, frame = self.capture_object.read()
                counter += 1
                #print(counter)
                frame = flip(frame, 1)  # Vertical flip

                if ret == True:
                    self.frame_buffer.append(frame)  # Put the frame in a buffer
                    imshow('frame', frame)  # For fun, not required if that takes too much resources
                    if waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    raise ValueError('[ERROR] Problem here, look at it.')
            else:
                #print('[INFO] Camera is not ready')
                pass
        self.t1 = time.perf_counter()

        # Define the codec and create VideoWriter object
        print("[INFO] Recorded video for {} s".format(self.t1 - self.t0))
        fourcc = VideoWriter_fourcc(*'MPEG')
        actual_fps = len(self.frame_buffer) / (self.t1 - self.t0)  # Cannot be controlled
        self.out = VideoWriter(self.s['path_video'], fourcc, actual_fps, actual_resolution)
        for frame in self.frame_buffer:
            self.out.write(frame)

        self.capture_object.release()
        self.out.release()
        destroyAllWindows()
        print("[INFO] Camera process is now terminating.")






