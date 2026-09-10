from mediapipe_pose_capture import MediaPipePoseCapture
from pythonosc.udp_client import SimpleUDPClient
import time

def main():
    # Initialize the capture with the debug window enabled
    print("Starting camera... Press Ctrl+C in the terminal to stop.")
    capture = MediaPipePoseCapture(show_debug=True)

    # Initialize the sender
    print("Starting OSC sender...")
    sender = MediaPipePoseSender()
    
    try:
        # Loop continuously to grab frames and keep the OpenCV window responsive
        while True:
            pose_frame = capture.read()
            
            # You can print out the data here to verify it's working behind the scenes
            if pose_frame.valid:
                print(f"Tracking! Found {len(pose_frame.landmarks[0])} landmarks.", end="\r")
                sender.send(pose_frame.worldLandmarks[0])
                
                '''
                # DEBUG FOR PRINTING ALL LANDMARK POSITIONS
                landmarks = pose_frame.landmarks[0]
                world_landmarks = pose_frame.worldLandmarks[0]
                
                print("PoseLandmarkerResult:")
                print("  Landmarks:")
                for i, lm in enumerate(landmarks):
                    print(f"    Landmark #{i}:")
                    print(f"      x            : {lm.x:.6f}")
                    print(f"      y            : {lm.y:.6f}")
                    print(f"      z            : {lm.z:.6f}")
                    print(f"      visibility   : {lm.visibility}")
                    print(f"      presence     : {lm.presence}")

                print("  WorldLandmarks:")
                for i, wlm in enumerate(world_landmarks):
                    print(f"    Landmark #{i}:")
                    print(f"      x            : {wlm.x:.6f}")
                    print(f"      y            : {wlm.y:.6f}")
                    print(f"      z            : {wlm.z:.6f}")
                    print(f"      visibility   : {wlm.visibility}")
                    print(f"      presence     : {wlm.presence}")
                '''
            else:
                print("Searching for pose...                        ", end="\r")
                
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        capture.close()

class MediaPipePoseSender:
    """Captures webcam pose and streams world-space landmarks over OSC."""

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 9001,
        osc_address: str = "/mediapipe/pose",
    ) -> None:
        self.osc_address = osc_address
        self.osc_client = SimpleUDPClient(ip, port)
        self._start_time = time.perf_counter()

    def _timestamp_ms(self) -> int:
        return int((time.perf_counter() - self._start_time) * 1000)

    def _encode_frame(self, world_landmarks) -> list[float]:
        """Flattens the 33 world landmarks into a single OSC arg list."""
        args: list[float] = [float(self._timestamp_ms()), float(len(world_landmarks))]
        for lm in world_landmarks:
            args.extend([lm.x, lm.y, lm.z, lm.visibility, lm.presence])
        return args

    def send(self, worldLandmarks) -> None:
        if worldLandmarks:
            args = self._encode_frame(worldLandmarks)
            self.osc_client.send_message(self.osc_address, args)

if __name__ == "__main__":
    main()