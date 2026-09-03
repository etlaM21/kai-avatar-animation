from mediapipe_pose_capture import MediaPipePoseCapture

def main():
    # Initialize the capture with the debug window enabled
    print("Starting camera... Press Ctrl+C in the terminal to stop.")
    capture = MediaPipePoseCapture(show_debug=True)
    
    try:
        # Loop continuously to grab frames and keep the OpenCV window responsive
        while True:
            pose_frame = capture.read()
            
            # You can print out the data here to verify it's working behind the scenes
            if pose_frame.valid:
                print(f"Tracking! Found {len(pose_frame.landmarks[0])} landmarks.", end="\r")
            else:
                print("Searching for pose...                        ", end="\r")
                
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        capture.close()

if __name__ == "__main__":
    main()