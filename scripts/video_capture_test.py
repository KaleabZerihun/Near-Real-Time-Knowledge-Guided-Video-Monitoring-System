import cv2
import time

print("STARTING video_capture_test.py")

def main():
    # Change this to a video file path if needed, e.g. "data/sample.mp4"
    source = 0

    cap = cv2.VideoCapture(source)

    #Check camera opened successfully
    if not cap.isOpened():
        raise RuntimeError(
            "ERROR: Could not open video source.\n"
            "Tips:\n"
            "  - Try changing source from 0 to 1 (another webcam index)\n"
            "  - Close Zoom/Teams/Discord (they can lock the camera)\n"
            "  - Ensure OpenCV is installed: pip install opencv-python\n"
        )

    #Enforce requested video specs 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # 720p width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # 720p height
    cap.set(cv2.CAP_PROP_FPS, 30)             # target FPS

    #Read back actual values 
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(
        f"Requested: 1280x720 @ 30fps | "
        f"Actual: {actual_width}x{actual_height} @ {actual_fps:.1f}fps"
    )

    #Reduce internal buffering (latency-related)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "Video Capture Test (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    start_time = time.time()
    last_print = start_time
    frames = 0


    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("WARNING: Frame read failed. Exiting.")
                break

            frames += 1

            # Display frame
            cv2.imshow(window_name, frame)

            # Print FPS every 1 second
            now = time.time()
            if now - last_print >= 1.0:
                elapsed = now - start_time
                fps = frames / elapsed if elapsed > 0 else 0.0
                h, w = frame.shape[:2]
                print(f"Frames: {frames} | FPS: {fps:.2f} | Resolution: {w}x{h}")
                last_print = now

            # Quit if user presses 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        total_elapsed = time.time() - start_time
        avg_fps = frames / total_elapsed if total_elapsed > 0 else 0.0
        print("\n=== Summary ===")
        print(f"Total frames: {frames}")
        print(f"Total time:   {total_elapsed:.2f} seconds")
        print(f"Average FPS:  {avg_fps:.2f}")


if __name__ == "__main__":
    main()
