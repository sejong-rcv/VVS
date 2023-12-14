import cv2
import time
import numpy as np


def center_crop(frame, desired_size):
    if frame.ndim == 1:
        return frame
    elif frame.ndim == 3:
        old_size = frame.shape[:2]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[top: top+desired_size, left: left+desired_size, :]
    else: 
        old_size = frame.shape[1:3]
        top = int(np.maximum(0, (old_size[0] - desired_size)/2))
        left = int(np.maximum(0, (old_size[1] - desired_size)/2))
        return frame[:, top: top+desired_size, left: left+desired_size, :]


def resize_frame(frame, desired_size):
    min_size = np.min(frame.shape[:2])
    ratio = desired_size / min_size
    frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return frame


def load_video(video, slow_mo=1, all_frames=False, fps=1, cc_size=224, rs_size=256): 
    cv2.setNumThreads(1) 
    cap = cv2.VideoCapture(video)
    fps_div = fps
    fps = cap.get(cv2.CAP_PROP_FPS) / slow_mo
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps > 144 or fps is None:
        fps = 25
    frames = []
    count = 0
    start = time.time()
    for fi in range(frame_count):
        ret = cap.grab()
        if int(fi % round(fps / fps_div)) == 0 or all_frames:
            ret, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rs_size is not None:
                    frame = resize_frame(frame, rs_size)
                frames.append(frame)
            else:
                break
    cap.release()
    frames = np.array(frames)
    if cc_size is not None:
        frames = center_crop(frames, cc_size)
    return frames


def slow_video_load(video, cc_size=224, rs_size=256, mask_length=0, original_video_length=0):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(original_video_length * fps)

    # Modified to ensure video of the same size as the artificially generated mask for Slowmotion unconditionally
    try:
        extract_point = np.arange(0, frame_count, frame_count/mask_length)
    except:
        print(frame_count)
        print(mask_length)
        print(fps)
        print(original_video_length)
        print(video)
        import pdb; pdb.set_trace()
    extract_point = [round(i) for i in extract_point]

    tmp_frame = None
    load_fail_flag = False
    frames = []
    for fi in range(frame_count):
        ret = cap.grab()
        if fi in extract_point or load_fail_flag:
            ret, frame = cap.retrieve()
            if isinstance(frame, np.ndarray):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rs_size is not None:
                    frame = resize_frame(frame, rs_size)
                frames.append(np.array(frame))
                tmp_frame = frame
                load_fail_flag = False
            else:
                if fi in extract_point and load_fail_flag:
                    # If load failed until next extract point, Fill it as a previous frames.
                    frames.append(np.array(tmp_frame))
                    frames.append(np.array(tmp_frame))
                    load_fail_flag = False
                else:
                    load_fail_flag = True
    cap.release()

    frames = np.array(frames)
    if cc_size is not None:
        frames = center_crop(frames, cc_size)
        
    return frames