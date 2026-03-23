def should_process_frame(frame_index: int, sample_every_n_frames: int) -> bool:
    return frame_index % sample_every_n_frames == 0