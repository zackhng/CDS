import os
import cv2
from moviepy import VideoFileClip

class VideoPreprocessor:
    def __init__(self, output_dir, fps=1):
        """
        output_dir: root directory for preprocessed data
        fps: frames per second to extract
        """
        self.output_dir = output_dir
        self.fps = fps
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "transcripts"), exist_ok=True)

    def process_video(self, video_path, transcript_text, label):
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Extract frames
        frames_folder = os.path.join(self.output_dir, "frames", video_id)
        os.makedirs(frames_folder, exist_ok=True)
        vidcap = cv2.VideoCapture(video_path)
        video_fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / self.fps))
        count, saved = 0, 0
        while True:
            success, frame = vidcap.read()
            if not success:
                break
            if count % frame_interval == 0:
                frame_path = os.path.join(frames_folder, f"frame_{saved:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved += 1
            count += 1
        vidcap.release()

        # Extract audio
        audio_folder = os.path.join(self.output_dir, "audio")
        audio_path = os.path.join(audio_folder, f"{video_id}.wav")
        clip = VideoFileClip(video_path)
        if clip.audio is not None:
            clip.audio.write_audiofile(audio_path, logger=None)
        else:
            print(f"[Warning] {video_id} has no audio track.")

        # Save transcript
        transcript_folder = os.path.join(self.output_dir, "transcripts")
        transcript_path = os.path.join(transcript_folder, f"{video_id}.txt")
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        # Return indexing info
        return {
            "video_id": video_id,
            "frames_folder": frames_folder,
            "audio_file": audio_path,
            "transcript_file": transcript_path,
            "label": label
        }

    def process_folder(self, videos_folder, transcripts_dict, labels_dict):
        """
        Process all MP4 videos in a folder.
        transcripts_dict: {video_id: transcript_text}
        labels_dict: {video_id: label}
        Returns a list of dicts for indexing all processed videos.
        """
        processed = []
        for file in os.listdir(videos_folder):
            if file.lower().endswith(".mp4"):
                video_path = os.path.join(videos_folder, file)
                video_id = os.path.splitext(file)[0]
                transcript = transcripts_dict.get(video_id, "")
                label = labels_dict.get(video_id, "unknown")
                info = self.process_video(video_path, transcript, label)
                processed.append(info)
        return processed