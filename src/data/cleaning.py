import os
import cv2
import pandas as pd
import subprocess

class MELDVideoProcessor:
    def __init__(self, output_dir, fps=1):
        self.output_dir = output_dir
        self.fps = fps

        self.frames_root = os.path.join(output_dir, "frames")
        self.audio_root = os.path.join(output_dir, "audio")

        os.makedirs(self.frames_root, exist_ok=True)
        os.makedirs(self.audio_root, exist_ok=True)

    # -------------------------------------------------
    def extract_frames(self, video_path, video_id):
        frames_folder = os.path.join(self.frames_root, video_id)
        os.makedirs(frames_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / self.fps))

        count, saved = 0, 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            if count % frame_interval == 0:
                frame_path = os.path.join(frames_folder, f"frame_{saved:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved += 1

            count += 1

        cap.release()
        return frames_folder

    # -------------------------------------------------
    def extract_audio(self, video_path, video_id):
        """
        Extract audio using FFmpeg directly (avoids MoviePy issues)
        Saves audio as WAV 16kHz mono
        """
        audio_path = os.path.join(self.audio_root, f"{video_id}.wav")
        command = [
            "ffmpeg",
            "-y",                 # overwrite if exists
            "-i", video_path,
            "-vn",                # no video
            "-acodec", "pcm_s16le",  # WAV format
            "-ar", "16000",       # sample rate 16kHz
            "-ac", "1",           # mono
            audio_path
        ]

        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"[Warning] Failed extracting audio for {video_id}")
            audio_path = None

        if os.path.exists(audio_path) and not os.path.basename(audio_path).startswith("._"):
            return audio_path
        else:
            return None

    # -------------------------------------------------

    def process_from_csv(self, csv_path, video_folder, max_samples=None, num_workers=4, output_csv="processed_videos.csv"):
        import concurrent.futures
        df = pd.read_csv(csv_path)

        if max_samples is not None:
            df = df.head(max_samples)

        records = []

        def process_row(row):
            video_id = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
            video_path = os.path.join(video_folder, video_id + ".mp4")

            if not os.path.exists(video_path):
                print(f"Missing: {video_id}")
                return {
                    "video_id": video_id,
                    "frames_path": None,
                    "audio_path": None,
                    "utterance": row.get("Utterance", ""),
                    "speaker": row.get("Speaker", ""),
                    "emotion": row.get("Emotion", ""),
                    "sentiment": row.get("Sentiment", ""),
                    "status": "missing_file"
                }

            print("Processing:", video_id)

            try:
                frames = self.extract_frames(video_path, video_id)
                audio = self.extract_audio(video_path, video_id)
                status = "ok"
            except Exception as e:
                print(f"[Error] Failed processing {video_id}: {e}")
                frames, audio = None, None
                status = "failed"

            return {
                "video_id": video_id,
                "frames_path": frames,
                "audio_path": audio,
                "utterance": row.get("Utterance", ""),
                "speaker": row.get("Speaker", ""),
                "emotion": row.get("Emotion", ""),
                "sentiment": row.get("Sentiment", ""),
                "status": status
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(process_row, [row for _, row in df.iterrows()])

        records = list(results)

        # Save all results, including failed ones
        output_path = os.path.join(self.output_dir, output_csv)
        pd.DataFrame(records).to_csv(output_path, index=False)
        print(f"Saved log to {output_path}")

        return pd.DataFrame(records)