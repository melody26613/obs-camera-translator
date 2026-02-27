from moviepy.editor import VideoFileClip

clip = VideoFileClip("2025-08-24_10-03-11.mkv")
clip.write_gif("2025-08-24_10-03-11.gif", fps=10)  # fps 越低檔案越小
