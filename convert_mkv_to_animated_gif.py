from moviepy.editor import VideoFileClip

# Load your MKV file
clip = VideoFileClip("video.mkv")

trimmed_clip = clip.subclip(5) # this creates a clip starting at 5 seconds to the end

# Write the GIF with a frame rate of 10 fps
trimmed_clip.write_gif("output.gif", fps=10)
