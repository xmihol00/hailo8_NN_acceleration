mkdir -p audio
ffmpeg -i datasets/tennis/tennis_1.mp4 -q:a 0 -map a audio/tennis_1.aac
ffmpeg -i cropped/video.mp4 -i audio/tennis_1.aac -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 cropped/video_audio.mp4
