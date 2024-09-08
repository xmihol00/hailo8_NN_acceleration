mkdir -p audio

# check the 1st parameter for the video file name
video_file=video
if ! [ -z "$1" ]; then
  video_file=$1
fi

ffmpeg -i datasets/tennis/${video_file}.mp4 -q:a 0 -map a audio/${video_file}.aac
ffmpeg -i cropped/${video_file}.mp4 -i audio/${video_file}.aac -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 cropped/${video_file}_audio.mp4
