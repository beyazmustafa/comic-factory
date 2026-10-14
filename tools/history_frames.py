"""Four larger frames of a finished video (hook, two from the middle, the end
card) side by side, for a closer look at typography and crops in the history."""
import subprocess
import sys


def main(video, out, length):
    length = float(length)
    times = [1.6, length * 0.35, length * 0.62, max(0.0, length - 1.4)]
    parts = []
    for t in times:
        parts += ["-ss", f"{t:.2f}", "-i", video]
    chain = "".join(f"[{i}:v]scale=480:-2,trim=end_frame=1,setpts=PTS-STARTPTS[v{i}];" for i in range(4))
    chain += "".join(f"[v{i}]" for i in range(4)) + "hstack=inputs=4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *parts, "-filter_complex", chain, "-frames:v", "1", "-q:v", "4", out],
                   check=False, timeout=300)


if __name__ == "__main__":
    main(*sys.argv[1:4])
