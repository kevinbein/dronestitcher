import sys
import os
import cv2
import timeit

if len(sys.argv) != 2:
    print('Usage: python stitch_video_frames.py <video_file>')
    sys.exit(1)

final_frame = None
video = None
def main():
    global final_frame 
    global video

    # Open the video file
    video = cv2.VideoCapture(sys.argv[1])

    # Load the frames and store them in a list
    interval = 10
    level = 6
    frame_couple = 2

    level_value = 2 ** level
    levels = [int(2 ** (level - i)) for i in range(level)]
    #levels = [128, 32, 8, 2]
    levels = [32, 16, 8, 4, 2]

    print(f'Setup done: ')
    print(f'\tinterval={interval}')
    print(f'\tlevel={level}')
    print(f'\tlevel_value={level_value}')
    print(f'\tframe_couple={frame_couple}')
    print(f'\tlevels={levels}')

    print("Start collecting frames")
    i = 0
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if i % interval == 0:
            frames.append(frame)
            # print('Added frame', len(frames))
        i += 1
        if i >= interval * level_value:
            break
    print(f'Added {len(frames)} frames')

    print("Create stitcher")
    stitcher = cv2.createStitcher(cv2.Stitcher_SCANS) if cv2.__version__.startswith('3') else cv2.Stitcher_create(cv2.Stitcher_SCANS)
    print("Start stitching")
    for i in levels:
        results = []
        for j in range(0, i, frame_couple):
            # print(f'Stitching {j}: {i} and {i+1}')
            (status, stitched_image) = stitcher.stitch(frames[j:j+frame_couple])
            if status == cv2.Stitcher_OK:
                results.append(stitched_image)
            else: 
                print('Failed stitching!')
                sys.exit(1)
        
        # print(f'Results: {len(results)}')
        frames = results
        # cv2.imshow('Intermediate Stitched Image', results[0])
        # cv2.waitKey()

    final_frame = frames[0]


if __name__ == "__main__":
    print(f'Python: Start stitching file "{sys.argv[1]}"')
    elapsed_time = timeit.timeit(main, number=1)
    elapsed_time_ms = elapsed_time * 1000 
    print(f"Stitching took {elapsed_time_ms:.2f} milliseconds")

    cv2.imshow('Stitched Image', final_frame)
    cv2.waitKey()

    # Release the video file
    video.release()
