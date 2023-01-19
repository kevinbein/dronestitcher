"""Module providing the exit function"""
import sys
import argparse
from datetime import datetime
import numpy as np
import cv2 as cv
import cProfile, pstats


def blendWithGaussianPyramids(in_img1, in_img2, levels=6):
    # Add padding such that the shape is divislbe without yielding odd
    # values down the road (<=> values follow a geometrical sequence)
    in_shape = np.array(in_img1.shape[:2])
    new_shape = (np.ceil(in_shape / (2 ** levels)) * (2 ** levels)).astype(int)
    img1 = np.zeros((new_shape[0], new_shape[1], 3), dtype='uint8')
    img2 = img1.copy()
    img1[:in_shape[0], :in_shape[1]] = in_img1
    img2[:in_shape[0], :in_shape[1]] = in_img2

    _, mask1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY)
    _, mask2 = cv.threshold(img2, 0, 255, cv.THRESH_BINARY)
    mask = np.float32(np.array(cv.bitwise_and(mask1, mask2)) / 255.0)

    GMask = mask.copy()
    G1 = img1.copy()
    G2 = img2.copy()
    gpMask = [GMask]
    gpA = [G1]
    gpB = [G2]
    for i in range(levels):
        GMask = cv.pyrDown(GMask)
        G1 = cv.pyrDown(G1)
        G2 = cv.pyrDown(G2)
        gpMask.append(np.float32(GMask))
        gpA.append(np.float32(G1))
        gpB.append(np.float32(G2))

    lpMask = [gpMask[levels - 1]]
    lpA = [gpA[levels - 1]]
    lpB = [gpB[levels - 1]]
    for i in range(levels - 1, 0, -1):
        A = np.subtract(gpA[i - 1], cv.pyrUp(gpA[i]))
        B = np.subtract(gpB[i - 1], cv.pyrUp(gpB[i]))
        lpMask.append(gpMask[i - 1])
        lpA.append(A)
        lpB.append(B)

    LS = []
    for la, lb, mask in zip(lpA, lpB, lpMask):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)

    L = LS[0]
    for i in range(1, levels):
        L = cv.pyrUp(L)
        L = cv.add(L, LS[i])

    return np.uint8(L)


def stitchFrames(frame1, frame2, iteration=0):
    # Switch around
    framet1 = frame1.copy()
    framet2 = frame2.copy()
    frame1 = framet2
    frame2 = framet1

    # Find feature points
    #sift = cv.SIFT_create()
    #detector = cv.xfeatures2d.SURF_create(400)
    #detector = cv.ORB_create()
    detector = cv.FastFeatureDetector_create()
    kp1, des1 = detector.detectAndCompute(frame1, None)
    kp2, des2 = detector.detectAndCompute(frame2, None)

    # Match correspondences
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #search_params = dict(checks=50)
    #flann = cv.FlannBasedMatcher(index_params, search_params)
    #matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good.append([m])

    # Find the Homography matrix
    pts1 = np.float32([kp1[m[0].queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m[0].trainIdx].pt for m in good])
    H, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 4.0)

    # Calculate the new frame size
    h, w, _ = frame2.shape
    mat_initial = np.array([
        [0, w - 1, w - 1, 0],
        [0, 0, h - 1, h - 1],
        [1, 1, 1, 1]
    ])
    [x, y, c] = np.dot(H, mat_initial)
    x = np.divide(x, c)
    y = np.divide(y, c)
    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))
    new_width = max_x
    new_height = max_y
    correction = [0, 0]
    if min_x < 0:
        new_width -= min_x
        correction[0] = abs(min_x)
    if min_y < 0:
        new_height -= min_y
        correction[1] = abs(min_y)
    if new_width < frame1.shape[1] + correction[0]:
        new_width = frame1.shape[1] + correction[0]
    if new_height < frame1.shape[0] + correction[1]:
        new_height = frame1.shape[0] + correction[1]

    x = np.add(x, correction[0])
    y = np.add(y, correction[1])
    old_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    new_pts = np.float32(np.array([x, y]).transpose())

    H = cv.getPerspectiveTransform(old_pts, new_pts)

    # Stitch everything together
    img_stitched = cv.warpPerspective(frame2, H, (new_width, new_height))
    img_stitched[
        correction[1]:correction[1]+frame1.shape[0],
        correction[0]:correction[0]+frame1.shape[1]
    ] = frame1

    img_warped = cv.warpPerspective(frame2, H, (new_width, new_height))
    img_extended = np.zeros(img_warped.shape, dtype='uint8')
    img_extended[
        correction[1]:correction[1]+frame1.shape[0],
        correction[0]:correction[0]+frame1.shape[1]
    ] = frame1

    # alpha = 0.5
    # beta = 1.0 - alpha
    # img_blended = cv.addWeighted(img_extended, alpha, img_warped, beta, 0.0)

    # Naive: Do alpha blending based and take color channel into consideration
    # img_blended = np.zeros((new_height, new_width, 3), dtype='uint8')
    # for i in range(img_warped.shape[0]):
    #     for j in range(img_warped.shape[1]):
    #         a1 = 1.0e-8 if tuple(img_warped[i, j]) == (0, 0, 0) else 0.5
    #         a2 = 1.0e-8 if tuple(img_extended[i, j]) == (0, 0, 0) else 0.5
    #         a1 = 1.0 if a1 == 0.5 and a2 == 0.0 else a1
    #         a2 = 1.0 if a2 == 0.5 and a1 == 0.0 else a2
    #         alpha = 1 - (1 - a1) * (1 - a2)
    #         r1, g1, b1 = img_warped[i, j]
    #         r2, g2, b2 = img_extended[i, j]
    #         img_blended[i, j, 0] = r1 * a1 / alpha + r2 * a2 / alpha
    #         img_blended[i, j, 1] = g1 * a1 / alpha + g2 * a2 / alpha
    #         img_blended[i, j, 2] = b1 * a1 / alpha + b2 * a2 / alpha

    img_blended = blendWithGaussianPyramids(img_warped, img_extended)

    img_stitched_resized = ResizeWithAspectRatio(img_stitched, width=1280)
    #cv.imshow(f"Image blended {iteration}", img_stitched_resized)
    #img_blended_resized = ResizeWithAspectRatio(img_blended, width=1280)
    #cv.imshow(f"Image blended {iteration}", img_blended_resized)
    #cv.waitKey(0)
    return img_stitched
    return img_blended


def ResizeWithAspectRatio(img, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = img.shape[:2]
    if width is None and height is None:
        return img
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv.resize(img, dim, interpolation=inter)


def main():
    argparser = argparse.ArgumentParser(
        prog='dronestitcher',
        description="""Stitches frames of a mp4 video together. Specifically 
        drone footage which records top-down at 90° angle.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument('filename', help='The input video')
    argparser.add_argument('-s', dest='skip', metavar='N', type=int, default=50, help='Number of frames to skip')
    argparser.add_argument('-m', dest='max', metavar='N', type=int, default=2, help='Maximum number of frames to process')
    argparser.add_argument('-o', dest='offset', metavar='N', type=int, default=0, help='Frame with which to start with')
    args = argparser.parse_args()

    # input_file = "../assets/sequence1.mp4"
    input_file = args.filename
    frame_skip = args.skip
    frame_max = args.max
    frame_offset = args.offset

    cap = cv.VideoCapture(input_file)  # RGB format
    if not cap.isOpened():
        print(f"Cannot open file '{input_file}'")
        sys.exit()

    capinfo_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    capinfo_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    capinfo_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    capinfo_fps = cap.get(cv.CAP_PROP_FPS)
    print(f"Opened file {input_file}")
    print("")
    print(f"\tLength: {capinfo_length}")
    print(f"\tWidth: {capinfo_width}")
    print(f"\tHeight: {capinfo_height}")
    print(f"\tFPS: {capinfo_fps}")
    print("")

    start_time = datetime.now()
    immediate_time = datetime.now()
    frame_count = 0
    output = None
    while True:
        cap.set(1, frame_offset + (frame_count * frame_skip))
        ret, frame = cap.read()
        if not ret:
            print(f"Reached end of movie after {frame_count} iterations")
            break
        if frame_count == 0:
            output = frame.copy()
            frame_count += 1
            continue
        print(f"{frame_count}: Stitch frames", end="")
        prev_output = output.shape
        output = stitchFrames(output, frame, iteration=frame_count)
        print(f" -> Done {output.shape}")
        frame_count += 1
        if frame_count > frame_max:
            break
    
    diff_time = datetime.now() - start_time
    print(f"Calculation took {diff_time}")

    resized_output = ResizeWithAspectRatio(output, width=1280)
    cv.imshow("Stitched image", resized_output)
    cv.waitKey(0)
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
    # pr = cProfile.Profile()
    # pr.enable()
    # main()
    # pr.disable()
    # ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    # ps.print_stats(20)