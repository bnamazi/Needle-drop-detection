import os
import glob

from joblib import delayed, Parallel, cpu_count
import cv2


VIDEOS_PATH = "./data/videos"
IMAGES_PATH = "./data/images/"


def crop_image(img, threshold=20):
    """
    Crop the black margin out from an image
    """
    mask = img > threshold
    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def extract_cropped_images(video: str):
    video_name = video.split("\\")[-1].split(".")[0]

    if not os.path.exists(f"{IMAGES_PATH}{video_name}/"):
        os.mkdir(f"{IMAGES_PATH}{video_name}/")

    cap = cv2.VideoCapture(video)
    success, frame = cap.read()
    count = 0

    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(video_name, frame_rate, width, height)

    while success:

        if not os.path.exists(
            f"{IMAGES_PATH}{video_name}/{video_name}Frame{str(count+1).zfill(3)}.jpg"
        ):
            cv2.imwrite(
                f"{IMAGES_PATH}{video_name}/{video_name}Frame{str(count+1).zfill(3)}.jpg",
                cv2.resize(
                    crop_image(frame), (500, 500), interpolation=cv2.INTER_CUBIC
                ),
            )
        success, frame = cap.read()
        count += 1


if __name__ == "__main__":
    videos = glob.glob(f"{VIDEOS_PATH}/fps1/*.mp4", recursive=True)

    Parallel(n_jobs=cpu_count())(delayed(extract_cropped_images)(vid) for vid in videos)
