from pathlib import Path
import cv2
import matplotlib.pyplot as plt


def save_point_visualization(image_path, points, output_path, threshold=0.5):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for point in points:
        cv2.circle(
            image,
            (int(round(point[0])), int(round(point[1]))),
            3,
            (0, 0, 255),
            -1,
        )
    cv2.imwrite(str(output_path), image)


def save_actual_vs_predicted(actual, predicted, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(actual, predicted, alpha=0.6)
    maximum = max(max(actual, default=0), max(predicted, default=0))
    ax.plot([0, maximum], [0, maximum], "r--", label="Ideal")
    ax.set_xlabel("Actual count")
    ax.set_ylabel("Predicted count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
