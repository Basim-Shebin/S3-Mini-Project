@dataclass
class PreprocessConfig:
    target_width: int = 1280  # keep aspect ratio; height auto
    median_ksize: int = 3
    use_clahe: bool = False  # optional


def resize_keep_aspect(img: np.ndarray, target_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_width:
        return img
    scale = target_width / float(w)
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_CUBIC)


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """Luminosity grayscale conversion via OpenCV."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def median_filter(img_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    k = max(3, int(ksize) | 1)  # ensure odd >=3
    return cv2.medianBlur(img_gray, k)


def otsu_threshold(img_gray: np.ndarray) -> np.ndarray:
    # Otsu's binarization (returns binary image with 0/255)
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_image(bgr: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> Dict[str, np.ndarray]:
    out = {}
    resized = resize_keep_aspect(bgr, cfg.target_width)
    out["resized"] = resized

    gray = to_grayscale(resized)
    out["gray"] = gray

    den = median_filter(gray, cfg.median_ksize)
    out["denoised"] = den

    bw = otsu_threshold(den)
    out["binary"] = bw

    return out
