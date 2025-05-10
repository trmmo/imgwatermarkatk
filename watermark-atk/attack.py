import numpy as np


class Attack:

    @staticmethod
    def crop5(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        return img[int(w * 0.05):, :]
