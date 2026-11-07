"""HSV 검출 단위 시험. 시뮬레이터 없음. 주황은 거절하고 빨강만 받는다."""
import numpy as np
from camera_vision import detect_red, red_mask


def _img(h=180, w=320, color=(40, 40, 40)):
    im = np.zeros((h, w, 3), np.uint8)
    im[:] = color
    return im


def _disk(im, cx, cy, r, color):
    yy, xx = np.ogrid[: im.shape[0], : im.shape[1]]
    im[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = color


def test_red_disk_center():
    im = _img()
    _disk(im, 160, 90, 14, (230, 20, 25))
    d = detect_red(im)
    assert d is not None
    assert abs(d["cx"] - 160) < 3
    assert abs(d["ex"]) < 0.08
    assert d["area"] > 100


def test_red_on_the_right():
    im = _img()
    _disk(im, 260, 100, 10, (220, 10, 15))
    d = detect_red(im)
    assert d is not None and d["ex"] > 0.4


def test_orange_foot_rejected():
    im = _img()
    _disk(im, 160, 150, 18, (255, 102, 31))  # 턱·발 주황
    assert detect_red(im) is None
    assert int(red_mask(im).sum()) == 0


def test_no_red():
    im = _img(color=(180, 190, 200))
    _disk(im, 40, 40, 12, (80, 160, 220))
    assert detect_red(im) is None


def test_speck_rejected():
    im = _img()
    im[10:13, 10:14] = (240, 0, 0)
    assert detect_red(im) is None


if __name__ == "__main__":
    test_red_disk_center()
    test_red_on_the_right()
    test_orange_foot_rejected()
    test_no_red()
    test_speck_rejected()
    print("camera_vision tests ok")
