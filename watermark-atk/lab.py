import argparse

import cv2

from attack import Attack
from DCT import DCT


def main(args):
    img = cv2.imread(args.pathin)
    watermark = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE)
    action = args.action

    if action == 'embed':
        model = DCT()
        embeded = model.embed(img, watermark)
        cv2.imwrite(args.pathout, embeded)
        print(f"Anh duoc watermark luu tai: {args.pathout}")

    elif action == 'extract':
        model = DCT()
        signature = model.extract(img)
        cv2.imwrite(args.pathout, signature)
        print(f"Watermark duoc trich xuat tai: {args.pathout}")

    elif action == "attack":
        att_img = Attack.crop5(img)
        cv2.imwrite(args.pathout, att_img)
        print(f"Anh tan cong duoc luu tai: {args.pathout}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pi", "--pathin", default="./images/fakepathin.jpg", help="Duong dan file input")
    parser.add_argument("-wm", "--watermark", default="./images/fakewatermark.jpg", help="Duong dan watermark")
    parser.add_argument("-po", "--pathout", default="./images/fakepathout.jpg", help="Duong dan file output")
    parser.add_argument("-ac", "--action", default="nothing", help="Tuy chon: embed, extract, attack")
    args = parser.parse_args()
    main(parser.parse_args())
