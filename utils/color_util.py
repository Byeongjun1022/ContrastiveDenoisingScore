import cv2
import torch
import numpy as np

# from scipy import misc
from PIL import Image


def preprocess_lab(lab):
    """
    Split the L*a*b* image channels and normalize them to ranges [-1, +1].

    Parameters:
    * lab: L*a*b* image; pytorch tensor, shape [H, W, 3]. L* range [0, 100], a* & b* range [-110, +110]

    Output:
    * L_chan: L* channel image; pytorch tensor, shape [H, W]. value range [-1, +1]
    * a_chan: a* channel image; pytorch tensor, shape [H, W]. value range [-1, +1]
    * b_chan: b* channel image; pytorch tensor, shape [H, W]. value range [-1, +1]
    """
    L_chan, a_chan, b_chan = torch.unbind(lab, dim=2)
    # L_chan: black and white with input range [0, 100]
    # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
    # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
    return [L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0]


def deprocess_lab(L_chan, a_chan, b_chan):
    """
    Denormalize L*, a*, b* channels and recombine them to a single L*a*b* image.

    Parameters:
    * L_chan: L* channel image; pytorch tensor, shape [H, W]. value range [-1, +1]
    * a_chan: a* channel image; pytorch tensor, shape [H, W]. value range [-1, +1]
    * b_chan: b* channel image; pytorch tensor, shape [H, W]. value range [-1, +1]

    Output:
    * lab: L*a*b* image; pytorch tensor, shape [H, W, 3]. L* range [0, 100], a* & b* range [-110, +110]
    """
    # TODO This is axis=3 instead of axis=2 when deprocessing batch of images
    # ( we process individual images but deprocess batches)
    # return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
    return torch.stack(
        [(L_chan + 1) / 2.0 * 100.0, a_chan * 110.0, b_chan * 110.0], dim=2
    )


def rgb_to_lab(srgb):
    """
    Convert RGB (normalized to [0, 1]) image into L*a*b* image.

    Parameter(s):
    * srgb: RGB image, normalized to [0, 1]; pytorch tensor, shape [H, W, 3]

    Output:
    * lab: L*a*b* image; pytorch tensor, shape [H, W, 3]. L* range [0, 100], a* & b* range [-110, +110]
    """
    srgb_pixels = torch.reshape(srgb, [-1, 3])

    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).cuda()
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).cuda()
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
        ((srgb_pixels + 0.055) / 1.055) ** 2.4
    ) * exponential_mask  # dtype float64
    rgb_pixels = rgb_pixels.type(dtype=torch.float32)

    rgb_to_xyz = (
        torch.tensor(
            [
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ]
        )
        .type(torch.FloatTensor)
        .cuda()
    )  # dtype float32

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(
        xyz_pixels,
        torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754]).type(torch.FloatTensor).cuda(),
    )

    epsilon = 6.0 / 29.0

    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).cuda()

    exponential_mask = (
        (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).cuda()
    )

    fxfyfz_pixels = (
        xyz_normalized_pixels / (3 * epsilon**2) + 4.0 / 29.0
    ) * linear_mask + (
        (xyz_normalized_pixels + 0.000001) ** (1.0 / 3.0)
    ) * exponential_mask
    # convert to lab
    fxfyfz_to_lab = (
        torch.tensor(
            [
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ]
        )
        .type(torch.FloatTensor)
        .cuda()
    )
    lab_pixels = (
        torch.mm(fxfyfz_pixels, fxfyfz_to_lab)
        + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda()
    )
    # return tf.reshape(lab_pixels, tf.shape(srgb))
    return torch.reshape(lab_pixels, srgb.shape)


def lab_to_rgb(lab):
    """
    Convert L*a*b* image to RGB (normalized to [0, 1]) image.

    Parameter(s):
    * srgb: RGB image, normalized to [0, 1]; pytorch tensor, shape [H, W, 3]

    Output:
    * lab: L*a*b* image; pytorch tensor, shape [H, W, 3]. L* range [0, 100], a* & b* range [-110, +110]
    """
    lab_pixels = torch.reshape(lab, [-1, 3])
    # convert to fxfyfz
    lab_to_fxfyfz = (
        torch.tensor(
            [
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ]
        )
        .type(torch.FloatTensor)
        .cuda()
    )
    fxfyfz_pixels = torch.mm(
        lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda(),
        lab_to_fxfyfz,
    )

    # convert to xyz
    epsilon = 6.0 / 29.0
    linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).cuda()
    exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).cuda()

    xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4 / 29.0)) * linear_mask + (
        (fxfyfz_pixels + 0.000001) ** 3
    ) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = torch.mul(
        xyz_pixels,
        torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).cuda(),
    )

    xyz_to_rgb = (
        torch.tensor(
            [
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ]
        )
        .type(torch.FloatTensor)
        .cuda()
    )

    rgb_pixels = torch.mm(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    # clip
    rgb_pixels[rgb_pixels > 1] = 1
    rgb_pixels[rgb_pixels < 0] = 0

    linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).cuda()
    exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).cuda()
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
        ((rgb_pixels + 0.000001) ** (1 / 2.4) * 1.055) - 0.055
    ) * exponential_mask

    return torch.reshape(srgb_pixels, lab.shape)


def replace_luminance(L_chan, rgb):
    """
    Take rgb image, extract only chroma, and recombine with luminance given.
    Operate in the LAB color space for now

    Params:
    * L_chan: L channel. pytorch tensor; shape [H, W]; value range [-1, +1]
    * rgb: RGB image in value range [0, +1]. pytorch tensor; shape [H, W, 3]

    Returns:
    * rgb: RGB image in value range [0, +1]. pytorch tensor; shape [H, W, 3]
    """
    rgb = torch.from_numpy(rgb).to(device=L_chan.device)
    lab = rgb_to_lab(rgb)
    _, a_chan, b_chan = preprocess_lab(lab)
    lab_new = deprocess_lab(L_chan, a_chan, b_chan)
    rgb_new = lab_to_rgb(lab_new)
    return rgb_new.cpu().numpy()


# test
if __name__ == "__main__":
    # read image file and convert bgr to rgb & normalize to [0, 1]
    img = cv2.imread("sample/cat1.png", 1) / 255.0
    img = img[:, :, (2, 1, 0)]  # bgr to rgb
    # img = misc.imread('data/test_rgb.jpg')/255.0
    img = torch.from_numpy(img).cuda()  # tensor, shape [H, W, 3] ([512, 512, 3])

    # img_pil = Image.open("sample/cat1.png")
    # lab_img = img_pil.convert("LAB")
    # l, a, b = lab_img.split()

    # l_array = np.array(l)

    # convert from (normalized [0, 1]) rgb to lab
    lab = rgb_to_lab(
        img
    )  # output is tensor, shape [H, W, 3] ([512, 512, 3]), input range [0, 1], output range [0, 100] (L) [-110, +110] (a, b)

    # function to split into channels and normalize to [-1, +1]
    L_chan, a_chan, b_chan = preprocess_lab(
        lab
    )  # shapes: L_chan: [512, 512], a_chan: [512, 512], b_chan: [512, 512], output range [-1, 1]

    # L_chan_denorm = (L_chan + 1.0) / 2.0 * 100.0

    # function to denormalize channels and recombine
    # lab = deprocess_lab(
    #     L_chan, a_chan, b_chan
    # )  # output shape: [H, W, 3] ([512, 512, 3])

    lab = deprocess_lab(
        L_chan, L_chan, L_chan
    )  # output shape: [H, W, 3] ([512, 512, 3])

    # convert from lab to (normalized [0, 1]) rgb
    true_image = lab_to_rgb(lab)  # range [0.0, 1.0], shape: [H, W, 3] ([512, 512, 3])

    # denormalize to [0, 255] and save
    true_image = np.round(true_image.cpu() * 255.0)
    true_image = np.uint8(true_image)
    # np.save('torch.npy',np.array(img.cpu()))
    # conv_img = Image.fromarray(true_image, 'RGB')
    # conv_img.save('converted_test_pytorch.jpg')
    true_image = true_image[:, :, (2, 1, 0)]
    cv2.imwrite("lll.jpg", true_image)  # no image warping or destruction
    # import pdb; pdb.set_trace()
