import torch
from torch import nn
import numpy as np
import argparse
import math
import cv2
import time
from pathlib import Path
import sys
import os
import glob
import contextlib
import torchvision
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

from .classifier import Model, SimpleMobileNetV2


register_heif_opener()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Set classification model

def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from yolo import Detect, Model
    model = Ensemble()
    weights = os.path.join(os.getcwd(), weights)
    ckpt = torch.load(weights, map_location='cpu')  # load
    ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model
    # Model compatibility updates
    if not hasattr(ckpt, 'stride'):
        ckpt.stride = torch.tensor([32.])
    if hasattr(ckpt, 'names') and isinstance(ckpt.names, (list, tuple)):
        ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
    model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, 'fuse') else ckpt.eval())  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f'Models have different class counts: {[m.nc for m in model]}'
    return model


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device='cpu', dnn=False, data=None, fp16=False, fuse=True):
        super().__init__()
        w = weights
        # stride = 32  # default stride
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # use CUDA
        model = attempt_load(self.weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
        self.stride = max(int(model.stride.max()), 32)  # model stride
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half() if fp16 else model.float()
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        self.fp16 = fp16

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if isinstance(im, np.ndarray):
            im = np.expand_dims(im, 0)
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)
        elif isinstance(im, torch.Tensor):
            pass
        else:
            raise Exception("ERROR❗Detector supports only these input formats: numpy.array, torch.Tensor")
        y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        if self.device != 'cpu':
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(1):
                self.forward(im)  # warmup


def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        imgsz = list(imgsz)  # convert to list if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


# class LoadImages:
#     # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
#     def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
#         # if isinstance(path, str) and Path(path).suffix == '.txt':  # *.txt file with img/vid/dir on each line
#         #     path = Path(path).read_text().rsplit()
#         # files = []
#         # for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
#         #     p = str(Path(p).resolve())
#         #     if '*' in p:
#         #         files.extend(sorted(glob.glob(p, recursive=True)))  # glob
#         #     elif os.path.isdir(p):
#         #         files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
#         #     elif os.path.isfile(p):
#         #         files.append(p)  # files
#         #     else:
#         #         raise FileNotFoundError(f'{p} does not exist')
#         #
#         # IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', 'heic'  # include image suffixes
#         # VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
#         #
#         # images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
#         # videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
#         # ni, nv = len(images), len(videos)
#
#         self.img_size = img_size
#         self.stride = stride
#         self.files = path
#         # self.nf = ni + nv  # number of files
#         # self.video_flag = [False] * ni + [True] * nv
#         # self.mode = 'image'
#         self.auto = auto
#         # self.transforms = transforms  # optional
#         # self.vid_stride = vid_stride  # video frame-rate stride
#         # if any(videos):
#         #     self._new_video(videos[0])  # new video
#         # else:
#         #     self.cap = None
#         # assert self.nf > 0, f'No images or videos found in {p}. ' \
#         #                     f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'
#
#     def __iter__(self):
#         self.count = 0
#         return self
#
#     def __next__(self):
#         if self.count == self.nf:
#             raise StopIteration
#         im0 = self.files
#         im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
#         im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         im = np.ascontiguousarray(im)  # contiguous
#
#         return im, im0
#
#     def _new_video(self, path):
#         # Create a new video capture object
#         self.frame = 0
#         self.cap = cv2.VideoCapture(path)
#         self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
#         self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
#         # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493
#
#     def _cv2_rotate(self, im):
#         # Rotate a cv2 video manually
#         if self.orientation == 0:
#             return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
#         elif self.orientation == 180:
#             return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         elif self.orientation == 90:
#             return cv2.rotate(im, cv2.ROTATE_180)
#         return im
#
#     def __len__(self):
#         return self.nf  # number of files


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()



def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        # use cv2
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    def box_label(self, i, box, label='', color=(0, 0, 255), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        # cv2
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            label = f'{i} {label}'
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

    def masks(self, masks, colors, im_gpu, alpha=0.5, retina_masks=False):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        if self.pil:
            # convert to numpy first
            self.im = np.asarray(self.im).copy()
        if len(masks) == 0:
            self.im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        colors = torch.tensor(colors, device=im_gpu.device, dtype=torch.float32) / 255.0
        colors = colors[:, None, None]  # shape(n,1,1,3)
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

        inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

        im_gpu = im_gpu.flip(dims=[0])  # flip channel
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255).byte().cpu().numpy()
        self.im[:] = im_mask if retina_masks else scale_image(im_gpu.shape, im_mask, self.im.shape)
        if self.pil:
            # convert im back to PIL and update draw
            self.fromarray(self.im)

    def rectangle(self, xy, fill=None, outline=None, width=1):
        # Add rectangle to image (PIL-only)
        self.draw.rectangle(xy, fill, outline, width)

    def text(self, xy, text, txt_color=(255, 255, 255), anchor='top'):
        # Add text to image (PIL-only)
        if anchor == 'bottom':  # start y from font bottom
            w, h = self.font.getsize(text)  # text width, height
            xy[1] += 1 - h
        self.draw.text(xy, text, fill=txt_color, font=self.font)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    if prediction is None:
        print('No detection for the source')
        return None

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def draw_boxes(image, boxes, classes, scores):
    """
    Draws bounding boxes with class and confidence on the input image.

    Parameters:
        image: numpy array
            The input image to draw bounding boxes on.
        boxes: numpy array
            A numpy array of shape (N, 4) containing the coordinates of N bounding boxes in the format [x1, y1, x2, y2].
        classes: numpy array
            A numpy array of shape (N,) containing the class labels for each bounding box.
        scores: numpy array
            A numpy array of shape (N,) containing the confidence scores for each bounding box.

    Returns:
        numpy array
            The resulting image with the drawn bounding boxes, class labels, and confidence scores.
    """
    # Convert the image to a format that can be drawn on (i.e., convert from BGR to RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Iterate over each bounding box and draw it on the image
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        class_name = str(classes[i])
        score = scores[i]
        color = (0, 255, 0)  # Set the color to green
        thickness = 2  # Set the thickness of the bounding box lines

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw the class label and confidence score on the bounding box
        label = f"{class_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x1
        text_y = y1 - text_size[1]
        cv2.putText(image, label, (text_x, text_y), font, font_scale, color, thickness)

    # Convert the image back to BGR format for consistency with the input format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


def save_one_box(xyxy, im, gain=1.02, pad=10, square=False, BGR=False):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    return crop


class GreenhouseModel:
    def __init__(self,
                 classifier=SimpleMobileNetV2,
                 classifier_weights='detection_pipeline/weights/classifier.pt',
                 classifier_labels='detection_pipeline/dataset/labels.pkl',
                 detector_weights='detection_pipeline/weights/yolo.pt',  # model path or triton URL
                 source=None,  # file/dir/URL/glob/screen/0(webcam)
                 data=None,  # dataset.yaml path
                 imgsz=(800, 800),  # inference size (height, width)
                 conf_thres=0.05,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 view_img=False,  # show results
                 save_txt=False,  # save results to *.txt
                 save_conf=False,  # save confidences in --save-txt labels
                 save_crop=True,  # save cropped prediction boxes
                 nosave=False,  # do not save images/videos
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 update=False,  # update all models
                 project='runs/detect',  # save results to project/name
                 name='exp',  # save results to project/name
                 exist_ok=False,  # existing project/name ok, do not increment
                 line_thickness=3,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 vid_stride=1,  # video frame-rate stride
                 ):

        self.source = None,  # file/dir/URL/glob/screen/0(webcam)
        self.data = None,  # dataset.yaml path
        self.imgsz = (800, 800),  # inference size (height, width)
        self.conf_thres = conf_thres,  # confidence threshold
        self.iou_thres = iou_thres,  # NMS IOU threshold
        self.max_det = max_det,  # maximum detections per image
        self.device = str(device),  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = False,  # show results
        self.save_txt = False,  # save results to *.txt
        self.save_conf = False,  # save confidences in --save-txt labels
        self.save_crop = True,  # save cropped prediction boxes
        self.nosave = False,  # do not save images/videos
        self.classes = None,  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False,  # class-agnostic NMS
        self.augment = False,  # augmented inference
        self.visualize = False,  # visualize features
        self.update = False,  # update all models
        self.project = 'runs/detect',  # save results to project/name
        self.name = 'exp',  # save results to project/name
        self.exist_ok = False,  # existing project/name ok, do not increment
        self.line_thickness = 3,  # bounding box thickness (pixels)
        self.hide_labels = False,  # hide labels
        self.hide_conf = False,  # hide confidences
        self.half = False,  # use FP16 half-precision inference
        self.dnn = False,  # use OpenCV DNN for ONNX inference
        self.vid_stride = 1,  # video frame-rate stride

        print('Building detection model..')
        self.detector = DetectMultiBackend(weights=detector_weights, device=device, dnn=dnn, data=data, fp16=half)
        self.stride = self.detector.stride
        self.names = self.detector.names
        self.imgsz = imgsz # check image size
        self.detector_warmup = 0

        print('Building classification model..')
        self.classifier = Model(model=classifier, weights=classifier_weights, labels=classifier_labels,
                                height=224, width=224)
        
        self.detector.to(device)

    def predict(self, source):
        if isinstance(source, str):
            source = cv2.imread(source)
        if self.detector_warmup == 0:
            self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
            self.detector.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
            self.detector_warmup = 1
        im0 = source.copy()
        im = letterbox(im0, self.imgsz, self.stride)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.detector.device)
        im = im.half() if self.detector.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im.unsqueeze(0)  # expand for batch dim
        assert len(im.shape) == 4, f"Wrong shape of input tensor: {im.shape}, but must be (b, c, h, w)"
        # Inference
        pred = self.detector.model(im)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres[0], self.iou_thres[0], self.classes[0], self.agnostic_nms[0],
                                   max_det=self.max_det[0])
        if pred is None:
            return im0
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, line_width=self.line_thickness[0], example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                dt = 0
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    crop = save_one_box(xyxy=xyxy, im=im0, BGR=True)
                    y = self.classifier.predict(crop, decoded=False)
                    leaf_disease_confidence = torch.nn.functional.softmax(y).max().detach().cpu().numpy().item()
                    leaf_disease_name = self.classifier.labels[y.argmax(axis=1).detach().cpu().numpy().item()]

                    label = f'{self.names[c]}:{conf:.2f} | {leaf_disease_name}:{leaf_disease_confidence:.2f}'

                    if leaf_disease_name == 'good_leafs':
                        annotator.box_label(i=dt, box=xyxy, label=label, color=(0, 255, 0))
                    else:
                        annotator.box_label(i=dt, box=xyxy, label=label)
                    dt += 1
            # Stream results
            im0 = annotator.result()
        return im0





# def run(
#         weights=None,  # model path or triton URL
#         source=None,  # file/dir/URL/glob/screen/0(webcam)
#         data=None,  # dataset.yaml path
#         imgsz=(640, 640),  # inference size (height, width)
#         conf_thres=None,  # confidence threshold
#         iou_thres=None,  # NMS IOU threshold
#         max_det=1000,  # maximum detections per image
#         device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         view_img=False,  # show results
#         save_txt=False,  # save results to *.txt
#         save_conf=False,  # save confidences in --save-txt labels
#         save_crop=True,  # save cropped prediction boxes
#         nosave=False,  # do not save images/videos
#         classes=None,  # filter by class: --class 0, or --class 0 2 3
#         agnostic_nms=False,  # class-agnostic NMS
#         augment=False,  # augmented inference
#         visualize=False,  # visualize features
#         update=False,  # update all models
#         project='runs/detect',  # save results to project/name
#         name='exp',  # save results to project/name
#         exist_ok=False,  # existing project/name ok, do not increment
#         line_thickness=3,  # bounding box thickness (pixels)
#         hide_labels=False,  # hide labels
#         hide_conf=False,  # hide confidences
#         half=False,  # use FP16 half-precision inference
#         dnn=False,  # use OpenCV DNN for ONNX inference
#         vid_stride=1,  # video frame-rate stride
# ):
#     # Load model
#     device = str(device)
#     model = DetectMultiBackend(weights=weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride, names = model.stride, model.names
#     imgsz = check_img_size(imgsz, s=stride)  # check image size
#     model.warmup(imgsz=(1, 3, *imgsz))  # warmup
#
#     # TODO: LOOP WITH FRAMES TO BE HERE!
#     im0 = source
#     im = letterbox(im0, imgsz, stride)[0]  # padded resize
#     im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#     im = np.ascontiguousarray(im)  # contiguous
#     im = torch.from_numpy(im).to(model.device)
#     im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#     im /= 255  # 0 - 255 to 0.0 - 1.0
#     if len(im.shape) == 3:
#         im = im.unsqueeze(0) # expand for batch dim
#     assert len(im.shape) == 4, f"Wrong shape of input tensor: {im.shape}, but must be (b, c, h, w)"
#     # Inference
#     pred = model.model(im, augment=augment, visualize=visualize)
#     # NMS
#     pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
#     for i, det in enumerate(pred):  # per image
#         annotator = Annotator(im0, line_width=line_thickness, example=str(names))
#         if len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
#             # Write results
#             leaf = 0
#             for *xyxy, conf, cls in reversed(det):
#                 c = int(cls)  # integer class
#                 crop = save_one_box(xyxy=xyxy, im=im0, BGR=True)
#                 y = classifier.predict(crop, decoded=False)
#                 leaf_disease_confidence = torch.nn.functional.softmax(y).max().detach().cpu().numpy().item()
#                 leaf_disease_name = classifier.labels[y.argmax(axis=1).detach().cpu().numpy().item()]
#
#                 label = f'{names[c]}:{conf:.2f} | {leaf_disease_name}:{leaf_disease_confidence:.2f}'
#
#                 if leaf_disease_name == 'good_leafs':
#                     annotator.box_label(i=leaf, box=xyxy, label=label, color=(0, 255, 0))
#                 else:
#                     annotator.box_label(i=leaf, box=xyxy, label=label)  # i, box, label='', color=(0, 0, 255), txt_color=(255, 255, 255)
#                 leaf += 1
#         # Stream results
#         im0 = annotator.result()
#     return im0


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
#     parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
#     parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     # print_args(vars(opt))
#     return opt
#
#
# def main(opt):
#     leaves = run(**vars(opt))
#
#
# if __name__ == '__main__':
#     opt = parse_opt()
#     main(opt)

# if __name__ == '__main__':

    # for f in tqdm.tqdm(files):
    #     leaves = run(weights='classifier.pt',  # model path or triton URL
    #                 source=f, # '/Users/yauhenikavaliou/PycharmProjects/leaf_desease/20230313_134213.jpg',  # file/dir/URL/glob/screen/0(webcam)
    #                 data=None,  # dataset.yaml path
    #                 imgsz=(640, 640),  # inference size (height, width)
    #                 conf_thres=0.25,  # confidence threshold
    #                 iou_thres=0.45,  # NMS IOU threshold
    #                 max_det=1000,  # maximum detections per image
    #                 device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    #                 view_img=False,  # show results
    #                 save_txt=False,  # save results to *.txt
    #                 save_conf=False,  # save confidences in --save-txt labels
    #                 save_crop=True,  # save cropped prediction boxes
    #                 nosave=False,  # do not save images/videos
    #                 classes=None,  # filter by class: --class 0, or --class 0 2 3
    #                 agnostic_nms=False,  # class-agnostic NMS
    #                 augment=False,  # augmented inference
    #                 visualize=False,  # visualize features
    #                 update=False,  # update all models
    #                 project='runs/detect',  # save results to project/name
    #                 name='exp',  # save results to project/name
    #                 exist_ok=False,  # existing project/name ok, do not increment
    #                 line_thickness=3,  # bounding box thickness (pixels)
    #                 hide_labels=False,  # hide labels
    #                 hide_conf=False,  # hide confidences
    #                 half=False,  # use FP16 half-precision inference
    #                 dnn=False,  # use OpenCV DNN for ONNX inference
    #                 vid_stride=1,  # video frame-rate stride
    #                 )
    #     print(len(leaves))

if __name__ == '__main__':


    p = '/Users/yauhenikavaliou/PycharmProjects/leaf_desease/data/13-3-23/20230313_133721.jpg'
    # files = os.listdir(p)
    # files = [os.path.join(p, f) for f in files]
    # det = 0
    frame = cv2.imread(p)
    # for f in files:
    leaves = run(weights='yolo.pt',  # model path or triton URL
                 source=frame, #'/Users/yauhenikavaliou/PycharmProjects/leaf_desease/20230313_134213.jpg',  # file/dir/URL/glob/screen/0(webcam)
                 data=None,  # dataset.yaml path
                 imgsz=(800, 800),  # inference size (height, width)
                 conf_thres=0.01,  # confidence threshold
                 iou_thres=0.20,  # NMS IOU threshold
                 max_det=150,  # maximum detections per image
                 device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 view_img=False,  # show results
                 save_txt=False,  # save results to *.txt
                 save_conf=False,  # save confidences in --save-txt labels
                 save_crop=True,  # save cropped prediction boxes
                 nosave=False,  # do not save images/videos
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 update=False,  # update all models
                 project='runs/detect',  # save results to project/name
                 name='exp',  # save results to project/name
                 exist_ok=False,  # existing project/name ok, do not increment
                 line_thickness=4,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 vid_stride=1,  # video frame-rate stride
                 )
    cv2.imwrite('iamge.jpg', leaves)

# TODO:
#  1. set yolo
#  2. set classifier
#  3. detect and get cropped bboxes
#  3.1. classify cropped bboxes
#  3.2. draw boxes with classified classes on the input frame/image
#  4. show up the frame/image with bboxes