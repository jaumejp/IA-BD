
import numpy as np

from PIL import Image, ImageFile
from PIL import ExifTags, ImageOps

# +++ bbox functions

def bbox2tlbr(bbox):
    return np.array([(bbox[0], bbox[1]), (bbox[2], bbox[3])], dtype = 'int')

def tlbr2bbox(tlbr):
    return (tlbr[0][0], tlbr[0][1], tlbr[1][0], tlbr[1][1])

def sqrbbox(tlbr, size):
    # image width, height
    img_w, img_h = size
    # bbox width, height
    bbx_w, bbx_h = (tlbr[1][0] -tlbr[0][0]), (tlbr[1][1] -tlbr[0][1])
    # maxim bbox side
    if bbx_h > bbx_w:
        bbx_s = max(bbx_h, 256)
    else:
        bbx_s = max(bbx_w, 256)
    # center of current bbox
    x0, y0 = tlbr[0][0] + bbx_w /2, tlbr[0][1] + bbx_h /2
    # minimal square bbox (with maximal side)
    left = x0 -bbx_s /2
    rght = left +bbx_s
    # when the bbox is highly rectangular it may be impossible to fit a square bbox with maximal side
    if left < 0:      # move bbox to the right up to left = 0
        rght += (0 -left)
        left = 0
    if rght > img_w: # move bbox to the left down to right = img_w
        left -= (rght -img_w)
        rght = img_w
    top = y0 -bbx_s /2
    btm = top +bbx_s
    if top < 0:       # move bbox down to top = 0
        btm += (0 -top)
        top = 0
    if btm > img_h:   # move bbox up to btm = img_h
        top -= (btm -img_h)
        btm = img_h
    return (left, top, rght, btm)

def compute_IoU(imgSize, true_bbox, pred_bbox, verbose = False):
    # gt_mask
    true = np.zeros((imgSize[1], imgSize[0]), dtype=np.uint8)
    (bbx_xtl, bbx_ytl), (bbx_xbr, bbx_ybr) = true_bbox
    true[bbx_ytl:bbx_ybr, bbx_xtl:bbx_xbr] = 1
    # pred_bbox mask
    pred = np.zeros((imgSize[1], imgSize[0]), dtype=np.uint8)
    (prd_xtl, prd_ytl), (prd_xbr, prd_ybr) = pred_bbox
    pred[prd_ytl:prd_ybr, prd_xtl:prd_xbr] = 1
    # IoU
    union = np.bitwise_or(true, pred)
    intersection = np.bitwise_and(true, pred)
    if verbose:
        print(np.sum(intersection), np.sum(union))
    return np.sum(intersection) /(np.sum(union) +1)

# +++ image open function with orientation tag check

def imageOpen(imagePath):

    pilImg = Image.open(imagePath).convert('RGB')

    img_exif = pilImg.getexif()
    orientation = 0
    if img_exif is not None:
        for key, val in img_exif.items():
            if key in ExifTags.TAGS and key == 274:
                orientation = val
    if orientation > 1:
        pilImg = ImageOps.exif_transpose(pilImg)

    return pilImg

# +++ score normalization

def normScore(filterScores, filterSettings):

    '''
    - normalized class score (ranging from -1 to 1) is the FPR[score] relative to the FPR[threshold]
    .  0 meaning that the class score is just on the threshold of acceptance
    '''

    def norm_score(s, species):

        score = filterScores[s]
        thrsh = filterSettings['threshold'][s]
        FPR = filterSettings['ROCs'][species]['FPR']

        score, thrsh = int(score *100), int(thrsh *100)
        if score >= thrsh:
            return (FPR[thrsh] -FPR[score]) /(FPR[thrsh] -0.0)
        else:
            return -(FPR[score] -FPR[thrsh]) /(1.0 -FPR[thrsh])

    norm_scores = np.array([norm_score(s, species) for s, species in enumerate(filterSettings['classes'])])
    bestScoreClass = np.argsort(norm_scores)[-1]

    if norm_scores[bestScoreClass] < 0:
        return (len(filterSettings['classes']), 'not-classified', norm_scores[bestScoreClass])
    else:
        return (bestScoreClass, filterSettings['classes'][bestScoreClass], norm_scores[bestScoreClass])

