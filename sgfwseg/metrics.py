import numpy as np


def confusion(label, pred, classes):
    valid = (label >= 0) & (label < classes)
    if label.shape != pred.shape:
        raise ValueError('Prediction and target shapes differ')
    if np.any((pred < 0) | (pred >= classes)):
        raise ValueError('Invalid predicted class IDs')
    return np.bincount(classes * label[valid].astype(int) + pred[valid].astype(int),
                       minlength=classes * classes).reshape(classes, classes)


def scores(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, union, out=np.full(len(tp), np.nan), where=union > 0)
    return dict(OA=float(tp.sum() / cm.sum()) if cm.sum() else None,
                mIoU=float(np.nanmean(iou)) if np.any(union) else None,
                IoU=[None if not np.isfinite(x) else float(x) for x in iou],
                confusion=cm.astype(int).tolist())
