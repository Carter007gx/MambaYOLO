# Ultralytics YOLO 🚀, AGPL-3.0 license
import torch
from torch import nn

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING, colorstr

try:
    # WARNING: do not move import due to protobuf issue in https://github.com/ultralytics/ultralytics/pull/4674
    from torch.utils.tensorboard import SummaryWriter

    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["tensorboard"] is True  # verify integration is enabled
    WRITER = None  # TensorBoard SummaryWriter instance

except (ImportError, AssertionError, TypeError):
    # TypeError for handling 'Descriptors cannot not be created directly.' protobuf errors in Windows
    SummaryWriter = None


def _log_scalars(scalars, step=0):
    """Logs scalar values to TensorBoard."""
    if WRITER:
        for k, v in scalars.items():
            WRITER.add_scalar(k, v, step)

def _log_tensorboard_hist(trainer):
    # BN param, not in the orin-yolov8, just for the image with sparity training!!!
    module_list = []
    # module_bias_list = []
    for i, layer in trainer.model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            bnw = layer.state_dict()['weight']
            bnb = layer.state_dict()['bias']
            module_list.append(bnw)
            # module_bias_list.append(bnb)
            # bnw = bnw.sort()
            # print(f"{i} : {bnw} : ")
    size_list = [idx.data.shape[0] for idx in module_list]

    bn_weights = torch.zeros(sum(size_list))
    # bnb_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in enumerate(size_list):
        bn_weights[index:(index + size)] = module_list[idx].data.abs().clone()
        # bnb_weights[index:(index + size)] = module_bias_list[idx].data.abs().clone()
        index += size
    # print(bn_weights)
    WRITER.add_histogram("gamma", bn_weights, trainer.epoch)

def _log_tensorboard_graph(trainer):
    """Log model graph to TensorBoard."""
    try:
        import warnings

        from ultralytics.utils.torch_utils import de_parallel, torch

        imgsz = trainer.args.imgsz
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        p = next(trainer.model.parameters())  # for device, type
        im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)  # input image (must be zeros, not empty)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)  # suppress jit trace warning
            WRITER.add_graph(torch.jit.trace(de_parallel(trainer.model), im, strict=False), [])
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ TensorBoard graph visualization failure {e}")


def on_pretrain_routine_start(trainer):
    """Initialize TensorBoard logging with SummaryWriter."""
    if SummaryWriter:
        try:
            global WRITER
            WRITER = SummaryWriter(str(trainer.save_dir))
            prefix = colorstr("TensorBoard: ")
            LOGGER.info(f"{prefix}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/")
        except Exception as e:
            LOGGER.warning(f"WARNING ⚠️ TensorBoard not initialized correctly, not logging this run. {e}")


def on_train_start(trainer):
    """Log TensorBoard graph."""
    if WRITER:
        _log_tensorboard_graph(trainer)


def on_train_epoch_end(trainer):
    """Logs scalar statistics at the end of a training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)
    _log_tensorboard_hist(trainer)


def on_fit_epoch_end(trainer):
    """Logs epoch metrics at end of training epoch."""
    _log_scalars(trainer.metrics, trainer.epoch + 1)


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_epoch_end": on_train_epoch_end,
    }
    if SummaryWriter
    else {}
)
