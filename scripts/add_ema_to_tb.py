"""从已有 TensorBoard 日志中读取 train/loss，计算 EMA 后写回 train/ema_loss。"""

import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb_dir", required=True, help="TensorBoard log directory")
    parser.add_argument("--decay", type=float, default=0.99, help="EMA decay factor")
    parser.add_argument("--source_tag", default="train/loss")
    parser.add_argument("--target_tag", default="train/ema_loss")
    args = parser.parse_args()

    ea = EventAccumulator(args.tb_dir)
    ea.Reload()

    loss_events = ea.Scalars(args.source_tag)
    print(f"Read {len(loss_events)} entries from '{args.source_tag}'")

    ema = None
    ema_data = []
    for e in loss_events:
        if ema is None:
            ema = e.value
        else:
            ema = args.decay * ema + (1 - args.decay) * e.value
        ema_data.append((e.wall_time, e.step, ema))

    writer = SummaryWriter(log_dir=args.tb_dir)
    for wall_time, step, val in ema_data:
        writer.file_writer.event_writer.add_event(
            _make_scalar_event(wall_time, step, args.target_tag, val)
        )
    writer.flush()
    writer.close()

    print(f"Wrote {len(ema_data)} entries to '{args.target_tag}'")
    print(f"  step {ema_data[0][1]}: {ema_data[0][2]:.6f}")
    print(f"  step {ema_data[-1][1]}: {ema_data[-1][2]:.6f}")


def _make_scalar_event(wall_time, step, tag, value):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary

    summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
    return Event(wall_time=wall_time, step=step, summary=summary)


if __name__ == "__main__":
    main()
