import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--checkpoints", type=str, default="wanet_bd/")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--save_dir", type=str, default="saved/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", type=bool, default=False)

    parser.add_argument("--topk", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="breast_ultrasound")
    parser.add_argument("--label_count", type=int, default=3)
    parser.add_argument("--input_height", type=int, default=300)
    parser.add_argument("--input_width", type=int, default=300)

    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=float, default=2)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_period", type=int, default=20)

    return parser
