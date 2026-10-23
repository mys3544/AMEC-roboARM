"""lerobot-train on the Windows laptop: LeRobot marks the newest checkpoint with a symlink
(checkpoints/last), which Windows refuses without Developer Mode or admin (WinError 1314), and the
run dies at its first save. Here checkpoints/last.txt names the newest step instead; the rest is
lerobot-train's own, its arguments as they are (2026-10-23, LeRobot 0.6.1 in ~/.roboarm/lerobot):

    ~/.roboarm/lerobot/Scripts/python.exe train/lerobot_train_win.py --dataset.repo_id=local/NAME
        --dataset.root=m3pro/out/lerobot/NAME --policy.type=act --policy.push_to_hub=false ...

Not for hekate or Linux, where lerobot-train itself does.
"""

from pathlib import Path

from lerobot.scripts import lerobot_train


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last = checkpoint_dir.parent / "last.txt"
    last.write_text(checkpoint_dir.name)
    return last


lerobot_train.update_last_checkpoint = update_last_checkpoint

if __name__ == "__main__":
    lerobot_train.main()
