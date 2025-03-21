import argparse as arg
import os
import shutil as shu
import sys
import traceback

import pandas as pd


def best_energy_ratio(src: str, dst: str, margin: float = 0) -> None:
    """Copies the best energy efficient days from src to dst.

    Args:
        src is the directory where the search occurs
        dst is the directory where the files are copied
        margin is the margin around the ratio where files are selected
    Returns:
        Nothing
    """
    (rmin, rall) = ((float("+inf"), None), [])
    for f in os.listdir(src):
        df = pd.read_csv(os.path.join(src, f))
        (fj, iea) = (df.iloc[-1]["FaitJour"], df.iloc[-1]["IEA"])
        if isinstance(fj, float) and isinstance(iea, float) and fj != 0:
            r = iea / fj
            if r < rmin[0]:
                rall += [rmin]
                rmin = (r, f)
            else:
                rall += [(r, f)]
    if rmin[1] is not None:
        shu.copy2(os.path.join(src, rmin[1]), os.path.join(dst, rmin[1]))
        for (r, f) in rall:
            if rmin[0] <= r <= rmin[0] * (1 + margin):
                shu.copy2(os.path.join(src, f), os.path.join(dst, f))


def main() -> int:
    """Select a sample of the most energy-per-pieces efficient days."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av.add_argument("--dst", default="./", help="destination directory")
        av.add_argument("--src", default="./", help="src directory")
        av.add_argument("--margin", default=0, type=float, help="r margin")
        av = av.parse_args()
        best_energy_ratio(av.src, av.dst, av.margin)
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc(), file=sys.stderr)
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
