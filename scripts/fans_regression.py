import argparse as arg
import sys
import traceback


def main() -> int:
    """Regression to predict the number of fan per asking machine."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="traceback mode")
        av = av.parse_args()
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc(), file=sys.stderr)
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
