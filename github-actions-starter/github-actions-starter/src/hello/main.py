from . import greet
import argparse


def main():
    parser = argparse.ArgumentParser(description="Say hello from a tiny demo app.")
    parser.add_argument("--name", default="World", help="Name to greet (default: World)")
    args = parser.parse_args()
    print(greet(args.name))


if __name__ == "__main__":
    main()
