"""Command line interface for omnisolver."""
import argparse


def main():
    """Entrypoint of omnisolver."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path of the input BQM file in BQM format. If not specified, stdin is used.", type=argparse.FileType("r"), default="-")
    parser.add_argument("--output", help="Path of the output file. If not specified, stdout is used.", type=argparse.FileType("w"), default="-")
    parser.add_argument("--vartype", help="Variable type", choices=["SPIN", "BINARY"], default="BINARY")

    solver_commands = parser.add_subparsers(title="Solvers")

    parser.parse_args()


if __name__ == "__main__":
    main()
