"""Command line interface for omnisolver."""
import argparse
from typing import Iterable
from pkg_resources import resource_listdir, resource_stream
from omnisolver.adapters import Adapter, SimpleAdapter
import yaml


def get_all_adapters() -> Iterable[Adapter]:
    adapters = []
    for entry in resource_listdir("omnisolver", "specifications"):
        stream = resource_stream("omnisolver", f"specifications/{entry}")
        adapters.append(SimpleAdapter(yaml.safe_load(stream)))

    return adapters


def main():
    """Entrypoint of omnisolver."""
    root_parser = argparse.ArgumentParser()
    common_parser = argparse.ArgumentParser()
    common_parser.add_argument(
        "--input",
        help="Path of the input BQM file in BQM format. If not specified, stdin is used.",
        type=argparse.FileType("r"),
        default="-",
    )
    common_parser.add_argument(
        "--output",
        help="Path of the output file. If not specified, stdout is used.",
        type=argparse.FileType("w"),
        default="-",
    )
    common_parser.add_argument(
        "--vartype", help="Variable type", choices=["SPIN", "BINARY"], default="BINARY"
    )

    solver_commands = root_parser.add_subparsers(title="Solvers")

    adapters = get_all_adapters()
    for adapter in adapters:
        if adapter.is_available():
            adapter.add_argparse_subparser(solver_commands, parent=common_parser)

    args = root_parser.parse_args()

    if not hasattr(args, "sample"):
        print("Missing argument. You need to supply solver name.")
        root_parser.print_usage()
        exit(1)

    result = args.sample(args)

    result.to_pandas_dataframe().to_csv(args.output)


if __name__ == "__main__":
    main()
