"""Definition of adapter protocols and other things common to all adapters."""
import argparse
import importlib
import dimod
from typing_extensions import Protocol


class Adapter(Protocol):

    def __init__(self) -> None:
        pass

    def is_available(self) -> bool:
        pass

    def create_sampler(self, cmd_args) -> dimod.Sampler:
        pass

    def sample(self, sampler: dimod.Sampler, cmd_args) -> dimod.SampleSet:
        pass

    def add_argparse_subparser(self, root_group: argparse._SubParsersAction, parent: argparse.ArgumentParser):
        pass



class SimpleAdapter:

    type_mapping = {
        "bool": bool,
        "int": int,
        "float": float
    }

    def __init__(self, specification) -> None:
        if specification["schema_version"] != 1:
            raise ValueError("Unknown version of specification file.")
        self.sample_args_spec = specification["sample_args"]
        self.module_path, self.class_name = specification["sampler_class"].rsplit(".", 1)
        self.parser_name = specification["parser_name"]
        self.description = specification["description"]

    def load_sampler_module(self):
        print(self.module_path)
        return importlib.import_module(self.module_path)

    def is_available(self) -> bool:
        try:
            self.load_sampler_module()
            return True
        except ImportError:
            return False

    def create_sampler(self, cmd_args) -> dimod.Sampler:
        module = self.load_sampler_module()
        # TODO: We only handle sample args. It would be cool if we handled __init__ args too.
        return getattr(module, self.class_name)()  # Initialize new instance of our sampler class.

    def add_argparse_subparser(self, root_group: argparse._SubParsersAction, parent: argparse.ArgumentParser):
        parser = root_group.add_parser(self.parser_name, parents=[parent], add_help=False)
        for arg_spec in self.sample_args_spec:
            parser.add_argument(f"--{arg_spec['name']}", help=arg_spec["help"], type=self.type_mapping[arg_spec["type"]])

    def sample(self, sampler, cmd_args) -> dimod.SampleSet:
        pass
