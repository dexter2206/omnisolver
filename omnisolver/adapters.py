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
        self.init_args_spec = specification["init_args"]
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
        kwargs = {arg_spec["name"]: getattr(cmd_args, arg_spec["name"]) for arg_spec in self.sample_args_spec}
        return getattr(module, self.class_name)(**kwargs)

    def add_argparse_subparser(self, root_group: argparse._SubParsersAction, parent: argparse.ArgumentParser):
        parser = root_group.add_parser(self.parser_name, parents=[parent], add_help=False)

        for arg_spec in self.sample_args_spec:
            parser.add_argument(f"--{arg_spec['name']}", help=arg_spec["help"], type=self.type_mapping[arg_spec["type"]])

        for arg_spec in self.init_args_spec:
            parser.add_argument(f"--{arg_spec['name']}", help=arg_spec["help"], type=self.type_mapping[arg_spec["type"]])

        parser.set_defaults(sample=self.sample)

    def sample(self, cmd_args) -> dimod.SampleSet:
        sampler = self.create_sampler(cmd_args)
        kwargs = {arg_spec["name"]: getattr(cmd_args, arg_spec["name"]) for arg_spec in self.sample_args_spec}
        with open(cmd_args.input) as bqm_file:
            bqm = dimod.BinaryQuadraticModel.from_coo(cmd_args.input, vartype=cmd_args.vartype)
        return sampler.sample_qubo(bqm, **kwargs)
