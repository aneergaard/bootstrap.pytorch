import os
import sys
import yaml
import json
import copy
import argparse
import collections
from .utils import merge_dictionaries

# Options is a singleton
# https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

class Options(object):

    # Attributs

    __instance = None # singleton
    options = None # dictionnary of the singleton

    class HelpParser(argparse.ArgumentParser):
        def error(self, message):
            print('\nError: %s\n' % message)
            self.print_help()
            sys.exit(2)

    # Build the singleton as Options()

    def __new__(self, path_yaml=None, arguments_callback=None):
        if not Options.__instance:
            Options.__instance = object.__new__(Options)

            if path_yaml:
                Options.__instance.options = Options.load_yaml_opts(path_yaml)

            else:
                try:
                    optfile_parser = argparse.ArgumentParser(add_help=False)
                    if optfile_parser.parse_known_args()[1][0] == '-h':
                        print('/!\\ -o/--path_opts needed to load the yaml options file')
                    fullopt_parser = Options.HelpParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

                    optfile_parser.add_argument('-o', '--path_opts', type=str, required=True)
                    fullopt_parser.add_argument('-o', '--path_opts', type=str, required=True)


                    options_yaml = Options.load_yaml_opts(optfile_parser.parse_known_args()[0].path_opts)
                    Options.__instance.add_options(fullopt_parser, options_yaml)

                    arguments = fullopt_parser.parse_args()
                    if arguments_callback:
                        arguments = arguments_callback(Options.__instance, arguments, options_yaml)

                    Options.__instance.options = {}
                    for argname in vars(arguments):
                        nametree = argname.split('.')
                        value = getattr(arguments, argname)

                        position = Options.__instance.options
                        for piece in nametree[:-1]:
                            if piece in position and isinstance(position[piece], collections.Mapping):
                                position = position[piece]
                            else:
                                position[piece] = {}
                                position = position[piece]
                        position[nametree[-1]] = value

                except:
                    Options.__instance = None
                    raise

        return Options.__instance


    def __getattr__(self, name):
        return self.options[name]


    def __str__(self):
        return json.dumps(self.options, indent=2)


    def add_options(self, parser, options, prefix=''):
        if prefix:
            prefix += '.'

        for key, value in options.items():
            #try:
            if isinstance(value, dict):
                self.add_options(parser, value, '{}{}'.format(prefix, key))
            else:
                argname = '--{}{}'.format(prefix, key)
                nargs = '*' if isinstance(value, list) else '?'
                if value is None:
                    datatype = str
                elif isinstance(value, bool):
                    datatype = self.str_to_bool
                elif isinstance(value, list):
                    if len(value) == 0:
                        datatype = str
                    else:
                        datatype = type(value[0])
                else:
                    datatype = type(value)
                #datatype = str if value is None else type(value[0]) if isinstance(value, list) else type(value)
                parser.add_argument(argname, help='Default: %(default)s', default=value, nargs=nargs, type=datatype)
            # except:
            #     import ipdb; ipdb.set_trace()


    def str_to_bool(self, v):
        true_strings = ['yes', 'true']#, 't', 'y', '1')
        false_strings = ['no', 'false']#, 'f', 'n', '0')
        if isinstance(v, str):
            if v.lower() in true_strings:
                return True
            elif v.lower() in false_strings:
                return False
        raise argparse.ArgumentTypeError('{} cant be converted to bool ('.format(v)+'|'.join(true_strings+false_strings)+' can be)')


    def save(self, path_yaml):
        Options.save_yaml_opts(self.options, path_yaml)

    # Static methods

    def load_yaml_opts(path_yaml):
        # TODO: include the parent options when seen, not after having loaded the main options
        result = {}
        with open(path_yaml, 'r') as yaml_file:
            options_yaml = yaml.load(yaml_file)
            includes = options_yaml.get('__include__', False)
            if includes:
                if type(includes) != list:
                    includes = [includes]
                for include in includes:
                    parent = Options.load_yaml_opts('{}/{}'.format(os.path.dirname(path_yaml), include))
                    merge_dictionaries(result, parent)
            merge_dictionaries(result, options_yaml) # to be sure the main options overwrite the parent options
        result.pop('__include__', None)
        return result

    def save_yaml_opts(opts, path_yaml):
        # Warning: copy is not nested
        # TODO: save the options in same order they have been viewed
        options = copy.copy(opts)
        del options['path_opts']
        with open(path_yaml, 'w') as yaml_file:
            yaml.dump(options, yaml_file, default_flow_style=False)


