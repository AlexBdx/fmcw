from os.path import dirname, basename, isfile, join, abspath, split
import glob

path_glob = join(dirname(abspath(__file__)), '*.py')
modules = glob.glob(path_glob)
#valid_modules = [mod for mod in modules if mod[-3:] == '.py']
module_blacklist = ['__init__.py', 'setup.py', '190903_rt_fmcw3.py']
__name__ = "fmcw"
__version__ = "3.1.6"
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and split(f)[1] not in module_blacklist]
