from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os
# import re
# from distutils.command.install_lib import install_lib as _install_lib
#
#
# def batch_rename(src, dst, src_dir_fd=None, dst_dir_fd=None):
#     '''Same as os.rename, but returns the renaming result.'''
#     os.rename(src, dst,
#               src_dir_fd=src_dir_fd,
#               dst_dir_fd=dst_dir_fd)
#     return dst
#
#
# class _CommandInstallCythonized(_install_lib):
#     def __init__(self, *args, **kwargs):
#         _install_lib.__init__(self, *args, **kwargs)
#
#     def install(self):
#         # let the distutils' install_lib do the hard work
#         outfiles = _install_lib.install(self)
#         # batch rename the outfiles:
#         # for each file, match string between
#         # second last and last dot and trim it
#         matcher = re.compile('\.([^.]+)\.so$')
#         return [batch_rename(file, re.sub(matcher, '.so', file))
#                 for file in outfiles]
#

setup(
    ext_modules=cythonize(["solver_c.pyx", "types_common.pyx", "boat.pyx", "paddle.pyx", "misc_methods.pyx", "tf.pyx", "controller.pyx"], language_level="3"),
    include_dirs = [numpy.get_include()], requires=['numpy']
    # cmdclass={
    #     'install_lib': _CommandInstallCythonized,
    # },
)
