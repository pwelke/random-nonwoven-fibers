from os.path import *


def featurefolder(filename):
    folder, filename = split(filename)
    _, resource = split(folder)

    base = '/results/features/'
    if not exists(join(base, resource)):
        mkdir(join(base, resource))

    return join(base, resource, filename)
