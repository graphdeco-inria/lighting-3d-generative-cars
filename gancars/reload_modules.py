from training.triplane import TriPlaneGenerator
from torch_utils import misc

def reload_modules(net, verbose=True):
    if verbose:
        print("Reloading Modules!")
    net_new = TriPlaneGenerator(*net.init_args,**net.init_kwargs).eval().requires_grad_(False)
    misc.copy_params_and_buffers(net, net_new, require_all=True)
    net_new.neural_rendering_resolution = net.neural_rendering_resolution
    net_new.rendering_kwargs = net.rendering_kwargs
    net = net_new
    net.rendering_kwargs['ray_start'] = 'auto'
    net.rendering_kwargs['ray_end'] = 'auto'
    return net