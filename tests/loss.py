import torch
import numpy as np

def cost_func(x: torch.tensor, y: torch.tensor):
    '''Example 2-d cost function for visualization
    
    '''
    # slight negative gradient across the whole surface
    z = 1.0*x + 0.75*y

    # two local minima near (0, 0)
    z -= __f1(x, y)

    # 3rd local minimum at (-0.5, -0.8)
    z -= -1 * __f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.35, y_sig=0.35)

    # one steep gaussian trench at (0, 0)
    z -= __f2(x, y, x_mean=0, y_mean=0, x_sig=0.2, y_sig=0.2)

    # three steep gaussian trenches
    z -= __f2(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= __f2(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
    z -= __f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)

    return z


# noisy hills of the cost function
def __f1(x, y):
    if isinstance(x, torch.Tensor):
        return -1 * (x * x).sin() * (3 * y * y).cos() * (-(x * y) * (x * y)).exp() - (-(x + y) * (x + y)).exp()
    elif isinstance(x, np.ndarray):
        return -1 * np.sin(x * x) * np.cos(3 * y * y) * np.exp(-(x * y) * (x * y)) - np.exp(-(x + y) * (x + y))
    else:
        raise Exception('x must be a pytorch tensor or numpy array')

# bivar gaussian hills of the cost function
def __f2(x, y, x_mean, y_mean, x_sig, y_sig):

    if isinstance(x, torch.Tensor):
        normalizing = 1 / (2 * np.pi * x_sig * y_sig)
        x_exp = (-1 * (x - x_mean).pow(2)) / (2 * (x_sig**2))
        y_exp = (-1 * (y - y_mean).pow(2)) / (2 * (y_sig**2))
        return normalizing * (x_exp + y_exp).exp()
    
    elif isinstance(x, np.ndarray):

        normalizing = 1 / (2 * np.pi * x_sig * y_sig)
        x_exp = (-1 * np.square(x - x_mean)) / (2 * (x_sig**2))
        y_exp = (-1 * np.square(y - y_mean)) / (2 * (y_sig**2))
        return normalizing * np.exp(x_exp + y_exp)

    else:
        raise Exception('x must be a pytorch tensor or numpy array')