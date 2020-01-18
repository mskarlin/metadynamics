import numpy as np
import torch
from torch.autograd import Function


class MetaDynamicsPlot():

    def __init__(self, cost_func, history, iteration, mag_scale=0.1, distance_scale=0.1):
        self.cost_func = cost_func
        self.history = history
        self.iteration = iteration
        self.mag_scale = mag_scale
        self.distance_scale = distance_scale

    def update_state(self, iteration, history):
        self.iteration = iteration
        self.history = history

    def numpy_metadynamics(self, x, y):
            """method designed to take larger numpy arrays for loss surfaces"""
            # use a tensor form to align with the formulation in the pytorch version
            tensor = np.stack((x, y), 0).reshape(2, x.shape[0]).transpose()
            directions = []
            scale = []
            for i in range(self.iteration+1):
                distance_sq = (
                    np.square(self.history[i]-tensor).sum(-1)+0.00001)
                loss = self.mag_scale * \
                    np.exp(-distance_sq / self.distance_scale)
                loss[loss < 0.0001] = 0.
                scale.append(loss)
            return sum(scale) + self.cost_func(x, y)[:, 0]


class MetaDynamics(torch.nn.Module):
    def __init__(self, input_shape, loss_func=None, max_iterations=100, mag_scale=0.1,
                 distance_scale=0.1, deposit_rate=10, initial_history=None):

        super(MetaDynamics, self).__init__()

        if initial_history is not None:
            self.history = torch.nn.Parameter(initial_history, requires_grad=False)
        else:
            self.history = torch.nn.Parameter(
                torch.stack([torch.zeros(*input_shape, requires_grad=False) for i in range(max_iterations)], 0),
                requires_grad=False)

        self.mag_scale = mag_scale
        self.distance_scale = distance_scale
        self.deposit_rate = deposit_rate
        self.max_iterations = max_iterations
        self.loss_func = loss_func

    def forward(self, input, curr_iteration, set_history=True):

        deposit_n = min(curr_iteration // self.deposit_rate -
                        1, self.max_iterations)

        # do we need to detach here?
        if set_history and curr_iteration % self.deposit_rate == 0 and curr_iteration > 0:
            self.history[deposit_n] = input.clone()
        print(f'HISTORY: {self.history}')
        md = MetaDynamicsFunction.apply(input, self.history, deposit_n, self.mag_scale, self.distance_scale)

        if self.loss_func:

            return self.loss_func(input) + md
        
        else:

            return md


class MetaDynamicsFunction(Function):

    @staticmethod
    def forward(ctx, input_tensor, history, deposit_n, mag_scale=0.1, distance_scale=0.1, distance_fragment=1e-6):

        ctx.save_for_backward(input_tensor)
        ctx.history = history
        ctx.deposit_n = deposit_n
        ctx.mag_scale = mag_scale
        ctx.distance_scale = distance_scale
        ctx.distance_fragment = distance_fragment

        # get the loss repulsion from the centers

        scale = []
        for i in range(deposit_n+1):
            distance_sq = (
                (history[i]-input_tensor).pow(2).sum()+distance_fragment)
            loss = mag_scale * (-distance_sq / distance_scale).exp()
            scale.append(loss)

        return sum(scale)

    @staticmethod
    def backward(ctx, grad_output):
        tensor, = ctx.saved_tensors
        history = ctx.history
        deposit_n, mag_scale, distance_scale, distance_fragment = ctx.deposit_n, ctx.mag_scale, ctx.distance_scale, ctx.distance_fragment
        # get the gradient repulsion from the centers
        directions = []
        for i in range(deposit_n+1):
            distance_sq = ((history[i]-tensor).pow(2).sum()+distance_fragment)
            grad = 2 * mag_scale * \
                (history[i]-tensor) * (-distance_sq /
                                       distance_scale).exp() / (distance_scale)
            directions.append(grad)

        print(f'grad_output: {grad_output}')
        print(f'directions: {directions}')

        return grad_output * sum(directions), None, None, None, None, None
