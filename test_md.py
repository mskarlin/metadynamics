import torch
import numpy as np
from torch.autograd import gradcheck
from torch.autograd import Function
from functools import partial
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from metadynamics.loss import MetaDynamics, MetaDynamicsPlot
from tests.loss import cost_func
from tests.plotter import plot_settings, OptimizationPlotter, render_gif
from torch import optim


def small_test():
    # # this part is really simple -- it takes a tensor and applies the above methods
    x = torch.ones(2, 2, requires_grad=True)
    n_deposits, mag_scale, distance_scale  = 500, 0.1, 0.1

    cv_history = [torch.zeros(*x.shape, requires_grad=False) for i in range(n_deposits)]

    metadynamics = partial(MetaDynamics.apply, history=cv_history, iteration=0, mag_scale=mag_scale, distance_scale=distance_scale)

    # applied_x = metadynamics(x, cv_history, 4, 0, mag_scale, distance_scale)

    # print(applied_x)

    # applied_x_grad = applied_x.backward(x)

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = torch.randn(2,2,dtype=torch.double,requires_grad=True)

    test = gradcheck(metadynamics, input, eps=1e-6, atol=1e-4)
    
    print(test)


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


def nn_example():
    dtype = torch.double
    device = torch.device("cpu")
    # device = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 13, 5, 1

    # set up the metadynamics variables
    n_deposits, mag_scale, distance_scale  = 500, 0.1, 0.1

    # # this part is really simple -- it takes a tensor and applies the above methods
    # x = torch.ones(2, 2, requires_grad=True)

    metadynamics = MetaDynamics.apply

    # applied_x = metadynamics(x, cv_history, 4, 0, 0.1, 0.1)

    # print(applied_x)

    # applied_x_grad = applied_x.backward(x)

    # Create random Tensors to hold input and outputs.
    x, y = load_boston(return_X_y=True)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    scaler = StandardScaler()
    y = scaler.fit_transform(y.reshape(-1, 1))

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    # Create random Tensors for weights.
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    # this is the state storage, we don't want to track anything for it
    cv_history = [torch.zeros(*w1.shape, requires_grad=False) for i in range(n_deposits)]

    learning_rate = 1e-5
    for t in range(n_deposits):
        # To apply our Function, we use Function.apply method. We alias this as 'relu'.
        relu = MyReLU.apply

        # Forward pass: compute predicted y using operations; we compute
        # ReLU using our custom autograd operation.
        y_pred = relu(x.mm(w1)).mm(w2)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum() + metadynamics(w1, cv_history, t, mag_scale, distance_scale)
        # if t % 100 == 99:
        print(t, loss.item())
        print(f"w1: {w1}")

        # Use autograd to compute the backward pass.
        loss.backward()

        # Update weights using gradient descent
        with torch.no_grad():
            w1 -= learning_rate * w1.grad
            w2 -= learning_rate * w2.grad

            # Manually zero the gradients after updating weights
            w1.grad.zero_()
            w2.grad.zero_()

def visual_test(steps=100):

    ax = plot_settings()
    
    x = torch.tensor(0.75, requires_grad=True)
    y = torch.tensor(1.0, requires_grad=True)

    learning_rate = 0.1

    optimizer = optim.Adam([x,y], lr=0.1)
    # optimizer = optim.SGD([x,y], lr=0.05)
    op = OptimizationPlotter(ax)

    metadynamics = MetaDynamics.apply
    mag_scale, distance_scale = 1.0, 0.1
    cv_history = torch.stack([torch.zeros(*torch.stack((x,y), 0).shape, requires_grad=False) for i in range(steps)], 0)

    for t in range(steps):

        optimizer.zero_grad()

        print(f'Step {t}')

        print(f'x: {x}, y: {y}')

        # just add back in cost_func(x,y)[:,0]
        md_cost_func = lambda x,y : cost_func(x,y)[:,0] + metadynamics(torch.stack((x,y), 0).reshape(2, x.shape[0]).t(), cv_history, t, mag_scale, distance_scale, False)

        # md_cost_func = lambda x,y : cost_func(x,y)[:,0]

        # just add back in cost_func(x,y)
        non_shaped_cost_func = lambda x,y : cost_func(x,y) + metadynamics(torch.stack((x,y), 0), cv_history, t, mag_scale, distance_scale)

        # non_shaped_cost_func = lambda x,y : cost_func(x,y)

        # Forward pass: compute predicted y using operations; we compute
        z = non_shaped_cost_func(x, y)
        
        op.loss_update(md_cost_func)
        op.plot_surface()
        if t == 0:
            op.adjust_camera_angle()

        op.update_plot(x.detach().numpy(), y.detach().numpy(), z.detach().numpy())

        # Use autograd to compute the backward pass.
        z.backward()
        # import ipdb; ipdb.set_trace()
        # Update weights using gradient descent
        optimizer.step()
        
    op.plot_gif()

def test_gradients():
     
    tensor = torch.randn(2,2,dtype=torch.double,requires_grad=True)
    history = torch.stack([tensor.clone().detach().requires_grad_(False)+0.1 for i in range(5)], 0)

    metadynamics = MetaDynamics(tensor.shape, None, initial_history=history)

    input = (tensor, 20)

    test = gradcheck(metadynamics, input, eps=1e-6, atol=1e-4)

    print(test)

test_gradients()

# next we need to build some tests with this same structure and a valid loss function!!