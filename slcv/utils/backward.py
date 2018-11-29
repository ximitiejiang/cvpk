#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:19:52 2018

@author: ubuntu
"""
import torch

def _make_grads(outputs, grads):
    '''这个子程序是tensor backward()函数的子函数，主要是实现创建了grad变量
    输入: outputs为传入的loss值， tensor类型。
         grads为梯度，但为空
    '''
    new_grads = []
    for out, grad in zip(outputs, grads):
        if isinstance(grad, torch.Tensor):
            new_grads.append(grad)
        elif grad is None:
            if out.requires_grad:
                if out.numel() != 1:
                    raise RuntimeError("grad can be implicitly created only for scalar outputs")
                new_grads.append(torch.ones_like(out))
            else:
                new_grads.append(None)
        else:
            raise TypeError("gradients can be either Tensors or None, but got " +
                            type(grad).__name__)
    return tuple(new_grads)


def backward(self, gradient=None, retain_graph=None, create_graph=False):
    '''tensor的backward()函数，这个函数是可以直接用来进行反向求梯度值的
    在内部，他是调用了torch.autograd.backward()函数。
    
    '''
    r"""Computes the gradient of current tensor w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If the tensor is
    non-scalar (i.e. its data has more than one element) and requires
    gradient, the function additionally requires specifying ``gradient``.
    It should be a tensor of matching type and location, that contains
    the gradient of the differentiated function w.r.t. ``self``.

    This function accumulates gradients in the leaves - you might need to
    zero them before calling it.

    Arguments:
        gradient (Tensor or None): Gradient w.r.t. the
            tensor. If it is a tensor, it will be automatically converted
            to a Tensor that does not require grad unless ``create_graph`` is True.
            None values can be specified for scalar Tensors or ones that
            don't require grad. If a None value would be acceptable then
            this argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute
            the grads will be freed. Note that in nearly all cases setting
            this option to True is not needed and often can be worked around
            in a much more efficient way. Defaults to the value of
            ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative
            products. Defaults to ``False``.
    """
    torch.autograd.backward(self, gradient, retain_graph, create_graph)


def ultimate_backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None):
    '''这是另外一个backward来自torch.autograd.backward()
    名字被我从backward()改为ultimate_backward()了，是真正计算反向求导计算tensor梯度的程序，
    内部最终引入了Variable._execution_engine.run_backward()进行反向梯度计算
    '''
    
    r"""Computes the sum of gradients of given tensors w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If any of ``tensors``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains gradient of
    the differentiated function w.r.t. corresponding tensors (``None`` is an
    acceptable value for all tensors that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    them before calling it.

    Arguments:
        tensors (sequence of Tensor): Tensors of which the derivative will be
            computed.
        grad_tensors (sequence of (Tensor or None)): Gradients w.r.t.
            each element of corresponding tensors. None values can be specified for
            scalar Tensors or ones that don't require grad. If a None value would
            be acceptable for all grad_tensors, then this argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
    """
    if grad_variables is not None:
        warnings.warn("'grad_variables' is deprecated. Use 'grad_tensors' instead.")
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError("'grad_tensors' and 'grad_variables' (deprecated) "
                               "arguments both passed to backward(). Please only "
                               "use 'grad_tensors'.")

    tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)

    if grad_tensors is None:
        grad_tensors = [None] * len(tensors)
    elif isinstance(grad_tensors, torch.Tensor):
        grad_tensors = [grad_tensors]
    else:
        grad_tensors = list(grad_tensors)

    grad_tensors = _make_grads(tensors, grad_tensors)
    if retain_graph is None:
        retain_graph = create_graph

    Variable._execution_engine.run_backward(
        tensors, grad_tensors, retain_graph, create_graph,
        allow_unreachable=True)  # allow_unreachable flag    
    
    
def test_backward():
    x = torch.tensor([1,2],dtype=torch.float32,requires_grad=True)
    y = x * 2
    z = torch.mean(y)
    
    z.backward()
    
    print('z.grad:', z.grad)
    
    
    
    
if __name__ == '__main__':
    test_id = 2
    
    if test_id ==1:
        #尝试使用backward函数
        test_backward()
        
    elif test_id == 2:
        # 尝试_make_grids()
        loss = torch.tensor([1.213])
        grads = torch.tensor([])
        new_grads = _make_grads(loss, grads)
    
    else:
        pass
    