#%%
import os
import math
import torch
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, BinaryIO, IO
from collections.abc import Callable, Iterable
from cs336_basics.transformer.module import softmax
from cs336_basics.utils.logger import get_logger
logger = get_logger(__name__,"transformer_utils.txt")
#%%
def cross_entropy(inputs:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    """
    Parameters:
        inputs (Float[Tensor, "batch_size ... vocab_size"]): inputs[i][...][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size ..."]): Tensor of shape (batch_size,...) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    inputs = inputs - inputs.max(dim=-1)[0].unsqueeze(dim=-1).expand(*inputs.size())  # "batch_size ... vocab_size"
    sum = torch.log(torch.exp(inputs).sum(-1,keepdim=True))  # "batch_size ... 1"
    log_p = sum - inputs.gather(-1, targets.unsqueeze(-1))  # "batch_size ... 1"
    return log_p.mean()  # ""

#%%
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate{lr}")
        defaults = {'lr':lr}
        super().__init__(params,defaults)
    
    def step(self,closure:Optional[Callable[[],float]]=None)->Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get('t',0)
                p.data -= lr/math.sqrt(t+1) * grad
                state['t'] = t+1
        return loss
    
weights = torch.nn.Parameter(5 * torch.randn((10,10)))
opt = SGD([weights], lr=1e3)
for t in range(10):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()

# %%
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr:float, weight_decay:float, betas:Tuple[float,float]=(0.9,0.95), eps:float=1e-8):
        """
        Parameters:
            params: an iterable of :class:`torch.Tensor` s or :class:`dict` s. Specifies what Tensors should be optimized
            lr (float): learning rate
            weight_decya (float): used to improve regularization
            betas (Tuple[float, float]): a pair of hyperparameters that control the updates to the moment estimates
            eps (float): a small value used to improve numerical stability
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate {lr}!")
        defaluts = {'lr':lr, 'weight_decay':weight_decay, 'betas':betas, 'eps':eps}
        super().__init__(params, defaluts)
        self.logger = logger
        
    def step(self,closure:Optional[Callable[[],float]]=None)->Optional[float]:
        """
        Parameters:
            closure (Callable): A closure that reevaluates the model and returns the loss. Optional for most optimizers
        """
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            betas = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # 首次调用step(), 初始化
                    state['step'] = 1  # 当前步
                    state['exp_avg'] = torch.zeros_like(p.data)  # 一阶矩估计
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # 二阶矩估计
                step = state['step']
                grad = p.grad.data
                state['exp_avg'] = betas[0] * state['exp_avg'] + (1 - betas[0]) * grad
                state['exp_avg_sq'] = betas[1] * state['exp_avg_sq'] + (1 - betas[1]) * (grad**2)
                lr_t = lr * math.sqrt(1 - betas[1]**step) / (1 - betas[0]**step)
                p.data -= lr_t * state['exp_avg'] / (torch.sqrt(state['exp_avg_sq']) + eps)
                p.data -= lr * weight_decay * p.data
                state['step'] = step + 1
        return loss
    
#%%
def cosine_annealing_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Parameters:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.
    
    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        lr = it * max_learning_rate / warmup_iters
        return lr
    if it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
        return lr
    return min_learning_rate

#%%
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Parameters:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    total_grad = 0.0 
    grads = [p.grad for p in parameters if p.grad is not None]
    for g in grads:
        total_grad += torch.sum(g**2)
    total_norm = total_grad**(0.5)
    clip_coef = min(1, max_l2_norm/(total_norm+1e-6))
    for g in grads:
        g *= clip_coef

#%%
def get_batch(dataset:npt.NDArray, batch_size:int, context_length:int, device:str)->Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        out (Tuple[torch.Tensor, torch.Tensor]): Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    data_len = len(dataset)
    if data_len < context_length:
        raise ValueError(f"Dataset length {data_len} is less than context length {context_length}.")
    starts = np.random.randint(data_len-context_length,size=batch_size)
    inputs = np.stack([dataset[start:start+context_length] for start in starts],dtype=np.int64)
    targets = np.stack([dataset[start+1:start+context_length+1] for start in starts],dtype=np.int64)
    return (torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device))

#%%
def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: str | os.PathLike | BinaryIO | IO[bytes]):
    """
    Parameters:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    torch.save({'iteration':iteration,
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict()},
                out)

def load_checkpoint(src, model, optimizer)->int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Parameters:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    src = torch.load(src)
    if model is not None:
        model.load_state_dict(src['model'])
    if optimizer is not None:
        optimizer.load_state_dict(src['optimizer'])
    return src['iteration']

#%%
def compute_grad_norm(params:Iterable[torch.nn.Parameter])->float:
    """
    Parameters:
        params (Iterable[torch.nn.Parameter]): collection of trainable parameters
    
    Returns:
        out (float): the grad l2 norm of parameters
    """
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5

#%%
def evaluate_model(model:torch.nn.Module, dataset:npt.NDArray, batch_size:int, context_length:int, num_batches:int, device:str):
    """
    Parameters:
        model (torch.nn.Module): Model to be evaluated.
        dataset (np.array): 1D numpy array of integer token IDs in the validation dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        num_batches (int): the number of evaluation batches
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_batch(
                dataset=dataset,
                batch_size=batch_size,
                context_length=context_length,
                device=device
            )
            logits = model(inputs)
            loss = cross_entropy(inputs=logits, targets=targets)
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches