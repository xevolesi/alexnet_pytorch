import torch


@torch.no_grad()
def calculate_accuracy(result: torch.FloatTensor, answer: torch.Tensor, topk: int = 1) -> torch.Tensor:
    """Taken from here: https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b?permalink_comment_id=4681951#gistcomment-4681951"""
    #save the batch size before tensor mangling
    bz = answer.size(0)
    #ignore result values. its indices: (sz,cnt) -> (sz,topk)
    _, indices = result.topk(topk)
    #transpose the k best indice
    result = indices.t()  #(sz,topk) -> (topk, sz)

    #repeat same labels topk times to match result's shape
    answer = answer.view(1, -1)       #(sz) -> (1,sz)
    answer = answer.expand_as(result) #(1,sz) -> (topk,sz)

    correct = (result == answer)    #(topk,sz) of bool vals
    correct = correct.flatten()     #(topk*sz) of bool vals
    correct = correct.float()       #(topk*sz) of 1s or 0s
    correct = correct.sum()         #counts 1s (correct guesses)

    return correct.mul_(1/bz)


@torch.no_grad()
def calculate_batch_top_k_error_rate(inputs: torch.FloatTensor, targets: torch.Tensor, k: int = 5) -> torch.Tensor:
    return 1 - calculate_accuracy(inputs, targets, topk=k)
