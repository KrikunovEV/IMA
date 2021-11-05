import numpy as np

from rl.loss import get_data_by_key
POLICY = 'policy'


class IActionSampler:
    def __call__(self, **kwargs) -> int:
        raise NotImplementedError


class Uniform(IActionSampler):
    def __call__(self, **kwargs) -> int:
        policy = get_data_by_key(POLICY, kwargs)
        return np.random.randint(policy.shape[0])


class Choice(IActionSampler):
    def __call__(self, **kwargs) -> int:
        policy = get_data_by_key(POLICY, kwargs)
        return np.random.choice(policy.shape[0], 1, False, p=policy.detach().numpy())[0]


class Argmax(IActionSampler):
    def __call__(self, **kwargs) -> int:
        policy = get_data_by_key(POLICY, kwargs)
        return policy.argmax().item()


if __name__ == '__main__':
    from torch import Tensor
    for i in range(10):
        print(Choice()(policy=Tensor([0.1, 0.5, 0.4])))
