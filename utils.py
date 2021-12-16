import time
import torch
import numpy as np
import scipy.special
import math


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} in {end - start:0.8f} seconds")
        return result

    return wrapper


def append_dict2dict(input_dict, output_dict):
    for key in input_dict:
        if key in output_dict:
            output_dict[key].append(input_dict[key].detach().numpy())
        else:
            output_dict[key] = [input_dict[key].detach().numpy()]


def to_one_hot(index, size, is_numpy: bool = True):
    one_hot = np.zeros(size, dtype=np.int32)
    one_hot[index] = 1
    return one_hot if is_numpy else torch.tensor(one_hot)


def indices_except(index, num_elems):
    return np.arange(len(num_elems)) != index


def get_cooperation_relation(shape, to_draw: bool = False):
    indices = np.arange(shape)
    yield from [(current + 1 * to_draw, neighbor + 1 * to_draw)
                for current in indices for neighbor in indices[current + 1:]]


def calc_possible_cooperation(num_agents, only4pairs: bool = False, with_repeat: bool = False):
    return int(scipy.special.binom(num_agents, 2)) * (num_agents - 2) ** with_repeat if only4pairs \
        else np.sum(
        [scipy.special.binom(num_agents, n) * (num_agents - n) ** with_repeat for n in range(2, num_agents)], dtype=int)


# def calc_possible_cooperation1(num_agents, for_pairs: bool = False):
#     if for_pairs:
#         return (num_agents - 2) * scipy.special.binom(num_agents, 2)
#     else:
#         return np.sum([(num_agents - n) * scipy.special.binom(num_agents, n) for n in range(2, num_agents)], dtype=int)

def greater_divisor(number):
    gcd = 1
    i = number - 1
    while True:
        gcd = math.gcd(number, i)
        i -= 1
        if gcd != 1 or i < 2:
            break
    return gcd


@timer
def main(actions):
    for n in range(30):
        number = calc_possible_cooperation(n, only4pairs=True)

    arr = np.arange(actions.shape[-1])

    new_arr = np.array([1 if np.array_equal(actions[:, element], actions[:, another])
                        else 0
                        for element in arr
                        for another in arr[element + 1:]])
    print(new_arr)


if __name__ == '__main__':
    main(np.array([[[0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0]],
                   [[0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0]]]))
