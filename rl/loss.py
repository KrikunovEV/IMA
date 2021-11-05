def get_data_by_key(key: str, data: dict):
    if key not in data:
        raise Exception(f'Data does not contain \'{key}\' key')
    return data[key]


class IRLLoss:

    def __init__(self, gamma):
        self.data = []
        self.gamma = gamma

    def collect(self, **kwargs):
        self.data.append(kwargs)

    def compute(self, **kwargs) -> dict:
        raise NotImplementedError


class Reinforce(IRLLoss):

    LOG_P = 'log_p'
    REWARD = 'reward'
    ENTROPY = 'entropy'

    def __init__(self, gamma, entropy=None):
        super().__init__(gamma)
        self.entropy = entropy

    def compute(self, **kwargs):
        g, loss = 0, 0

        for keys in reversed(self.data):
            log_p = get_data_by_key(self.LOG_P, keys)
            reward = get_data_by_key(self.REWARD, keys)

            g = reward + self.gamma * g
            loss = loss - g * log_p

            if self.entropy is not None:
                entropy = get_data_by_key(self.ENTROPY, keys)
                loss = loss - self.entropy * entropy

        self.data = []
        return dict(loss=loss)


class A2C(Reinforce):

    VALUE = 'value'

    def __init__(self, gamma, entropy=None, critic_loss_penalize=0.5):
        super().__init__(gamma, entropy)
        self.critic_loss_penalize = critic_loss_penalize

    def compute(self, **kwargs):
        g = get_data_by_key(self.VALUE, kwargs) if self.VALUE in kwargs else 0.
        policy_loss, value_loss = 0., 0.

        for keys in reversed(self.data):
            log_p = get_data_by_key(self.LOG_P, keys)
            reward = get_data_by_key(self.REWARD, keys)
            value = get_data_by_key(self.VALUE, keys)

            g = reward + self.gamma * g
            advantage = g - value
            policy_loss = policy_loss - advantage.detach() * log_p
            if self.entropy is not None:
                entropy = get_data_by_key(self.ENTROPY, keys)
                policy_loss = policy_loss - self.entropy * entropy

            value_loss = value_loss + advantage.pow(2)
        value_loss = value_loss * self.critic_loss_penalize

        loss = policy_loss + value_loss

        self.data = []
        return dict(loss=loss, policy_loss=policy_loss, value_loss=value_loss)


if __name__ == '__main__':
    l = Reinforce(gamma=0.9)
    l.collect(reward=1, log_p=-0.16)
    print(l.compute())
