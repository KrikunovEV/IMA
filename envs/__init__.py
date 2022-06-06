def get_environment(env_module: str, players: int, debug: bool = False):
    import importlib
    return importlib.import_module(f'envs.{env_module}').Env(players, debug)
