def get_agent(agent_module: str, id: int, o_space: int, a_space: int, cfg):
    import importlib
    return importlib.import_module(f'agents.{agent_module}').Agent(id, o_space, a_space, cfg)
