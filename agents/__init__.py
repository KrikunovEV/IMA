def get_agent_module(algorithm: str):
    import importlib
    return importlib.import_module(f'agents.{algorithm}.agent').Agent
