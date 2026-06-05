from luna.config.config_loader import load_config
cfg = load_config()
print(f"cost_pct: {getattr(cfg.sop, 'cost_pct', 'NOT FOUND')}")
