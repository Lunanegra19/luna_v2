module.exports = {
  apps: [
    {
      // [FIX-PM2-MEMORY-2026-05-26] max_memory_restart aumentado de 250M a 800M.
      // El live trader usa ~515MB con 3 semillas cargadas. El limite de 250M causaba
      // 421+ reinicios y KeyboardInterrupt en loop desde las 03:00 AM.
      name: "luna-v2-live-demo",
      script: "./scripts/run_live_trader.py",
      interpreter: "/root/miniconda3/envs/luna_env/bin/python",
      watch: false,
      // [FIX-PM2-MEMORY-2026-05-26] max_memory_restart aumentado de 250M → 800M → 1500M → 4000M.
      // El live trader usa ~515MB en reposo pero alcanza ~1850-1900MB durante inferencia
      // (FracDiff + AutoEncoder 492-dim + 29 semillas XGB/LSTM). El limite de 1500M lo mataba
      // justo en el ciclo de inferencia. Nuevo limite seguro: 4000M (VPS tiene 7.7GB total).
      max_memory_restart: "4000M",
      exp_backoff_restart_delay: 100,
      restart_delay: 3000,
      autorestart: true,
      kill_timeout: 10000,
      listen_timeout: 10000
    },
    {
      // [FIX-PM2-MEMORY-2026-05-26] max_memory_restart aumentado de 150M a 1200M.
      // El dashboard usa ~930MB con datos en cache. El limite de 150M causaba
      // reinicios frecuentes e interrupciones del scheduler de reportes horarios.
      name: "luna-dashboard",
      script: "./dashboard/server.py",
      interpreter: "/root/miniconda3/envs/luna_env/bin/python",
      watch: false,
      max_memory_restart: "1200M",
      autorestart: true,
      kill_timeout: 10000,
      listen_timeout: 10000
    }
  ]
};
