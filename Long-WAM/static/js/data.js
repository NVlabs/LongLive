// All numbers shown on the page, copied from the Long-WAM manuscript.
// Each block names its source table / figure. Tables keep the paper's rows and rounding.
window.LW = {
  // Fig. memory_scaling — success (%) vs. observed-video context; each window is a separately trained model
  scaling: {
    gr1: { ctx: ['0', '2.4', '4.8', '9.6', '19.2', '38.4'],
      robot: [63.3, 66.3, 71.2, 74.5, 78.7, 75.2], ar: [62.4, 65.7, 66.9, 69.8, 76.7, 74.2], bi: [60.0, 61.7, 62.4, 64.1, 61.6, 61.6] },
    libero: { ctx: ['0', '2.4', '4.8', '9.6'],
      robot: [94.5, 99.5, 99.0, 99.0], ar: [94.2, 99.0, 99.0, 99.0], bi: [93.4, 96.5, 97.1, 96.2] },
    padding384: 80.4
  },
  // Table context-latency-5090 — end-to-end latency per action chunk vs. history (RTX 5090)
  ctxLatency: { '0': 74.6, '2.4': 107.4, '4.8': 138.3, '9.6': 204.5, '19.2': 341.0 },
  // Table context-latency-5090 as shown on the page
  ctxLatencyTable: { cols: ['History (s)', 'Context P', 'Latency (ms)'], better: [null, null, null], digits: [null, 0, 1],
    rows: [['0.0', 0, 74.6], ['2.4', 48, 107.4], ['4.8', 96, 138.3], ['9.6', 192, 204.5], ['19.2', 384, 341.0]],
    note: 'End-to-end inference per action chunk with the Long-WAM infrastructure. P counts preceding control intervals (P/20 s of history); 0 keeps only the current observation.' },
  // Table gr1-comparison
  gr1Table: { cols: ['Method', 'GR-1'], better: [null, 'max'], unit: '%',
    rows: [['π<sub>0</sub>', 62.5], ['Diffusion Policy', 32.7], ['Fast-WAM', 47.5], ['GR00T-N1.5', 64.1], ['DreamZero', 62.4], ['Cosmos Policy', 67.1],
           ['Long-WAM (2.4 s)', 66.3, 'ours'], ['Long-WAM (19.2 s)', 78.7, 'ours']] },
  // Table libero
  libero: { cols: ['Method', 'Spatial', 'Object', 'Goal', 'Long', 'Avg.'], better: [null, 'max', 'max', 'max', 'max', 'max'],
    rows: [['OpenVLA', 84.7, 88.4, 79.2, 53.7, 76.5], ['OpenVLA-OFT', 97.6, 98.4, 97.9, 94.5, 97.1], ['GR00T-N1', 94.4, 97.6, 93.0, 90.6, 93.9],
           ['π<sub>0</sub>', 96.8, 98.8, 95.8, 85.2, 94.1], ['π<sub>0.5</sub>', 98.8, 98.2, 98.0, 92.4, 96.9], ['UniVLA', 95.4, 98.8, 93.6, 94.0, 95.5],
           ['X-VLA', 98.2, 98.6, 97.8, 97.6, 98.1], ['LingBot-VA', 98.5, 99.6, 97.2, 98.5, 98.5], ['Motus', 96.8, 99.8, 96.6, 97.6, 97.7],
           ['Fast-WAM', 98.2, 100.0, 97.0, 95.2, 97.6],
           ['Long-WAM (w/o V)', 98.0, 99.5, 97.0, 94.5, 97.3, 'ours'], ['Long-WAM (CoD)', 98.6, 99.8, 96.8, 97.8, 98.3, 'ours'],
           ['Long-WAM (IDM)', 99.5, 100.0, 98.0, 99.5, 99.5, 'ours']],
    note: 'w/o V: no future-video denoising · CoD: video–action co-denoising · IDM: inverse dynamics (predict video latents, then actions).' },
  // Table robotwin
  robotwin: { cols: ['Method', 'Clean', 'Randomized', 'Avg.'], better: [null, 'max', 'max', 'max'],
    rows: [['π<sub>0</sub>', 65.9, 58.4, 62.2], ['π<sub>0.5</sub>', 82.7, 76.8, 79.8], ['Motus', 88.7, 87.0, 87.8], ['Motus (Wan2.2)', 77.6, 77.0, 77.3],
           ['LingBot-VA', 92.9, 91.5, 92.2], ['LingBot-VA 2.0', 93.8, 93.4, 93.6], ['Fast-WAM', 91.9, 91.8, 91.8], ['AHA-WAM', 93.4, 92.2, 92.8],
           ['Abot-M0.5', 94.0, 94.2, 94.1], ['Qwen-RobotManip', 93.7, 94.0, 93.9],
           ['Long-WAM (w/o V)', 92.4, 91.6, 92.0, 'ours'], ['Long-WAM (CoD)', 94.0, 93.1, 93.6, 'ours'], ['Long-WAM (IDM)', 94.7, 94.2, 94.4, 'ours']] },
  // Table domino-ft (after dynamic-data fine-tuning); SR = success rate, MS = manipulation score
  domino: { cols: ['Method', 'SR (%)', 'MS'], better: [null, 'max', 'max'],
    rows: [['OpenVLA', 1.5, 6.1], ['π<sub>0</sub>', 8.2, 24.0], ['π<sub>0.5</sub>', 9.6, 26.2], ['InternVLA-M1', 5.4, 27.6], ['OpenVLA-OFT', 9.1, 24.1],
           ['StarVLA-OFT', 10.9, 30.5], ['PUMA', 17.2, 35.0], ['Fast-WAM', 19.9, 33.3], ['Long-WAM', 34.9, 45.1, 'ours']] },
  // Fig. experiment_2 — Unitree G1, 20 trials per policy and condition; human teleoperation shown for reference
  conveyor: { speeds: ['3.0', '4.5', '6.0', '7.5'],
    series: [{ name: 'π<sub>0.5</sub>', key: 'pi', counts: ['3/20', '0/20', '0/20', '0/20'], values: [15, 0, 0, 0] },
             { name: 'Fast-WAM', key: 'fast', counts: ['15/20', '6/20', '0/20', '0/20'], values: [75, 30, 0, 0] },
             { name: 'Long-WAM', key: 'ours', counts: ['20/20', '20/20', '19/20', '18/20'], values: [100, 100, 95, 90] },
             { name: 'Human (teleoperation)', key: 'human', counts: ['50/52', '50/53', '50/52', '50/58'], values: [96, 94, 96, 86] }] },
  stacking: [{ name: 'π<sub>0.5</sub>', key: 'pi', count: '0/20', value: 0 }, { name: 'Fast-WAM', key: 'fast', count: '0/20', value: 0 },
             { name: 'Long-WAM', key: 'ours', count: '19/20', value: 95 }, { name: 'Human (teleoperation)', key: 'human', count: '50/54', value: 92.6 }],
  // Sec. 5.4 long-horizon tasks on YAM (20 trials per task; tasks last over 40 s on average)
  yam: [['Sort bricks by color', 80], ['Place dumplings in a pan', 80], ['Stack bowls', 85]], yamMean: 81.7,
  // Table efficiency-model-latency (RTX 5090; SR = RoboTwin 2.0 mean of Clean/Randomized)
  latency: { cols: ['Method', 'Latency (ms)', 'RoboTwin SR (%)'], better: [null, 'min', 'max'],
    rows: [['LingBot-VA', 3618.4, 92.2], ['Motus', 1201.1, 87.8], ['Cosmos Policy', 470.6, null], ['Fast-WAM', 244.1, 91.8],
           ['Long-WAM (V4/A4)', 107.4, 94.4, 'ours'], ['Long-WAM (V2/A2)', 81.8, 92.5, 'ours']],
    note: 'Each model at its own deployment configuration (not matched denoising budgets). V4/A4 = four denoising steps per video/action expert.' },
  // Table efficiency-cumulative-ablation (ms, speedup vs. BF16 eager; Long-WAM IDM, full VAE included)
  accel: { stages: ['BF16 eager', '+ NVFP4 quantization', '+ CUDA Graph + compile', '+ Denoising-invariant reuse', '+ Shared quantization',
                    '+ Video–action online softmax', '+ Quant/GEMM tuning', '+ VAE convolution tuning'],
    groups: ['Baseline', 'Shared', 'Shared', 'Shared', 'Shared', 'Shared', 'Device-specific', 'Device-specific'],
    devices: [{ name: 'GeForce RTX 5090', img: 'gpu-rtx5090.jpg', ms: [356.0, 360.4, 172.4, 144.9, 130.3, 126.9, 119.1, 107.4], x: [1.0, 1.0, 2.1, 2.5, 2.7, 2.8, 3.0, 3.3] },
              { name: 'DGX Spark', img: 'gpu-spark-device.jpg', ms: [1342.8, 1147.4, 686.4, 464.5, 427.8, 419.9, 415.6, 328.2], x: [1.0, 1.2, 2.0, 2.9, 3.1, 3.2, 3.2, 4.1] },
              { name: 'Jetson AGX Thor', img: 'gpu-thor-dark.jpg', ms: [1215.2, 1219.8, 762.4, 524.5, 476.5, 468.2, 458.9, 378.7], x: [1.0, 1.0, 1.6, 2.3, 2.6, 2.6, 2.6, 3.2] }] },
  // Table async-continuity (RoboTwin 2.0; all results from the authors' evaluation)
  async: { cols: ['Model', 'Sync SR', 'Async SR', 'Async RMSE', 'Async jerk'], better: [null, 'max', 'max', 'min', 'min'],
    rows: [['Fast-WAM', 91.2, 76.4, 0.0731, 0.1316], ['LingBot-VA', 91.2, 46.7, 0.1432, 0.6592],
           ['Long-WAM (CoD)', 93.6, 93.2, 0.0260, 0.0479, 'ours'], ['Long-WAM (IDM)', 94.4, 94.2, 0.0246, 0.0436, 'ours']],
    digits: [null, 1, 1, 4, 4],
    note: 'Pure asynchronous execution at stride S = 12, without blending or prefix guidance. Jerk is the paper’s per-step finite-difference proxy.' },
  // Table robocasa365-planning (Overall averages 50 tasks: 18 Atomic-Seen, 16 Composite-Seen, 16 Composite-Unseen)
  rc365: { cols: ['Method', 'Atomic-Seen', 'Composite-Seen', 'Composite-Unseen', 'Overall'], better: [null, 'max', 'max', 'max', 'max'],
    rows: [['Diffusion Policy', 15.7, 0.2, 1.3, 6.1], ['Azero-Robotics-1', 30.3, 3.8, 1.6, 12.6], ['π<sub>0</sub>', 34.6, 6.1, 1.1, 14.8],
           ['GigaWorld-Policy 0.1', 44.4, 11.8, 2.9, 20.7], ['GR00T N1.6', 51.1, 9.4, 1.7, 21.9], ['GR00T N1.5', 50.7, 14.8, 2.7, 23.9],
           ['WorldDreamer', 66.3, 26.7, 9.0, 35.3], ['RLDX-1', 67.6, 27.9, 8.5, 36.0], ['PRTS', 66.3, 30.3, 18.8, 39.6],
           ['Qwen-RobotManip', 68.6, 20.1, 14.9, 35.9], ['ABot-M0.5', 75.6, 37.7, 3.3, 40.3],
           ['GPT-6 Astra', 31.5, 22.4, 20.8, 25.2, 'sep'], ['π<sub>0.5</sub>', 39.6, 7.1, 1.2, 16.9], ['π<sub>0.5</sub> + GPT-6 Astra', 46.5, 24.0, 19.0, 30.5],
           ['Long-WAM', 67.9, 15.8, 6.1, 31.4, 'ours'], ['Long-WAM + GPT-6 Astra', 85.6, 38.8, 35.0, 54.4, 'ours']],
    note: 'Same Human300-trained Long-WAM checkpoint with 2.4 s of visual context; GPT-6 Astra is added without further policy training.' }
};
