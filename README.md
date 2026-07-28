
cd ~/IsaacLab
./isaaclab.sh -p train_dog_rsl.py --headless --num_envs 2048 --max_iterations 3000
[INFO] Curriculum Manager:  <CurriculumManager> contains 0 active terms.
+-------------------------+
| Active Curriculum Terms |
+-------------+-----------+
|    Index    | Name      |
+-------------+-----------+
+-------------+-----------+

[INFO]: Completed setting up the environment...
Environment created with 4096 envs
--------------------------------------------------------------------------------
Resolved observation sets: 
	 actor :  ['policy']
	 critic :  ['policy']
--------------------------------------------------------------------------------
Actor Model: MLPModel(
  (obs_normalizer): Identity()
  (mlp): MLP(
    (0): Linear(in_features=48, out_features=128, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=128, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=12, bias=True)
  )
)
Critic Model: MLPModel(
  (obs_normalizer): Identity()
  (mlp): MLP(
    (0): Linear(in_features=48, out_features=128, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=128, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=1, bias=True)
  )
)
Starting training...
Could not find git repository in /home/jimi/anaconda3/envs/isaac_lab_final/lib/python3.12/site-packages/rsl_rl/__init__.py. Skipping.
Traceback (most recent call last):
  File "/home/jimi/IsaacLab/train_dog_rsl.py", line 155, in <module>
    main()
  File "/home/jimi/IsaacLab/train_dog_rsl.py", line 149, in main
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
  File "/home/jimi/anaconda3/envs/isaac_lab_final/lib/python3.12/site-packages/rsl_rl/runners/on_policy_runner.py", line 108, in learn
    loss_dict = self.alg.update()
                ^^^^^^^^^^^^^^^^^
  File "/home/jimi/anaconda3/envs/isaac_lab_final/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py", line 207, in update
    for batch in generator:
                 ^^^^^^^^^
  File "/home/jimi/anaconda3/envs/isaac_lab_final/lib/python3.12/site-packages/rsl_rl/storage/rollout_storage.py", line 257, in mini_batch_generator
    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jimi/anaconda3/envs/isaac_lab_final/lib/python3.12/site-packages/rsl_rl/storage/rollout_storage.py", line 257, in <genexpr>
    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                                  ~^^^^^^^^^^^
IndexError: index 22988 is out of bounds for dimension 0 with size 16
(isaac_lab_final) jimi@jimi-JIGUANG-Series:~/IsaacLab$ 

