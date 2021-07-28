The iGibson Challenge 2021 consists of **2 tasks** available in **8 environments**. The code in this repository allows
each of these combinations to be run with **5 popular RL algorithms**.

For evaluation:
* A2C, DDPG, PPO will be trained in 3 environments (Rs_int, Wainscott_1_int, Beechwood_1_int) on the `social_nav` task
* This will occur twice: once on vanilla iGibson, once on Foresight iGibson
* Metrics (CPU Usage (%), Memory Usage (MB), GPU Memory Usage (MB)) will be compared between both training sessions
* Training occurred on device w/ Intel i9-9900k CPU, 16GB RAM, Nvidia 2080 Ti GPU