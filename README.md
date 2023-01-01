# Little complex MuJoCo usages (+some others)

 In this tutorial, we will be looking at slightly complex usages of [MuJoCo](https://mujoco.org/) other than using it for [gym](https://www.gymlibrary.dev/). In fact, the main motivation for making this tutorial comes from the inconveniences caused by having different versions of gym packages (with [mujocopy](https://github.com/openai/mujoco-py)) while collaborating with colleagues. I hope this codebase can be helpful for those who are facing similar issues. This repo also contains some useful codes for controlling robots such as Gaussain random paths for sampling smooth joint trajectories or variational autoencoder with its variations (e.g., VQVAE or GQVAE). 
 
 MuJoCo provides quite useful functionalities in robotics such as computing Jacobian matrics, solving inverse dynamics for gravity compensation, or contact force estimations. Hope you enjoy. :) 

### Manipulator (Franka Emika `Panda`)
- Parsing an MJCF file: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_00_parse.ipynb), [code](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/mujoco_parser.py)
- Forward dynamics: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_01_fd.ipynb)
- Forward kinematics: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_02_fk.ipynb)
- Inverse kinematics: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_03_ik.ipynb)
- Inverse dynamics (free-fall, gravity compensation, etc): [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_04_id.ipynb)
- PID control: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_05_pid.ipynb)
- Object spawning and tracking: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_06_objects.ipynb)
- Contact information: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_07_contact.ipynb)

### Legged robot (`Snapbot`)
- WIP

### Others
- Gaussian random path: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_unit_01_grp.ipynb)

Contact: sungjoon-choi at korea at ac dot kr
