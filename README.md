# Little complex MuJoCo usages
In this tutorial, we will be looking at slight complex usages of [MuJoCo](https://mujoco.org/) other than using it for [gym](https://www.gymlibrary.dev/). In fact, the main motivation for making this tutorial comes from the inconveniences caused by having different versions of gym packages while collaborating with colleagues. I hope this codebase can be helpful for those who are facing similar issues. 

### Manipulator (Franka Emika `Panda`)
- Parsing an MJCF file: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_00_parse.ipynb), [code](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/mujoco_parser.py)
- Forward dynamics: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_01_fd.ipynb)
- Forward kinematics: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_02_fk.ipynb)
- Inverse kinematics: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_03_ik.ipynb)
- Inverse dynamics (free-fall, gravity compensation, etc): [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_04_id.ipynb)
- PID control: [notebook](https://github.com/sjchoi86/little-complex-mujoco-usage/blob/main/code/demo_panda_05_pid.ipynb)

### Legged robot (`Snapbot`)
- WIP

Contact: sungjoon-choi at korea at ac dot kr
