# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import datetime
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

from hora.tasks.base.base_task import AsyncEnvGroup
from hora.utils.misc import set_np_formatting, set_seed, git_hash, git_diff_config
from hora.utils.reformat import omegaconf_to_dict

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config.
# used primarily for num_ensv
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)


@hydra.main(config_name="config", config_path="configs")
def main(config: DictConfig):
    if config.checkpoint:
        config.checkpoint = to_absolute_path(config.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()

    cprint("Start Building the Environment", "green", attrs=["bold"])
    device_ids = config.device_ids
    if device_ids is None:
        # TODO
        device_ids = list(range(1, 8))
    rl_device_id = int(config.rl_device.split(":")[-1])
    print(
        f"Using CUDA device(s) {', '.join(map(str, device_ids))} for data collection and {rl_device_id} for RL"
    )

    def mk_env_factory(i):
        def mk_env():
            from hora.tasks.task_map import isaacgym_task_map

            set_seed(config.seed + 1 + i)

            return isaacgym_task_map[config.task_name](
                config=omegaconf_to_dict(config.task),
                device_id=i,
                headless=config.headless,
            )

        return mk_env

    envs = [mk_env_factory(i) for i in device_ids]

    if len(envs) > 1:
        env = AsyncEnvGroup(envs)
    else:
        env = envs[0]()

    from hora.algo.ppo.ppo import PPO
    from hora.algo.padapt.padapt import ProprioAdapt

    # sets seed. if seed is -1 will pick a random one
    config.seed = set_seed(config.seed)
    try:
        output_dif = os.path.join("outputs", config.train.ppo.output_name)
        os.makedirs(output_dif, exist_ok=True)
        agent = eval(config.train.algo)(env, output_dif, full_config=config)
        if config.test:
            agent.restore_test(config.train.load_path)
            agent.test()
        else:
            date = str(datetime.datetime.now().strftime("%m%d%H"))
            print(git_diff_config("./"))
            os.system(f"git diff HEAD > {output_dif}/gitdiff.patch")
            with open(
                os.path.join(output_dif, f"config_{date}_{git_hash()}.yaml"), "w"
            ) as f:
                f.write(OmegaConf.to_yaml(config))

            # check whether execute train by mistake:
            best_ckpt_path = os.path.join(
                "outputs",
                config.train.ppo.output_name,
                "stage1_nn" if config.train.algo == "PPO" else "stage2_nn",
                "best.pth",
            )
            if os.path.exists(best_ckpt_path):
                user_input = input(
                    f"are you intentionally going to overwrite files in {config.train.ppo.output_name}, type yes to continue \n"
                )
                if user_input != "yes":
                    exit()

            agent.restore_train(config.train.load_path)
            agent.train()
    finally:
        env.close()


if __name__ == "__main__":
    main()
