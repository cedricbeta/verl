"""
Main entry point for OnlineDPO training with VERL.
"""
from verl.trainer.dpo.ray_trainer import RayDPOTrainer, Role, ResourcePoolManager

import os
import ray
import hydra


def get_custom_judge_fn(config):
    """
    Load a custom judge function from a user-specified path.
    
    Args:
        config: The configuration object containing judge function information.
        
    Returns:
        A callable judge function or None.
    """
    import importlib.util, os

    judge_fn_config = config.get("custom_judge_function") or {}
    file_path = judge_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Judge function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = judge_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Judge function '{function_name}' not found in '{file_path}'.")

    print(f"Using customized judge function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


def get_custom_reward_fn(config):
    """
    Load a custom reward function from a user-specified path.
    
    Args:
        config: The configuration object containing reward function information.
        
    Returns:
        A callable reward function or None.
    """
    import importlib.util, os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"Using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path='config', config_name='dpo_trainer', version_base=None)
def main(config):
    run_dpo(config)


def run_dpo(config) -> None:
    # Set environment variables for GPU isolation
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not ray.is_initialized():
        # Initialize local ray cluster
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # ensure main_task isn't scheduled on head
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local
        # Print initial config
        from pprint import pprint
        from omegaconf import OmegaConf
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True evaluates symbol values
        OmegaConf.resolve(config)

        # Download the checkpoint from hdfs if needed
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # Initialize tokenizer
        from verl.utils import hf_tokenizer, hf_processor
        trust_remote_code = config.data.get('trust_remote_code', False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # for multimodal LLM, could be None

        # Define worker classes based on strategy
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker, ActorRolloutRefWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker, ActorRolloutRefWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError(f"Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")

        # Set up role to worker mapping
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }

        # Create resource pool specification
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # Set up reward model if enabled
        if config.reward_model.enable:
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Set up judge if enabled
        # if config.judge.enable:
        #     from verl.workers.judge_worker import JudgeWorker
        #     role_worker_mapping[Role.Judge] = ray.remote(JudgeWorker)
        #     mapping[Role.Judge] = global_pool_id

        # Set up reward manager for validation
        reward_manager_name = config.reward_model.get("reward_manager", "naive")
        if reward_manager_name == 'naive':
            from verl.workers.reward_manager import NaiveRewardManager
            reward_manager_cls = NaiveRewardManager
        elif reward_manager_name == 'prime':
            from verl.workers.reward_manager import PrimeRewardManager
            reward_manager_cls = PrimeRewardManager
        else:
            raise NotImplementedError(f"Unsupported reward manager: {reward_manager_name}")

        # Get custom judge and reward functions
        judge_fn = get_custom_judge_fn(config)
        reward_fn = get_custom_reward_fn(config)

        # Create managers for reward and validation
        reward_manager = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=reward_fn)
        val_reward_manager = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=reward_fn)

        # Create resource pool manager
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Create and run DPO trainer
        trainer = RayDPOTrainer(config=config,
                               tokenizer=tokenizer,
                               processor=processor,
                               role_worker_mapping=role_worker_mapping,
                               resource_pool_manager=resource_pool_manager,
                               ray_worker_group_cls=ray_worker_group_cls,
                               judge_fn=judge_fn,
                               val_reward_fn=val_reward_manager)
                               
        trainer.init_workers()
        trainer.fit()


if __name__ == '__main__':
    main()