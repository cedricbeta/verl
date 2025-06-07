import os
import sys
import ray
import hydra # Import hydra
from omegaconf import OmegaConf, open_dict
from pprint import pprint
from copy import deepcopy
import torch

# VERL project structure imports (adjust paths if necessary)
from verl.trainer.ppo.ray_trainer_mc import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local

# Worker classes (dynamic import based on config)
# Reward Manager
try:
    from verl.workers.reward_manager import VQAMultiStageRewardManager
    print("Imported VQAMultiStageRewardManager.")
except ImportError:
    print("ERROR: VQAMultiStageRewardManager not found. Please ensure it's in verl/workers/ or accessible.")
    VQAMultiStageRewardManager = None
    exit(1)


# Use hydra.main decorator
@hydra.main(config_path='config', config_name='vqa', version_base=None)
def main_with_hydra_config(config: OmegaConf): # Function now takes OmegaConf object
    """
    Main function to set up and run the CG-Bench evaluation using Hydra config.
    """
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': config.trainer.get("nccl_debug", "WARN"),
                    'VLLM_LOGGING_LEVEL': config.trainer.get("vllm_logging_level", "WARN"),
                    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '')
                },
                "working_dir": os.getcwd(),
            }
        )
    print(f"Ray initialized: {ray.is_initialized()}")
    print(f"Available resources: {ray.available_resources()}")

    print("--- Effective Configuration for Evaluation ---")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    torch.autograd.set_detect_anomaly(config.trainer.get("detect_anomaly", False))

    # Model path for tokenizer/processor from config
    base_model_path = config.actor_rollout_ref.model.path
    print(f"Copying base model for tokenizer from: {base_model_path}")
    local_model_path_tokenizer = copy_to_local(base_model_path)
    print(f"Local model path for tokenizer/processor: {local_model_path_tokenizer}")

    tokenizer = hf_tokenizer(local_model_path_tokenizer)
    processor = hf_processor(local_model_path_tokenizer, use_fast=True)
    print(f"Tokenizer: {tokenizer}")
    print(f"Processor: {processor}")

    # Define worker classes and resource management based on config
    strategy = config.actor_rollout_ref.actor.strategy
    ActorRolloutRefWorker, CriticWorker, RewardModelWorker, RayWorkerGroupCls = None, None, None, None

    if strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker as FSDPActorWorker, \
                                               CriticWorker as FSDPCriticWorker, \
                                               RewardModelWorker as FSDPRMWorker
        from verl.single_controller.ray import RayWorkerGroup as FSDPRayWorkerGroup
        ActorRolloutRefWorker, CriticWorker, RewardModelWorker = FSDPActorWorker, FSDPCriticWorker, FSDPRMWorker
        RayWorkerGroupCls = FSDPRayWorkerGroup
        print("Using FSDP workers for evaluation.")
    elif strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker as MegatronActorWorker, \
                                                  CriticWorker as MegatronCriticWorker, \
                                                  RewardModelWorker as MegatronRMWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ActorRolloutRefWorker, CriticWorker, RewardModelWorker = MegatronActorWorker, MegatronCriticWorker, MegatronRMWorker
        RayWorkerGroupCls = NVMegatronRayWorkerGroup
        print("Using Megatron workers for evaluation.")
    else:
        raise NotImplementedError(f"Unsupported strategy in config: {strategy}")

    role_worker_mapping = {Role.ActorRollout: ray.remote(ActorRolloutRefWorker)}
    
    global_pool_id = config.trainer.get("eval_resource_pool_id", 'global_pool_cg_eval')
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {Role.ActorRollout: global_pool_id}
    
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker) # Assuming ref uses same worker
        mapping[Role.RefPolicy] = global_pool_id

    if config.reward_model.enable:
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
        
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # Instantiate VQAMultiStageRewardManager
    if VQAMultiStageRewardManager is None:
        raise RuntimeError("VQAMultiStageRewardManager was not imported correctly.")
        
    vqa_eval_reward_fn = VQAMultiStageRewardManager(
            tokenizer=tokenizer,
            grounding_weight=config.trainer.grounding_weight,
            qa_accuracy_weight=config.trainer.qa_accuracy_weight,
            qa_format_weight=config.trainer.qa_format_weight,
            num_examine=1,
        )

    # Instantiate RayPPOTrainer
    trainer_instance = RayPPOTrainer(
        config=config, # Pass the full OmegaConf object
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroupCls,
        reward_fn=vqa_eval_reward_fn,
        val_reward_fn=vqa_eval_reward_fn # For eval, val_reward_fn is the one used by evaluate_cg_bench
    )
    
    print("\n--- Initializing Workers for CG-Bench Evaluation ---")
    trainer_instance.init_workers() 
    
    print("\n--- Loading Checkpoint for CG-Bench Evaluation ---")
    if not (config.trainer.resume_mode == "resume_path" and config.trainer.resume_from_path):
        raise ValueError("For evaluation, config.trainer.resume_mode must be 'resume_path' and "
                         "config.trainer.resume_from_path must point to the checkpoint folder.")
    
    loaded_step = trainer_instance._load_checkpoint()
    print(f"--- Checkpoint from step {loaded_step} loaded. Trainer's global_steps is now {trainer_instance.global_steps} ---")

    cg_bench_json_path = config.trainer.cg_bench_json_path
    cg_bench_output_dir = config.trainer.cg_bench_output_dir.format(global_steps=trainer_instance.global_steps) # Format with loaded step
    os.makedirs(cg_bench_output_dir, exist_ok=True)

    print(f"\n--- Starting CG-Bench Evaluation on: {cg_bench_json_path} ---")
    if not hasattr(trainer_instance, 'evaluate_cg_bench'):
        raise AttributeError("RayPPOTrainer instance does not have 'evaluate_cg_bench' method.")
    
    cg_bench_metrics = trainer_instance.evaluate_cg_bench(
        cg_bench_json_path=cg_bench_json_path,
        output_dir=cg_bench_output_dir
    )
    
    print(f"\n--- CG-Bench Evaluation Metrics (Step {trainer_instance.global_steps}) ---")
    pprint(cg_bench_metrics)

    # Optional: Log to external tracker if configured in evaluate_cg_bench.yaml
    if config.trainer.get("eval_logger_enable", False): # Example: add a specific logger enable flag
        from verl.utils.tracking import Tracking
        logger_eval = Tracking(
            project_name=config.trainer.get("eval_logger_project", config.trainer.project_name),
            experiment_name=f"{config.trainer.experiment_name}_cg_bench_eval_step_{trainer_instance.global_steps}",
            default_backend=config.trainer.get("eval_logger_backend", config.trainer.logger),
            config=OmegaConf.to_container(config, resolve=True)
        )
        logger_eval.log(data=cg_bench_metrics, step=trainer_instance.global_steps)
        print(f"Logged CG-Bench metrics to {logger_eval.default_backend}.")

    print(f"\n--- Detailed results saved in: {cg_bench_output_dir} ---")
    print("--- CG-Bench Evaluation with Hydra Config Complete ---")
    ray.shutdown()


if __name__ == '__main__':
    # This will now be invoked by Hydra
    main_with_hydra_config()