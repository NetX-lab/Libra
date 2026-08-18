"""Support code for Ray launcher."""

import ray
from ray.util.placement_group import placement_group


@ray.remote(num_gpus=0)
class RayAsyncLauncher:
    """Ray async launcher implementation."""

    def __init__(self, config):
        self.config = config

    def launch(self):
        """Launch."""
        print("Initializing the Ray cluster...")


        pg = placement_group(
            bundles=[
                {"GPU": self.config.train_gpus},
                {"GPU": self.config.rollout_gpus},
            ],
            strategy="STRICT_PACK",
        )

        ray.get(pg.ready())


        train_actor = TrainActor.options(
            num_gpus=self.config.train_gpus,
            placement_group=pg,
            placement_group_bundle_index=0,
        ).remote(self.config)


        rollout_worker = RolloutWorker.options(
            num_gpus=self.config.rollout_gpus,
            placement_group=pg,
            placement_group_bundle_index=1,
        ).remote(self.config)

        print("Training actors and rollout workers have started")


        workflow = self._create_workflow()


        dataset = self._load_dataset()


        print("Starting asynchronous training...")
        train_actor.train.remote(workflow, dataset)

        return train_actor, rollout_worker

    def _create_workflow(self):
        """Create workflow."""
        from RL_Framework.workflow.rlvr import RLVRWorkflow
        from transformers import AutoTokenizer


        def dummy_reward(prompt, completion, **kwargs):
            return 1.0 if "correct" in completion.lower() else 0.0

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)

        workflow = RLVRWorkflow(
            reward_fn=dummy_reward,
            tokenizer=tokenizer,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            n_samples=self.config.n_samples,
        )

        return workflow

    def _load_dataset(self):
        """Load dataset."""
        from datasets import load_dataset

        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
        )

        return dataset


@ray.remote(num_gpus=1)
class TrainActor:
    """Train actor implementation."""

    def __init__(self, config):
        self.config = config
        self.trainer = None

    def train(self, workflow, dataset):
        """Train."""
        from RL_Framework.trainer.async_rl_trainer import AsyncRLTrainer

        self.trainer = AsyncRLTrainer(self.config)
        self.trainer.train(workflow, dataset)


@ray.remote(num_gpus=1)
class RolloutWorker:
    """Rollout worker implementation."""

    def __init__(self, config):
        self.config = config
        self.engine = None

    def start(self):
        """Start."""
        from RL_Framework.engine.rollout_engine import VLLMRolloutEngine

        self.engine = VLLMRolloutEngine(
            model_path=self.config.model_path,
            tp_size=self.config.tp_size,
            port=self.config.vllm_port,
            max_model_len=self.config.max_seq_length,
        )

        self.engine.start_server()
        print(f"Rollout engine started (port={self.config.vllm_port})")


def launch_ray_cluster(config):
    """Launch ray cluster."""

    ray.init(address=config.ray_address, ignore_reinit_error=True)

    print(f"Connected to Ray cluster: {ray.cluster_resources()}")


    launcher = RayAsyncLauncher.remote(config)


    train_actor, rollout_worker = ray.get(launcher.launch.remote())

    return launcher, train_actor, rollout_worker
