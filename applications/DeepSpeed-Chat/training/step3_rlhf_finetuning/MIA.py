import argparse
import os
import math
import sys
import torch
import deepspeed
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModelForCausalLM, get_scheduler
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer
from utils.model.model_utils import create_hf_model, create_critic_model

import time

os.environ['LOCAL_RANK'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(
        description="(Step 3) RLHF training arguments")

    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60%% of data for phase 1, 20%% for phase 2 and 20%% for phase 3.'
    )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        default="/root/DeepSpeedMIA/applications/DeepSpeed-Chat/output/step3-models/1.3b/actor",
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        )
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        default="--critic_model_name_or_path /root/DeepSpeedMIA/applications/DeepSpeed-Chat/output/reward-models/350m",
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_mini_train_batch_size",
        type=int,
        default=2,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--generation_batch_numbers",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')
    parser.add_argument('--disable_actor_dropout',
                        action='store_true',
                        help='Disable the dropout of the actor model.')
    parser.add_argument('--disable_critic_dropout',
                        action='store_true',
                        help='Disable the dropout of the critical model.')
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if (args.actor_gradient_checkpointing
            and args.actor_lora_dim > 0) or (args.critic_gradient_checkpointing
                                             and args.critic_lora_dim > 0):
        assert (
            not args.only_optimize_lora
        ), "--{actor,critic}_gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    return args

def get_generator(path):
    if os.path.exists(path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      fast_tokenizer=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True)

    tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(path)
    model = OPTForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config).half()

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         device="cuda:0")
    return generator

#----------------------------------------------------------------------------
def _init_reward(args, critic_model_name_or_path, tokenizer):
    # DS Config
    zero_stage = args.critic_zero_stage
    if zero_stage != 3:
        # If critic is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
        zero_stage = 0

    ds_config = get_eval_ds_config(offload=args.offload,
                                   stage=zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_mini_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    #TODO(jeff): should not be needed, we should be able to use ds_config above
    #TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3
    ds_eval_config = get_eval_ds_config(offload=False, stage=0)

    # Model
    reward_model = create_critic_model(
        model_name_or_path=critic_model_name_or_path,
        #tokenizer=self.tokenizer,
        tokenizer=tokenizer,
        ds_config=ds_eval_config,
        num_padding_at_beginning=args.num_padding_at_beginning,
        rlhf_training=True)

    reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                             config=ds_config)

    return reward_engine

def _init_actor(args, actor_model_name_or_path, tokenizer):

    # DS Config
    ds_config = get_eval_ds_config(offload=args.offload,
                                   stage=0)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_mini_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    #TODO(jeff): should not be needed, we should be able to use ds_config above
    #TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3
    ds_eval_config = get_eval_ds_config(offload=False, stage=0)

    # Model
    actor_model = create_hf_model(
        model_class=AutoModelForCausalLM,
        model_name_or_path=actor_model_name_or_path,
        tokenizer=tokenizer,
        ds_config=ds_config,
        disable_dropout=args.disable_actor_dropout)

    # DeepSpeed Engine
    #TODO: move enable_hybrid_engine and pin_parameters to ds_config
    actor_engine, *_ = deepspeed.initialize(model=actor_model,
                                            config=ds_config)


    return actor_engine

def create_datasets(args, tokenizer, datapath, train_phase, data_split = None):
    if data_split != None:
        args.data_split = data_split
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank, datapath, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_prompt_seq_len)
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    """
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    """
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_train_batch_size)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_train_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


#settings
datasets = ['Dahoas/rm-static', 'Dahoas/full-hh-rlhf', 'Dahoas/synthetic-instruct-gptj-pairwise' ,'yitingxie/rlhf-reward-datasets']

actor_path = "/root/DeepSpeedMIA/applications/DeepSpeed-Chat/output/step3-models/1.3b/actor"

reward_path = "/root/DeepSpeedMIA/applications/DeepSpeed-Chat/output/reward-models/350m"

def reward_test(args, dataset, actor_model, reward_model, tokenizer):
    device = torch.device("cuda")
    print("--------reward test start---------")
    batchsize = 0
    total = 0
    total_reward = 0 #the sum of reward in the whole dataset
    length = len(dataset)
    T1 = time.time()
    for step, batch_prompt in enumerate(dataset):
        batch_prompt = to_device(batch_prompt, device)
        prompts = batch_prompt['prompt']
        mask = batch_prompt['prompt_att_mask']
        batchsize = prompts.shape[0]
        total += batchsize
        max_min_length = args.max_answer_seq_len + prompts.shape[1]
        seq = actor_model.module.generate(
            prompts,
            attention_mask=mask,
            max_length=max_min_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]#me:all prompts are in a same length 
        ans = seq[:, prompt_length:]#me:the seq return full sentence including the prompt and answer
        valid_ans_len = (ans != tokenizer.pad_token_id).sum(dim=-1)#me: counting the token which is not a padding
        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim
        #mask is updated with the newly generated seq

        attention_mask = seq.not_equal(tokenizer.pad_token_id).long()
        reward_score = reward_model.forward_value(
            seq, attention_mask,
            prompt_length=prompt_length
        )['chosen_end_scores'].detach()
        total_reward += reward_score.sum().item()
        if step % 10 == 0:
            T2 = time.time()
            print(" step: ", step, " / ", length)
            print(" current runtime is: ", (T2 - T1), "s")
            print(" current total reward is: ", total_reward)


    print(" the total reward is: ", total_reward)
    print(" length is: ", length)
    print(" last batchsize is: ", batchsize)
    print(" total batch number is: ", total)
    avg = total_reward / float(total)
    print(" average is: ", avg)
    return total_reward, avg


def main():
    args = parse_args()
    args.end_of_conversation_token = "<|endoftext|>"
    #batchsize settings
    args.per_device_train_batch_size = 32
    args.per_device_mini_train_batch_size = 32

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:1234', world_size=1, rank=0)
    args.global_rank = torch.distributed.get_rank()

    #args.global_rank = torch.distributed.get_rank()

    # create common tokenizer based on actor model
    tokenizer = load_hf_tokenizer(args.actor_model_name_or_path,
                                  fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'

    reward_model = _init_reward(args, critic_model_name_or_path=reward_path, tokenizer=tokenizer)
    """
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
    """
    actor_model = _init_actor(args, actor_model_name_or_path=actor_path, tokenizer=tokenizer)

    reward_model.eval()
    actor_model.eval()

    #test Dahoas/rm-static (different phases)
    """
    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3) #在dataset中就已经传入了tokenizer 为什么model还需要
    """
    #dataset0_phase1, _, num_total_iters = create_datasets(args=args, tokenizer=tokenizer, train_phase=1, datapath=["Dahoas/rm-static"])
    #dataset0_phase2, _, num_total_iters = create_datasets(args=args, tokenizer=tokenizer, train_phase=2, datapath=['Dahoas/rm-static'])
    dataset0_phase3, _, num_total_iters = create_datasets(args=args, tokenizer=tokenizer, train_phase=3, datapath=['Dahoas/rm-static'])
    
    #test other datasets
    #dataset1, _, num_total_iters = create_datasets(args=args, tokenizer=tokenizer, train_phase=3, data_split='4,3,3', datapath=['Dahoas/full-hh-rlhf'])
    #dataset2, _, num_total_iters = create_datasets(args=args, tokenizer=tokenizer, train_phase=3, data_split='1,1,8', datapath=['Dahoas/synthetic-instruct-gptj-pairwise'])
    #dataset3, _, num_total_iters = create_datasets(args=args, tokenizer=tokenizer, train_phase=3, data_split='1,1,8', datapath=['yitingxie/rlhf-reward-datasets'])

    #avg_0_1 = reward_test(args, dataset=dataset0_phase1, actor_model=actor_model, reward_model=reward_model, tokenizer=tokenizer)
    #avg_0_2 = reward_test(args, dataset=dataset0_phase2, actor_model=actor_model, reward_model=reward_model, tokenizer=tokenizer)
    avg_0_3 = reward_test(args, dataset=dataset0_phase3, actor_model=actor_model, reward_model=reward_model, tokenizer=tokenizer)

    #avg_1 = reward_test(args, dataset=dataset1, actor_model=actor_model, reward_model=reward_model, tokenizer=tokenizer)
    #avg_2 = reward_test(args, dataset=dataset2, actor_model=actor_model, reward_model=reward_model, tokenizer=tokenizer)
    #avg_3 = reward_test(args, dataset=dataset3, actor_model=actor_model, reward_model=reward_model, tokenizer=tokenizer)

if __name__ == "__main__":
    main()