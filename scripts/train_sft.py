import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import random

# Constants
eps = 1e-3  # for forward_process
MASK_ID = 126336  # reserved mask token ID for LLaDA


def forward_process(input_ids: torch.Tensor, eps: float = eps):
    """
    Add noise to the entire sequence for SFT, to be applied only on the answer part later.
    Returns:
        noisy_batch (Tensor): with masked tokens
        mask_indices (Tensor): boolean mask of masked positions
        p_mask (Tensor): probability mask used for weighting
    """
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask.unsqueeze(1).expand(-1, l)
    mask_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    noisy_batch = torch.where(mask_indices, MASK_ID, input_ids)
    return noisy_batch, mask_indices, p_mask


def compute_sft_loss(model, input_ids: torch.Tensor, prompt_lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute supervised fine-tuning loss as described in GUIDELINES.md (Appendix B.1).
    Only the answer part is noised and masked.
    """
    # add noise
    noisy_batch, _, p_mask = forward_process(input_ids)

    # restore prompt tokens (no noise)
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    prompt_mask = positions < prompt_lengths.unsqueeze(1)
    noisy_batch[prompt_mask] = input_ids[prompt_mask]

    # compute answer lengths
    prompt_mask_int = prompt_mask.long()
    answer_lengths = (1 - prompt_mask_int).sum(dim=-1, keepdim=True)
    answer_lengths = answer_lengths.expand(-1, seq_len)

    # forward pass
    outputs = model(input_ids=noisy_batch).logits

    # identify masked positions
    masked_pos = noisy_batch == MASK_ID
    # gather predictions and targets
    pred = outputs[masked_pos]
    target = input_ids[masked_pos]

    # compute per-token loss
    loss_tok = F.cross_entropy(pred, target, reduction='none') / p_mask[masked_pos]
    # normalize by answer length
    loss = torch.sum(loss_tok / answer_lengths[masked_pos]) / batch_size
    return loss


def train(args):
    # device
    device = args.device

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16 if args.fp16 else torch.float32)
    model = model.to(device).train()

    # dynamic padding and random turn sampling for n-turn dialogues
    pad_id = tokenizer.eos_token_id
    def collate_fn(batch):
        processed_input_ids = []
        processed_prompt_lengths = []
        for item in batch:
            ids = item['input_ids']
            lengths = item['prompt_lengths']
            # if multi-turn, sample one turn pair
            if isinstance(ids[0], list):
                idx = random.randrange(len(ids))
                ids = ids[idx]
                lengths = lengths[idx]
            processed_input_ids.append(ids)
            processed_prompt_lengths.append(lengths)
        # pad all sequences to max length in batch using EOS token
        max_len = max(len(ids) for ids in processed_input_ids)
        batch_size = len(processed_input_ids)
        input_ids_tensor = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        prompt_lengths_tensor = torch.tensor(processed_prompt_lengths, dtype=torch.long)
        for i, ids in enumerate(processed_input_ids):
            input_ids_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return {'input_ids': input_ids_tensor, 'prompt_lengths': prompt_lengths_tensor}

    # load dataset
    ds = load_dataset('json', data_files={'train': args.train_file})['train']
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            prompt_lengths = batch['prompt_lengths'].to(device)

            loss = compute_sft_loss(model, input_ids, prompt_lengths)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1
            if global_step % args.log_steps == 0:
                print(f"Epoch {epoch}/{args.num_epochs} - Step {global_step}/{total_steps} - loss: {loss.item():.4f}")

        # save checkpoint at end of epoch
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    # final save
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for LLaDA")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="pretrained model identifier or path")
    parser.add_argument("--train_file", type=str, required=True,
                        help="training data file (.jsonl) with fields input_ids and prompt_lengths")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="where to save checkpoints and final model")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--fp16", action='store_true', help="use bf16 if set, else fp32")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
