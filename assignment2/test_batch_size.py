import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("SmolLM2-135M").cuda()

# test different BSZPERDEV
for bsz in [1, 2, 4, 8]:
    try:
        torch.cuda.reset_peak_memory_stats()
        
        # simulate one training step
        dummy_input = torch.randint(0, 1000, (bsz, 2048)).cuda() 
        outputs = model(dummy_input, labels=dummy_input)
        loss = outputs.loss
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"BSZPERDEV={bsz}: peak memory = {peak_memory:.2f} GB")
        
        # clear
        del dummy_input, outputs, loss
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"When BSZPERDEV={bsz}, OOM: {e}")
        break