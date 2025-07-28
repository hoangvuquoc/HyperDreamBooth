from safetensors.torch import load_file

state_dict = load_file("output/checkpoint-2000/pytorch_lora_weights.safetensors")
print(state_dict.keys())
