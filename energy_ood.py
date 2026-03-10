import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def compute_energy(model, tokenizer, text):
    """Compute the energy score (negative log-likelihood) for a given text input.

    Args:
        model: A GPT2LMHeadModel (or compatible) language model.
        tokenizer: The tokenizer corresponding to the model.
        text (str): The input text to score.

    Returns:
        float: The energy score (total negative log-likelihood over all tokens).
    """
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        # Energy: negative log-likelihood
        energy = loss.item() * input_ids.size(1)
    return energy


def calibrate_threshold(energies, percentile=95):
    """Calculate the energy threshold for OOD detection based on in-distribution samples.

    Args:
        energies (list[float]): Energy scores computed from in-distribution samples.
        percentile (float): The percentile used to set the threshold (default: 95).

    Returns:
        float: The calibrated energy threshold above which inputs are flagged as OOD.
    """
    return np.percentile(energies, percentile)


# Example usage
if __name__ == "__main__":
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    texts_in = ["This is a standard Wikipedia sentence."]  # in-distribution
    texts_ood = ["def foo(x): return x*x"]                 # out-of-distribution (e.g. code)

    energies_in = [compute_energy(model, tokenizer, text) for text in texts_in]
    energies_ood = [compute_energy(model, tokenizer, text) for text in texts_ood]

    threshold = calibrate_threshold(energies_in)
    detections = [energy > threshold for energy in energies_ood]
    print(f"OOD detected: {sum(detections)}/{len(detections)}")
