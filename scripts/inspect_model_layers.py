#!/usr/bin/env python3
"""
Utility script to extract EAGLE3 layer configuration from a pretrained model.
Use this to determine the --layer-ids argument for data generation.
"""

import argparse
import sys
from transformers import AutoConfig

# Register the config class
from src.speculators.models.eagle3 import Eagle3SpeculatorConfig  # noqa: F401

def main():
    parser = argparse.ArgumentParser(description="Extract EAGLE3 layer IDs")
    parser.add_argument("model_path", help="Path or HF ID of the pretrained EAGLE3 model")
    args = parser.parse_args()

    print(f"Loading config from {args.model_path}...")
    try:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        
        if hasattr(config, "eagle_aux_hidden_state_layer_ids") and config.eagle_aux_hidden_state_layer_ids:
            layers = config.eagle_aux_hidden_state_layer_ids
            print(f"\nFOUND LAYERS (Inputs): {layers}")
            
            # Try to determine the verifier's last layer to add it to the list
            verifier_last_layer = None
            if hasattr(config, "speculators_config") and config.speculators_config.verifier:
                verifier_path = config.speculators_config.verifier.name_or_path
                print(f"Verifier model: {verifier_path}")
                try:
                    v_config = AutoConfig.from_pretrained(verifier_path, trust_remote_code=True)
                    if hasattr(v_config, "text_config"):
                         v_config = v_config.text_config
                    
                    if hasattr(v_config, "num_hidden_layers"):
                        verifier_last_layer = v_config.num_hidden_layers - 1
                        print(f"Verifier has {v_config.num_hidden_layers} layers. Last layer index: {verifier_last_layer}")
                except Exception as e:
                    print(f"Could not load verifier config: {e}")
            
            gen_layers = list(layers)
            if verifier_last_layer is not None:
                if verifier_last_layer not in gen_layers:
                    gen_layers.append(verifier_last_layer)
                    gen_layers.sort()
            else:
                 print("\nIMPORTANT: You must also append the index of the last layer of the verifier model!")
                 print("e.g. if verifier has 32 layers (0-31), add 31.")

            # Formatted string for command line
            layers_str = " ".join(map(str, gen_layers))
            print(f"\nFor data_generation_offline.py, use:")
            print(f"    --layer-ids {layers_str}")
            
        else:
            print("\nWARNING: 'eagle_aux_hidden_state_layer_ids' not found in config.")
            print("This might not be an EAGLE3 model or uses an older config format.")
            print(f"Config keys: {list(config.__dict__.keys())}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
