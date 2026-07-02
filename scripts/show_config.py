import yaml
import sys
import os

def print_dict(d, prefix=''):
    for k, v in d.items():
        if isinstance(v, dict):
            print_dict(v, prefix + k + '.')
        else:
            if isinstance(v, list):
                v_str = ', '.join(map(str, v))
            else:
                v_str = str(v)
            print(f"| {prefix+k:<33} | {v_str:<25} |")

def main():
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("+" + "-"*35 + "+" + "-"*27 + "+")
        print(f"| {'Parameter':<33} | {'Value':<25} |")
        print("+" + "-"*35 + "+" + "-"*27 + "+")
        print_dict(config)
        print("+" + "-"*35 + "+" + "-"*27 + "+")
    except Exception as e:
        print(f"Error reading {config_path}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
