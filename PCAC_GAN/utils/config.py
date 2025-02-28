import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    training = config['training']
    training['lr_g'] = float(training['lr_g'])
    training['lr_d'] = float(training['lr_d'])
    training['batch_size'] = int(training['batch_size'])
    training['lambda_gp'] = float(training['lambda_gp'])
    training['lambda_rd'] = float(training['lambda_rd'])
    
    config['model']['quantization_bins'] = 2 ** config['training'].get('bit_depth', 8)
    return config
