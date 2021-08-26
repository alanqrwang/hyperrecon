import torch

def generate_coefficients(samples, num_losses, range_restrict):
    '''Generates coefficients from samples.'''
    if range_restrict and num_losses == 2:
        assert samples.shape[1] == 1, 'num_hyperparams and loss mismatch'
        alpha = samples[:, 0]
        coeffs = torch.stack((1-alpha, alpha), dim=1)

    elif range_restrict and num_losses == 3:
        assert samples.shape[1] == 2, 'num_hyperparams and loss mismatch'
        alpha = samples[:, 0]
        beta = samples[:, 1]
        coeffs = torch.stack((alpha, (1-alpha)*beta, (1-alpha)*(1-beta)), dim=1)

    else:
        assert samples.shape[1] == num_losses, 'num_hyperparams and loss mismatch'
        coeffs = None
        for i in range(num_losses):
            coeffs = samples[:,i:i+1] if coeffs is None else torch.cat((coeffs, samples[:,i:i+1]), dim=1)
        coeffs = coeffs / torch.sum(samples, dim=1)

    return coeffs