import torch
import os

def save_checkpoint(time, model, optimizer, scheduler, config):
    if time == 400:
        return
    
    ckpt_name = config.arch + '_Time:' + str(time) + \
                '_R1:' + str(config.nr_residual_blocks_1) + \
                '_R2:' + str(config.nr_residual_blocks_2) + \
                '_R3:' + str(config.nr_residual_blocks_3) + \
                '_B1:' + str(config.res_branches_1) + \
                '_B2:' + str(config.res_branches_2) + \
                '_B3:' + str(config.res_branches_3) + \
                '_Te:' + str(config.T_e) + \
                '_Budget:' + str(config.budget) + \
                '.pt'

    latest = config.arch + '_latest' + \
             '_R1:' + str(config.nr_residual_blocks_1) + \
             '_R2:' + str(config.nr_residual_blocks_2) + \
             '_R3:' + str(config.nr_residual_blocks_3) + \
             '_B1:' + str(config.res_branches_1) + \
             '_B2:' + str(config.res_branches_2) + \
             '_B3:' + str(config.res_branches_3) + \
             '_Te:' + str(config.T_e) + \
             '_Budget:' + str(config.budget) + \
             '.pt'

    if not os.path.exists(config.save):
        os.makedirs(config.save)

    torch.save({
       'state': model.state_dict(),
    }, open(os.path.join(config.save, ckpt_name), 'wb'))
    
    torch.save({
        'time': time,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler,
        'checkpoint': ckpt_name,
    }, open(os.path.join(config.save, latest), 'wb'))

    print('=> Model {} saved to {}'.format(ckpt_name, config.save))


def load_latest(config):
    if config.resume == None:
        return None
 
    latest = config.arch + '_latest' + \
             '_R1:' + str(config.nr_residual_blocks_1) + \
             '_R2:' + str(config.nr_residual_blocks_2) + \
             '_R3:' + str(config.nr_residual_blocks_3) + \
             '_B1:' + str(config.res_branches_1) + \
             '_B2:' + str(config.res_branches_2) + \
             '_B3:' + str(config.res_branches_3) + \
             '_Te:' + str(config.T_e) + \
             '_Budget:' + str(config.budget) + \
             '.pt'

    latest_path = os.path.join(config.resume, latest)
    if not os.path.exists(latest_path):
        return None

    print('=> Loading checkpoint ' + latest_path)
    latest = torch.load(latest_path)
    return latest

