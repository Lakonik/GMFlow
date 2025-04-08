name = 'gmflow_checkerboard_k64'

model = dict(
    type='Diffusion2D',
    diffusion=dict(
        type='GMFlow',
        denoising=dict(
            type='GMFlowMLP2DDenoiser',
            num_gaussians=64),
        flow_loss=dict(
            type='GMFlowNLLLoss',
            log_cfgs=dict(type='quartile', prefix_name='loss_trans', total_timesteps=1000),
            data_info=dict(
                pred_means='means',
                target='x_t_low',
                pred_logstds='logstds',
                pred_logweights='logweights'),
            weight_scale=1.0),
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=False),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
)

save_interval = 20000
must_save_interval = 40000  # interval to save regardless of max_keep_ckpts
work_dir = 'work_dirs/' + name

train_cfg = dict(
    trans_ratio=0.9,
)
test_cfg = dict(
    output_mode='sample',
    sampler='GMFlowSDE',
    num_timesteps=8,
    order=2
)

optimizer = {
    'diffusion': dict(type='AdamW', lr=2e-4, weight_decay=0.0)
}
data = dict(
    workers_per_gpu=8,
    train=dict(type='CheckerboardData', scale=4),
    train_dataloader=dict(samples_per_gpu=4096),
    val=dict(type='CheckerboardData', scale=4),
    val_dataloader=dict(samples_per_gpu=4096),
    test_dataloader=dict(samples_per_gpu=4096),
    persistent_workers=True,
    prefetch_factor=1024)
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001)
checkpoint_config = dict(
    interval=save_interval,
    must_save_interval=must_save_interval,
    by_epoch=False,
    max_keep_ckpts=1,
    out_dir='checkpoints/')

evaluation = []

total_iters = 200000
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHookMod',
        module_keys=('diffusion_ema', ),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='karras',
        momentum_cfg=dict(gamma=7.0),
        priority='VERY_HIGH'),
]

# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunnerMod',
    is_dynamic_ddp=False,
    pass_training_status=True,
    ckpt_trainable_only=True,
    ckpt_fp16_ema=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = f'checkpoints/{name}/latest.pth'  # resume by default
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
