name = 'gmflow_imagenet_k8_test'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        from_pretrained='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='bfloat16'),
    diffusion=dict(
        type='GMFlow',
        denoising=dict(
            type='GMDiTTransformer2DModel',
            num_gaussians=8,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            num_layers=28,
            sample_size=32,  # 256
            torch_dtype='bfloat16'),
        spectrum_net=dict(
            type='SpectrumMLP',
            base_size=(4, 32, 32),
            layers=[64, 8],
            torch_dtype='bfloat16'),
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=True),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    inference_only=True)

save_interval = 1000
must_save_interval = 20000  # interval to save regardless of max_keep_ckpts
eval_interval = 10000
work_dir = 'work_dirs/' + name

train_cfg = dict()
test_cfg = dict(
    latent_size=(4, 32, 32),
)

optimizer = {}
data = dict(
    workers_per_gpu=4,
    val=dict(
        type='ImageNet',
        test_mode=True),
    val_dataloader=dict(samples_per_gpu=125),
    test_dataloader=dict(samples_per_gpu=125),
    persistent_workers=True,
    prefetch_factor=64)
lr_config = dict()
checkpoint_config = dict()


guidance_scales = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.13, 0.16, 0.19, 0.23, 0.27, 0.33, 0.39, 0.47, 0.55, 0.65, 0.75]

methods = dict(
    gmode2=dict(
        output_mode='mean',
        sampler='FlowEulerODE',
        order=2),
    gmsde2=dict(
        output_mode='sample',
        sampler='GMFlowSDE',
        order=2)
)

evaluation = []

for step, substep in [(8, 16), (32, 4)]:
    for method_name, method_config in methods.items():
        for guidance_scale in guidance_scales:
            test_cfg_override = dict(
                orthogonal_guidance=1.0,
                guidance_scale=guidance_scale,
                num_timesteps=step)
            if 'gmode' in method_name:
                test_cfg_override.update(num_substeps=substep)
            test_cfg_override.update(method_config)
            evaluation.append(
                dict(
                    type='GenerativeEvalHook',
                    data='val',
                    prefix=f'{method_name}_g{guidance_scale:.2f}_step{step}',
                    sample_kwargs=dict(
                        test_cfg_override=test_cfg_override),
                    interval=eval_interval,
                    feed_batch_size=32,
                    viz_step=512,
                    metrics=[
                        dict(
                            type='InceptionMetrics',
                            resize=False,
                            reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'),
                    ],
                    viz_dir='viz/' + name + f'/{method_name}_g{guidance_scale:.2f}_step{step}',
                    save_best_ckpt=False))

total_iters = 200000
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

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
resume_from = None
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
