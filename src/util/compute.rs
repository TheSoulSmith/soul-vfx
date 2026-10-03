use bytemuck::{NoUninit, Pod};
use flume::bounded;
use pollster::block_on;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};

pub struct ComputePipeline {
    pipeline: wgpu::ComputePipeline,
    workgroup_size: [u32; 3],
    encoder: CommandEncoder,
    device: Device,
    queue: Queue,
}

impl ComputePipeline {
    pub fn new(
        shader: ShaderModuleDescriptor,
        entry_point: String,
        workgroup_size: [u32; 3],
    ) -> Result<ComputePipeline, ()> {
        let (device, queue) = block_on(ComputePipeline::new_helper());
        let shader = device.create_shader_module(shader);

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Test Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some(&entry_point),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });

        Ok(ComputePipeline {
            pipeline,
            workgroup_size,
            encoder,
            device,
            queue,
        })
    }

    async fn new_helper() -> (Device, Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::default(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        (device, queue)
    }

    pub fn dispatch<const N: usize>(
        &mut self,
        bind_group_entriess: Vec<[BindGroupEntry; N]>,
        size: [u32; 3],
    ) -> Result<(), ()> {
        let mut pass = self.encoder.begin_compute_pass(&Default::default());
        let (wx, wy, wz) = (
            size[0].div_ceil(self.workgroup_size[0]),
            size[1].div_ceil(self.workgroup_size[1]),
            size[2].div_ceil(self.workgroup_size[2]),
        );

        for bind_group_entries in bind_group_entriess {
            pass.set_pipeline(&self.pipeline); // this is the beginning of the part that needs to be adapted for multipass
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &bind_group_entries,
            });
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wx, wy, wz);
        }

        Ok(())
    }

    pub fn create_input_buffer<A: NoUninit>(&self, data: &[A]) -> Buffer {
        self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
        })
    }

    pub fn create_buffer<A>(&self, size: u64) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some("Buffer"),
            size: size * (std::mem::size_of::<A>() as u64),
            usage: BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    pub fn finish<T: Pod>(
        &mut self,
        reqs: &[Buffer],
    ) -> Vec<impl Future<Output = Result<Vec<T>, ()>>> {
        let mut enc = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });
        let t_buf: Vec<Buffer> = reqs
            .iter()
            .map(|buffer| {
                let t_buffer = self.device.create_buffer(&BufferDescriptor {
                    label: Some("Temporary Buffer"),
                    size: buffer.size(),
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                self.encoder
                    .copy_buffer_to_buffer(&buffer, 0, &t_buffer, 0, buffer.size());
                t_buffer
            })
            .collect();
        std::mem::swap(&mut enc, &mut self.encoder);
        self.queue.submit([enc.finish()]);
        let t_device = &self.device;
        t_buf
            .into_iter()
            .map(|t_buffer| async move {
                let (tx, rx) = bounded(1);
                t_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                    tx.send(result).unwrap()
                });
                match t_device.poll(PollType::wait_indefinitely()) {
                    Ok(_) => {}
                    Err(_) => return Err(()),
                };
                match rx.recv_async().await {
                    Ok(Ok(_)) => {}
                    Ok(Err(_)) => return Err(()),
                    Err(_) => return Err(()),
                };
                match t_buffer.get_mapped_range(..) {
                    Ok(t) => Ok(Vec::from(bytemuck::cast_slice(&t))),
                    Err(_) => Err(()),
                }
            })
            .collect()
    }
}
