use std::{
    sync::{Arc, Barrier, Mutex},
    task::Poll::{Pending, Ready},
};

struct CheckpointInner<T: Copy> {
    barrier: Barrier,
    values: Mutex<Vec<Option<T>>>,
}

/// An instance of a checkpoint, this contains the methods for actually reaching the checkpoint,
/// as well as submitting and retrieving the value(s)
pub struct CheckpointInstance<T: Copy> {
    inner: Arc<CheckpointInner<T>>,
    id: usize,
}

/// The structure that allows creation of a checkpoint and the generation of instances
pub struct Checkpoint<T: Copy> {
    inner: Arc<CheckpointInner<T>>,
    count: usize,
    id: usize,
}

impl<T: Copy> CheckpointInstance<T> {
    /// Store this participant's value and wait for every participant to arrive.
    ///
    /// Returns the values for this round in participant-creation order.
    pub fn reach(&mut self, value: T) -> Vec<T> {
        {
            let mut values = self
                .inner
                .values
                .lock()
                .expect("checkpoint values lock poisoned");
            values[self.id] = Some(value);
        }

        self.inner.barrier.wait();

        let values = {
            let values = self
                .inner
                .values
                .lock()
                .expect("checkpoint values lock poisoned");
            values
                .iter()
                .map(|value| value.expect("participant must provide a value before release"))
                .collect()
        };

        // Keep the next round from replacing values until all participants have copied this round.
        self.inner.barrier.wait();
        values
    }
}

impl<T: Copy> Checkpoint<T> {
    /// Create a reusable checkpoint for exactly `count` participants.
    pub fn new(count: usize) -> Self {
        assert!(count > 0, "checkpoint must have at least one participant");
        Self {
            inner: Arc::new(CheckpointInner {
                barrier: Barrier::new(count),
                values: Mutex::new(vec![None; count]),
            }),
            count,
            id: 0,
        }
    }

    /// Allocate the next participant. Exactly `count` instances can be created.
    pub fn checkpoint(&mut self) -> CheckpointInstance<T> {
        assert!(
            self.id < self.count,
            "attempt to create more than the set number of checkpoint instances"
        );
        let id = self.id;
        self.id += 1;
        CheckpointInstance {
            inner: Arc::clone(&self.inner),
            id,
        }
    }
}

/// Allows combining a [Vec] of [Future]s into a single future,
/// primarily useful for getting the combined outputs of many threads
pub struct CombinedFuture<T: Future> {
    main: Vec<std::pin::Pin<Box<T>>>,
    output: Vec<T::Output>,
    index: usize,
}

impl<T: Future> From<Vec<T>> for CombinedFuture<T> {
    fn from(value: Vec<T>) -> Self {
        CombinedFuture {
            main: value.into_iter().map(Box::pin).collect(),
            index: 0,
            output: Vec::new(),
        }
    }
}

impl<T: Future> Future for CombinedFuture<T> {
    type Output = Vec<T::Output>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = unsafe { self.get_unchecked_mut() };
        if this.index >= this.main.len() {
            return Ready(std::mem::take(&mut this.output));
        }

        match this.main[this.index].as_mut().poll(cx) {
            Ready(out) => {
                this.index += 1;
                this.output.push(out);
                if this.index >= this.main.len() {
                    Ready(std::mem::take(&mut this.output))
                } else {
                    Pending
                }
            }
            Pending => Pending,
        }
    }
}
