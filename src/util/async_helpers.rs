use std::{
    sync::{Arc, Barrier, Mutex},
    task::Poll::{Pending, Ready},
};

struct CheckpointInner<T: Copy> {
    barrier: Barrier,
    values: Mutex<Vec<Option<T>>>,
}

pub struct CheckpointInstance<T: Copy> {
    inner: Arc<CheckpointInner<T>>,
    id: usize,
}

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

pub struct CombinedFuture<T: Future> {
    main: Vec<std::pin::Pin<Box<T>>>,
    output: Vec<T::Output>,
    index: usize,
}

impl<T: Future> CombinedFuture<T> {
    pub fn from(main: Vec<T>) -> CombinedFuture<T> {
        CombinedFuture {
            main: main.into_iter().map(Box::pin).collect(),
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

#[cfg(test)]
mod tests {
    use super::Checkpoint;
    use std::thread;

    #[test]
    fn one_participant_can_reach_repeatedly() {
        let mut checkpoint = Checkpoint::new(1);
        let mut participant = checkpoint.checkpoint();

        assert_eq!(participant.reach(10), vec![10]);
        assert_eq!(participant.reach(20), vec![20]);
    }

    #[test]
    fn participants_exchange_values_across_repeated_rounds() {
        const PARTICIPANTS: usize = 4;
        const ROUNDS: usize = 5;

        let mut checkpoint = Checkpoint::new(PARTICIPANTS);
        let instances = (0..PARTICIPANTS)
            .map(|_| checkpoint.checkpoint())
            .collect::<Vec<_>>();

        let handles = instances
            .into_iter()
            .enumerate()
            .map(|(id, mut instance)| {
                thread::spawn(move || {
                    (0..ROUNDS)
                        .map(|round| {
                            let value = round * 100 + id;
                            (0..PARTICIPANTS)
                                .map(|participant_id| round * 100 + participant_id)
                                .zip(instance.reach(value))
                                .all(|(expected, actual)| expected == actual)
                        })
                        .all(|round_matches| round_matches)
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            assert!(handle.join().expect("participant thread panicked"));
        }
    }

    #[test]
    fn checkpoint_owner_can_be_dropped_before_instances() {
        let mut participant = {
            let mut checkpoint = Checkpoint::<usize>::new(1);
            checkpoint.checkpoint()
        };

        assert_eq!(participant.reach(42), vec![42]);
    }

    #[test]
    #[should_panic(expected = "checkpoint must have at least one participant")]
    fn rejects_zero_participants() {
        Checkpoint::<usize>::new(0);
    }

    #[test]
    fn permits_exactly_the_configured_number_of_instances() {
        let mut checkpoint = Checkpoint::<usize>::new(2);
        let _first = checkpoint.checkpoint();
        let _second = checkpoint.checkpoint();
    }

    #[test]
    #[should_panic(expected = "attempt to create more than the set number")]
    fn rejects_more_than_the_configured_number_of_instances() {
        let mut checkpoint = Checkpoint::<usize>::new(1);
        let _first = checkpoint.checkpoint();
        checkpoint.checkpoint();
    }
}
