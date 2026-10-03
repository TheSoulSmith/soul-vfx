use std::{
    alloc::{self, Layout},
    ptr::NonNull,
    sync::atomic::*,
    thread::Thread,
};

struct CheckpointInner {
    count: usize,
    done: AtomicUsize,
    maintaining: AtomicUsize,
    threads: NonNull<Thread>,
    reached: AtomicBool,
}

impl CheckpointInner {
    fn new(count: usize) -> Self {
        if count == 0 {
            panic!("No checkpoints with zero uses allowed >:(")
        }
        let layout = Layout::array::<Thread>(count).expect("Allocation too large");
        Self {
            count,
            done: AtomicUsize::new(0),
            maintaining: AtomicUsize::new(0),
            threads: match NonNull::new(unsafe { alloc::alloc(layout) } as *mut Thread) {
                Some(p) => p,
                None => alloc::handle_alloc_error(layout),
            },
            reached: AtomicBool::new(false),
        }
    }
}

pub struct CheckpointInstance<T> {
    ptr: NonNull<CheckpointInner>,
    data_ptr: NonNull<T>,
    id: usize,
}

pub struct Checkpoint<T> {
    inner: NonNull<CheckpointInner>,
    data: NonNull<T>,
    id: usize,
}

impl<T: Copy> CheckpointInstance<T> {
    fn inner(&self) -> &CheckpointInner {
        unsafe { self.ptr.as_ref() }
    }
    pub fn reach(&self, t: T) {
        let num = self.inner().done.load(Ordering::SeqCst) + 1;
        if std::mem::size_of::<T>() != 0 {
            unsafe {
                std::ptr::write(self.data_ptr.as_ptr().add(self.id), t);
            }
        }
        if num == self.inner().count {
            //wake the others
            self.inner().reached.store(true, Ordering::SeqCst);
            for id in 0..(self.inner().count - 1) {
                if id != self.id {
                    unsafe { std::ptr::read(self.inner().threads.as_ptr().add(self.id)) }.unpark();
                }
            }
        } else {
            unsafe {
                std::ptr::write(
                    self.inner().threads.as_ptr().add(self.id),
                    std::thread::current(),
                );
                self.inner().done.fetch_add(1, Ordering::SeqCst);
                while !self.inner().reached.load(Ordering::SeqCst) {
                    std::thread::park();
                }
            }
        }
        //get the data from data_ptr, copy it, and return a Vec containing the copy
    }
}

impl<T: Copy> Checkpoint<T> {
    pub fn new(count: usize) -> Self {
        let boxed = Box::new(CheckpointInner::new(count));
        let inner = NonNull::new(Box::into_raw(boxed)).unwrap();
        let data = if std::mem::size_of::<T>() == 0 {
            NonNull::dangling()
        } else {
            let layout = Layout::array::<T>(count).unwrap();
            match NonNull::new(unsafe { alloc::alloc(layout) } as *mut T) {
                Some(p) => p,
                None => alloc::handle_alloc_error(layout),
            }
        };
        Self { inner, data, id: 0 }
    }
    fn inner(&self) -> &CheckpointInner {
        unsafe { self.inner.as_ref() }
    }
    pub fn checkpoint(&mut self) -> CheckpointInstance<T> {
        let id = self.id;
        self.id += 1;
        if self.id >= self.inner().count {
            panic!("Attempt to create more than the set number of checkpoint instances")
        }

        CheckpointInstance {
            ptr: self.inner,
            data_ptr: self.data,
            id,
        }
    }
}
