const MF: [[i32; 3]; 6] = [
    [13107, 5243, 8066],
    [11916, 4660, 7490],
    [10082, 4194, 6554],
    [9362, 3647, 5825],
    [8192, 3355, 5243],
    [7282, 2893, 4559],
];
const V: [[i32; 3]; 6] = [
    [10, 16, 13],
    [11, 18, 14],
    [13, 20, 16],
    [14, 23, 18],
    [16, 25, 20],
    [18, 29, 23],
];

#[inline]
fn class(i: usize) -> usize {
    match ((i / 4) & 1, (i % 4) & 1) {
        (0, 0) => 0,
        (1, 1) => 1,
        _ => 2,
    }
}

fn fwd1(a: i32, b: i32, c: i32, d: i32) -> [i32; 4] {
    let (s0, s1, d0, d1) = (a + d, b + c, a - d, b - c);
    [s0 + s1, 2 * d0 + d1, s0 - s1, d0 - 2 * d1]
}

fn inv1(d: [i32; 4]) -> [i32; 4] {
    let e0 = d[0] + d[2];
    let e1 = d[0] - d[2];
    let e2 = (d[1] >> 1) - d[3];
    let e3 = d[1] + (d[3] >> 1);
    [e0 + e3, e1 + e2, e1 - e2, e0 - e3]
}

pub fn forward4x4(x: &[i32; 16]) -> [i32; 16] {
    let mut t = [0; 16];
    for r in 0..4 {
        let o = fwd1(x[r * 4], x[r * 4 + 1], x[r * 4 + 2], x[r * 4 + 3]);
        t[r * 4..r * 4 + 4].copy_from_slice(&o);
    }
    let mut w = [0; 16];
    for c in 0..4 {
        let o = fwd1(t[c], t[4 + c], t[8 + c], t[12 + c]);
        for r in 0..4 {
            w[r * 4 + c] = o[r];
        }
    }
    w
}

pub fn inverse4x4(d: &[i32; 16]) -> [i32; 16] {
    let mut t = [0; 16];
    for r in 0..4 {
        // rows first (spec order)
        let o = inv1([d[r * 4], d[r * 4 + 1], d[r * 4 + 2], d[r * 4 + 3]]);
        t[r * 4..r * 4 + 4].copy_from_slice(&o);
    }
    let mut out = [0; 16];
    for c in 0..4 {
        let o = inv1([t[c], t[4 + c], t[8 + c], t[12 + c]]);
        for r in 0..4 {
            out[r * 4 + c] = (o[r] + 32) >> 6;
        }
    }
    out
}

pub fn quantize(w: &[i32; 16], qp: u8, intra: bool) -> [i32; 16] {
    let qbits = 15 + (qp / 6) as u32;
    let f = (1i32 << qbits) / if intra { 3 } else { 6 };
    let mut z = [0; 16];
    for i in 0..16 {
        // |W| <= 4080 for 8-bit residuals; 4080 * 13107 fits in i32
        let m = (w[i].abs() * MF[(qp % 6) as usize][class(i)] + f) >> qbits;
        z[i] = if w[i] < 0 { -m } else { m };
    }
    z
}

pub fn dequantize(z: &[i32; 16], qp: u8) -> [i32; 16] {
    let mut d = [0; 16];
    for i in 0..16 {
        d[i] = (z[i] * V[(qp % 6) as usize][class(i)]) << (qp / 6);
    }
    d
}
